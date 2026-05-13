"""Async wrapper around the Gemini API with retries, rate limiting, cost tracking.

Why async? The ToT search is embarrassingly parallel within each BFS step:
fanning 5 frontier states x 8 propose calls x 3 value votes wants ~120
concurrent requests. With sync calls you're bottlenecked on per-minute
rate limits; with `asyncio.gather` you're bottlenecked on the network.

Why thinking_level=MINIMAL by default? Gemini 3.x is a "thinking" model that
runs internal chain-of-thought before answering. Letting it think defeats
the point of ToT (which externalizes reasoning into the search structure)
and makes IO/CoT vs ToT comparisons unfair. MINIMAL gets behavior closest
to the GPT-4 setup the original paper used.

Why a rate limiter? Free-tier Gemini caps requests per minute (15 RPM for
Flash-Lite) AND per day (1000 RPD). We enforce the per-minute cap inline so
asyncio.gather doesn't burst past it; the per-day cap is just for the user
to be aware of when planning runs.
"""

from __future__ import annotations

import asyncio
import os
from collections import deque
from dataclasses import dataclass

from google import genai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)


# Pricing (USD per 1M tokens) -- update as Gemini pricing changes.
# Source: https://ai.google.dev/pricing
PRICING = {
    "gemini-3.1-flash-lite-preview": {"input": 0.25, "output": 1.50},
    "gemini-3-flash-preview":        {"input": 0.50, "output": 3.00},
    "gemini-1.5-flash":              {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro":                {"input": 1.25, "output": 5.00},
    "gemini-2.5-flash":              {"input": 0.30, "output": 2.50},
}


@dataclass
class TokenUsage:
    """Accumulator for prompt/completion tokens across all calls."""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion

    def cost_usd(self, model: str) -> float:
        rates = PRICING.get(model, PRICING["gemini-3.1-flash-lite-preview"])
        return (self.prompt_tokens     * rates["input"]  / 1e6 +
                self.completion_tokens * rates["output"] / 1e6)


# --------------------------------------------------------------------
# Rate limiter
# --------------------------------------------------------------------

class AsyncRateLimiter:
    """Sliding-window rate limiter: at most `rate` calls in any `period` seconds.

    Each `await acquire()` either returns immediately (if the window has room)
    or sleeps until the oldest call falls out of the window. Safe under
    asyncio.gather -- the lock serializes admission decisions.
    """

    def __init__(self, rate: int, period: float = 60.0):
        self.rate = rate
        self.period = period
        self.timestamps: deque[float] = deque()
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self.lock:
                now = asyncio.get_event_loop().time()
                # Drop timestamps older than the window
                while self.timestamps and self.timestamps[0] <= now - self.period:
                    self.timestamps.popleft()

                if len(self.timestamps) < self.rate:
                    self.timestamps.append(now)
                    return

                # When does the oldest timestamp leave the window?
                wait = self.period - (now - self.timestamps[0]) + 0.05

            await asyncio.sleep(wait)


# --------------------------------------------------------------------
# Gemini wrapper
# --------------------------------------------------------------------

class GeminiWrapper:
    """Minimal async client.

    Public API is just `await wrapper.generate(prompt, temperature, n)`,
    which returns a list of `n` string completions.
    """

    # Models that accept a thinking_config parameter (others reject it).
    _THINKING_MODELS_PREFIX = ("gemini-3-", "gemini-3.")

    def __init__(self, model: str = "gemini-3.1-flash-lite-preview",
                 api_key: str | None = None,
                 thinking_level: str | None = "MINIMAL",
                 rpm: int | None = 15,
                 vertex: bool = False,
                 project: str | None = None,
                 location: str | None = None):
        """
        Args:
            model:           Gemini model identifier.
            api_key:         API key for AI Studio (defaults to $GEMINI_API_KEY).
                             Ignored when use_vertex=True.
            thinking_level:  "MINIMAL" / "LOW" / "MEDIUM" / "HIGH" / None
                             (None = use model default, which is HIGH for Gemini 3).
                             Ignored for non-thinking models like 1.5 Flash.
            rpm:             Requests-per-minute cap. Default 15 matches the
                             free-tier AI Studio limit. Pass None to disable
                             (Vertex doesn't have a tight RPM limit; you may
                             still want a conservative limit on preview models).
            use_vertex:      Route through Vertex AI instead of AI Studio.
                             Use this to bill against GCP credit.
            project:         GCP project ID for Vertex (defaults to $GCP_PROJECT).
                             Ignored when use_vertex=False.
            location:        Vertex region. If None, auto-picked: "global" for
                             Gemini 3.x preview models (which are global-only),
                             "us-central1" for older models. Ignored when
                             use_vertex=False.
        """
        self.model = model
        self.thinking_level = thinking_level

        if vertex:
            # Gemini 3.x preview models are only available on the global endpoint.
            if location is None:
                location = ("global" if model.startswith(self._THINKING_MODELS_PREFIX)
                            else "us-central1")
            gcp_project = project or os.environ.get("GCP_PROJECT")
            if not gcp_project:
                raise ValueError("When using --vertex, you must set the GCP_PROJECT environment variable (e.g., export GCP_PROJECT='your-project-id').")

            print(f"🔗 Connecting to Vertex AI (Project: {gcp_project}, Location: {location})")
            self.client = genai.Client(
                vertexai=True,
                project=gcp_project,
                location=location,
            )
        else:
            print("🔗 Connecting to Google AI Studio")
            self.client = genai.Client(
                api_key=api_key or os.environ.get("GEMINI_API_KEY")
            )

        self.usage = TokenUsage()
        self.rate_limiter = AsyncRateLimiter(rate=rpm) if rpm else None
    def _supports_thinking(self) -> bool:
        return self.model.startswith(self._THINKING_MODELS_PREFIX)

    def _build_config(self, temperature: float, max_tokens: int) -> dict:
        cfg: dict = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if self.thinking_level and self._supports_thinking():
            cfg["thinking_config"] = {"thinking_level": self.thinking_level}
        return cfg

    @retry(
        wait=wait_exponential(multiplier=2,min=2, max=60),
        stop=stop_after_attempt(8),
        reraise=True,
    )
    async def _single_call(self, prompt: str, temperature: float,
                           max_tokens: int) -> str:
        # Acquire BEFORE the API call -- and inside the retry, so failed
        # attempts also count toward the rate limit (which is what Gemini does).
        if self.rate_limiter is not None:
            await self.rate_limiter.acquire()

        resp = await self.client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self._build_config(temperature, max_tokens),
        )

        # Track usage if the API returned it
        meta = getattr(resp, "usage_metadata", None)
        if meta is not None:
            self.usage.add(
                getattr(meta, "prompt_token_count", 0) or 0,
                getattr(meta, "candidates_token_count", 0) or 0,
            )

        # `.text` raises if the response was blocked or empty -- coerce to ""
        try:
            return resp.text or ""
        except (ValueError, AttributeError):
            return ""

    async def generate(self, prompt: str, temperature: float = 0.7,
                       n: int = 1, max_tokens: int = 1000) -> list[str]:
        """Return `n` independent completions, fan-out via asyncio.gather.

        Gemini supports `candidate_count` for multi-sampling but it isn't
        available on every model -- gather is portable and gives the same
        diversity (with a slightly higher per-token cost from re-sending
        the prompt).
        """
        tasks = [self._single_call(prompt, temperature, max_tokens)
                 for _ in range(n)]
        return await asyncio.gather(*tasks)


# --------------------------------------------------------------------
# Cost / time estimation helpers (used by run.py at startup)
# --------------------------------------------------------------------

def estimate_calls_per_puzzle(method: str, task: str = "24", b: int = 5, 
                              n_votes: int = 3, avg_depth: int = 15) -> int:
    """
    Estimate the upper-bound of API calls per puzzle.
    
    Args:
        method: 'io', 'cot', or 'tot'.
        task: '24' or 'crosswords'.
        b: Beam width (for Game of 24 BFS).
        n_votes: Number of votes per evaluation.
        avg_depth: Estimated search depth (for Crosswords DFS).
    """
    if method in ("io", "cot"):
        return 1
    
    if task == "24":
        # ToT BFS logic for Game of 24: 3 expansions
        propose_calls = 1 + b + b
        value_calls = 3 * n_votes
        return propose_calls + value_calls
    
    elif task == "crosswords":
        # ToT DFS logic for Crosswords: 1 Propose + 1 Batched Value per step
        # Assuming an average exploration depth of avg_depth
        calls_per_step = 2 
        return avg_depth * calls_per_step
    
    return 10  # Default fallback