# Mini Crosswords — Aggregated Analysis

_Auto-generated from `results/crosswords/runs/` by `code/analysis/crosswords/build_report.py`._
_All numbers are means across the available runs (n_runs in each row); SE is the standard error of the mean._

---

## Part 1 — Paper reproduction (n=20 paper split, gemini-3.1-flash-lite)

| Method | Letter (Paper / Ours) | Word (Paper / Ours) | Game (Paper / Ours) | Steps (Paper / Ours) | API Calls (Paper / Ours) |
|---|---|---|---|---|---|
| IO (n_runs=3) | 38.7 / 45.4 | 14.0 / 18.3 | 0.0 / 0.0 | 1.0 / 1.0 | 1.0 / 10.0 |
| CoT (n_runs=3) | 40.6 / 38.8 | 15.6 / 15.0 | 1.0 / 0.3 | 1.0 / 1.0 | 1.0 / 10.0 |
| ToT (unbatched) (n_runs=5) | 78.0 / 77.4 | 60.0 / 59.1 | 20.0 / 33.0 | 27.8 / 24.3 | 499.7 / 461.9 |
| → + best state (n_runs=5) | 67.5 / 80.0 | 60.0 / 61.3 | 35.0 / 33.0 | 19.9 / 24.3 | 225.7 / 461.9 |
| → − prune (n_runs=2) | 41.5 / 70.7 | 5.0 / 47.5 | 5.0 / 17.5 | 18.6 / 17.5 | 103.2 / 87.8 |
| → − backtrack (n_runs=2) | 20.0 / 39.3 | 5.0 / 23.5 | 5.0 / 5.0 | 3.5 / 3.6 | 26.0 / 25.5 |

---

## Part 2 — Unbatched vs Half vs Full batch (mean across runs)

| Level | n_runs | Letter % | Word % | Game % (± SE) | Steps | API calls | Wall-time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unbatched | 5 | 77.4 | 59.1 | 33.0 ± 2.5 | 24.3 | 461.9 | 555.7 |
| Half Basic | 5 | 77.6 | 55.6 | 25.0 ± 1.6 | 37.2 | 413.5 | 499.5 |
| Full Basic | 5 | 72.7 | 50.6 | 24.0 ± 2.4 | 26.9 | 157.5 | 226.4 |

---

## Part 3.1 — Focused (top_k sweep)

### Full batch

| Variant | n_runs | Game % (± SE) | Range | Letter % | Word % | Calls | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| full_basic | 5 | 24.0 ± 2.4 | [20.0, 30.0] | 72.7 | 50.6 | 157.5 | 26.9 |
| full_focused (k=0) | 4 | 27.5 ± 3.2 | [20.0, 35.0] | 74.8 | 54.0 | 117.7 | 19.6 |
| full_focused_k3 | 4 | 26.2 ± 3.1 | [20.0, 35.0] | 71.8 | 50.1 | 142.7 | 23.7 |
| full_focused_k5 | 4 | 20.0 ± 2.0 | [15.0, 25.0] | 70.1 | 47.2 | 109.2 | 18.2 |
| full_focused_k7 | 4 | 21.2 ± 3.8 | [10.0, 25.0] | 75.0 | 52.2 | 115.9 | 19.3 |

### Half batch

| Variant | n_runs | Game % (± SE) | Range | Letter % | Word % | Calls | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| half_basic | 5 | 25.0 ± 1.6 | [20.0, 30.0] | 77.6 | 55.6 | 413.5 | 37.2 |
| half_focused (k=0) | 4 | 36.2 ± 2.4 | [30.0, 40.0] | 80.1 | 60.5 | 404.8 | 28.1 |
| half_focused_k3 | 4 | 30.0 ± 5.4 | [20.0, 45.0] | 73.5 | 53.2 | 509.9 | 35.2 |
| half_focused_k5 | 4 | 28.8 ± 1.2 | [25.0, 30.0] | 76.5 | 55.4 | 472.5 | 32.2 |
| half_focused_k7 | 4 | 36.2 ± 3.8 | [30.0, 45.0] | 82.2 | 63.4 | 420.6 | 29.6 |

### Δ (focused − basic)

| | Δ Game | Δ Calls | Δ Steps |
|---|---:|---:|---:|
| full_focused vs full_basic | +3.5 | -39.8 | -7.4 |
| half_focused vs half_basic | +11.2 | -8.8 | -9.0 |

---

## Part 3.2 — Verified (basic vs verified, focused alone vs focused+verified)

### Variance comparison (basic vs verified)

| Level | basic SE | verified SE | basic mean | verified mean |
|---|---:|---:|---:|---:|
| Full batch | 2.4 | 3.1 | 24.0 | 23.8 |
| Half batch | 1.6 | 2.4 | 25.0 | 28.8 |

### Focused (Step 1) vs Focused + Verified (Step 2)

| Config | Focused (Step 1) | Focused + Verified (Step 2) | Δ Game | Δ SE | Δ Calls |
|---|---:|---:|---:|---:|---:|
| Full · k=0 | 27.5 ± 3.2 | 23.3 ± 3.3 | -4.2 | +0.1 | +109.9 |
| Full · k=3 | 26.2 ± 3.1 | 25.0 ± 2.9 | -1.2 | -0.3 | +43.4 |
| Half · k=0 | 36.2 ± 2.4 | 31.7 ± 1.7 | -4.6 | -0.7 | +134.1 |
| Half · k=7 | 36.2 ± 3.8 | 27.5 ± 2.5 | -8.8 | -1.2 | +40.9 |

---

## Part 4 — Unbatched + Cache

| Metric | Unbatched (n=5) | Unbatched + cache (n=3) | Δ |
|---|---:|---:|---:|
| Game accuracy % | 33.0 ± 2.5 | 40.0 ± 5.0 | +7.0 |
| Game range | [25.0, 40.0] | [30.0, 45.0] | — |
| Letter accuracy % | 77.4 ± 1.6 | 77.1 ± 0.8 | -0.3 |
| Word accuracy % | 59.1 ± 1.5 | 61.5 ± 2.5 | +2.4 |
| API calls / puzzle | 461.9 ± 48.1 | 347.2 ± 39.0 | -114.7 |
|   · propose | 121.5 ± 6.5 | 135.2 ± 18.2 | +13.6 |
|   · value | 340.4 ± 44.3 | 212.0 ± 21.0 | -128.3 |
| Steps | 24.3 ± 1.3 | 27.0 ± 3.6 | +2.7 |
| Runtime (s) | 555.7 ± 94.3 | 373.0 ± 63.8 | -182.6 |

---

## Cost summary across the study

| Tier | Config | Game % | Calls | Cost vs paper-unbatched |
|---|---|---:|---:|---:|
| Cost-bound | full_focused (k=0) | 27.5 | 117.7 | 24% |
| Mid | half_focused (k=0) | 36.2 | 404.8 | 81% |
| Mid | half_focused_k7 | 36.2 | 420.6 | 84% |
| New | unbatched_cache | 40.0 | 347.2 | 69% |
| Paper-strict | unbatched | 33.0 | 461.9 | 92% |
