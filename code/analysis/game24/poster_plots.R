# ============================================================
# Tree of Thoughts — Poster Plots (all 5)
# CS 4782 Final Project, Cornell Spring 2026
#
# Run: source("poster_plots.R")   (set wd to project root)
# Output: plots/ directory, PDF + PNG for each
# ============================================================

library(ggplot2)
library(gridExtra)  # for Plot 1b table
library(grid)       # for tableGrob styling

dir.create("plots", showWarnings = FALSE)

# ============================================================
# COLOUR PALETTE
# ============================================================
CHARCOAL_BLUE <- "#233d4d"   # Gemini 3.1 / primary
PUMPKIN_SPICE <- "#fe7f2d"   # Gemini 2.5 / secondary
GOLDEN_POLLEN <- "#fcca46"   # GPT-4 paper reference
MUTED_OLIVE   <- "#a1c181"   # "good" conditions (un-batched, tiebreaker)
SEAGRASS      <- "#619b8a"   # neutral / batched conditions

# ============================================================
# SHARED POSTER THEME
# ============================================================
poster_theme <- theme_minimal(base_size = 16) +
  theme(
    plot.title       = element_text(face = "bold", size = 18, hjust = 0.5),
    plot.subtitle    = element_text(color = "gray40", size = 12, hjust = 0.5),
    legend.position  = "top",
    legend.text      = element_text(size = 12),
    legend.key.size  = unit(0.5, "cm"),
    axis.text.x      = element_text(size = 13, face = "bold"),
    axis.text.y      = element_text(size = 11),
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    plot.margin      = margin(10, 15, 10, 10)
  )


# ############################################################
# PLOT 0 — IO vs CoT vs ToT un-batched (b=5)
# ############################################################
cat("Building Plot 0...\n")

p0_df <- data.frame(
  method = rep(c("IO", "CoT", "ToT (b=5)"), each = 3),
  model  = rep(c("Paper (GPT-4)", "Gemini 2.5", "Gemini 3.1"), times = 3),
  rate   = c(7.3, 17, 16,     # IO
             4.0, 25, 84,     # CoT
             74,  87, 96)     # ToT un-batched
)
p0_df$method <- factor(p0_df$method, levels = c("IO", "CoT", "ToT (b=5)"))
p0_df$model  <- factor(p0_df$model,  levels = c("Paper (GPT-4)", "Gemini 2.5", "Gemini 3.1"))

p0 <- ggplot(p0_df, aes(x = method, y = rate, fill = model)) +
  geom_col(position = position_dodge(width = 0.78), width = 0.7) +
  geom_text(aes(label = paste0(rate, "%")),
            position = position_dodge(width = 0.78),
            vjust = -0.5, size = 4.5, fontface = "bold") +
  scale_fill_manual(values = c(
    "Paper (GPT-4)" = GOLDEN_POLLEN,
    "Gemini 2.5"    = PUMPKIN_SPICE,
    "Gemini 3.1"    = CHARCOAL_BLUE
  )) +
  scale_y_continuous(limits = c(0, 110), breaks = seq(0, 100, 20),
                     labels = function(x) paste0(x, "%"), expand = c(0, 0)) +
  labs(title    = "Game of 24: Success Rate by Method",
       subtitle = "Un-batched ToT (b = 5), n = 100 puzzles",
       x = NULL, y = NULL, fill = NULL) +
  poster_theme

ggsave("plots/plot0_main_results.pdf", p0, width = 9, height = 6)
ggsave("plots/plot0_main_results.png", p0, width = 9, height = 6, dpi = 300)
cat("  -> plots/plot0_main_results.pdf\n")


# ############################################################
# PLOT 1a — Batched vs Un-batched Accuracy
# ############################################################
cat("Building Plot 1a...\n")

p1a_df <- data.frame(
  group = rep(c("Gemini 2.5", "Gemini 3.1"), each = 2),
  eval  = rep(c("Batched", "Un-batched"), times = 2),
  rate  = c(15, 87,   # Gem 2.5 b=5
            88, 96)   # Gem 3.1 b=5
)
p1a_df$group <- factor(p1a_df$group, levels = c("Gemini 2.5", "Gemini 3.1"))
p1a_df$eval <- factor(p1a_df$eval, levels = c("Batched", "Un-batched"))

# Compute deltas for annotation
p1a_deltas <- data.frame(
  group = factor(c("Gemini 2.5", "Gemini 3.1"), levels = levels(p1a_df$group)),
  delta = c(87-15, 96-88),
  y_pos = c(87, 96)
)

p1a <- ggplot(p1a_df, aes(x = group, y = rate, fill = eval)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.65) +
  geom_text(aes(label = paste0(rate, "%")),
            position = position_dodge(width = 0.75),
            vjust = -0.5, size = 4.5, fontface = "bold") +
  # Delta annotations above the un-batched bars
  geom_text(data = p1a_deltas,
            aes(x = group, y = y_pos + 8,
                label = paste0("+", delta, "pp"),
                fill = NULL),
            color = MUTED_OLIVE, fontface = "bold.italic", size = 3.8,
            nudge_x = 0.19) +
  scale_fill_manual(values = c("Batched" = SEAGRASS, "Un-batched" = MUTED_OLIVE)) +
  scale_y_continuous(limits = c(0, 112), breaks = seq(0, 100, 20),
                     labels = function(x) paste0(x, "%"), expand = c(0, 0)) +
  labs(title    = "Batching Catastrophe",
       subtitle = "Same model, same prompts, same beam — only call structure differs",
       x = NULL, y = NULL, fill = NULL) +
  poster_theme

ggsave("plots/plot1a_batched_vs_unbatched.pdf", p1a, width = 9, height = 6)
ggsave("plots/plot1a_batched_vs_unbatched.png", p1a, width = 9, height = 6, dpi = 300)
cat("  -> plots/plot1a_batched_vs_unbatched.pdf\n")


# ############################################################
# PLOT 1b — Cost & Time Table (Gemini 2.5, b=5)
# ############################################################
cat("Building Plot 1b...\n")

# Build a styled table using tableGrob
tbl_data <- data.frame(
  ` `    = c("Batched", "Un-batched"),
  `Cost / 100 puzzles` = c("$0.32", "$0.99"),
  `Time / puzzle`      = c("~8 s", "~70 s"),
  `API calls / puzzle` = c("~20", "~170"),
  check.names = FALSE
)

# Theme for the table
tt <- ttheme_minimal(
  core = list(
    fg_params = list(fontsize = 14, fontface = "plain"),
    bg_params = list(fill = c("#f0f4f2", "white"), col = NA),
    padding   = unit(c(12, 8), "pt")
  ),
  colhead = list(
    fg_params = list(fontsize = 14, fontface = "bold", col = "white"),
    bg_params = list(fill = CHARCOAL_BLUE, col = NA),
    padding   = unit(c(12, 8), "pt")
  ),
  rowhead = list(
    fg_params = list(fontsize = 13, fontface = "bold")
  )
)

p1b_grob <- tableGrob(tbl_data, rows = NULL, theme = tt)

# Add a title
title_grob <- textGrob(
  "Cost of Un-batching (Gemini 2.5, ToT b = 5)",
  gp = gpar(fontsize = 16, fontface = "bold")
)
subtitle_grob <- textGrob(
  "3× cost, 9× time — but +72 pp accuracy",
  gp = gpar(fontsize = 12, col = "gray40")
)

p1b_final <- arrangeGrob(
  title_grob, subtitle_grob, p1b_grob,
  ncol = 1, heights = unit(c(1.2, 0.8, 3), "cm"),
  padding = unit(0.5, "cm")
)

ggsave("plots/plot1b_cost_table.pdf", p1b_final, width = 7, height = 3)
ggsave("plots/plot1b_cost_table.png", p1b_final, width = 7, height = 3, dpi = 300)
cat("  -> plots/plot1b_cost_table.pdf\n")


# ############################################################
# PLOT 2 — Verdict Distribution at Depth 0
# ############################################################
cat("Building Plot 2...\n")

# Data from analyze_verdicts.py output (depth 0 only, n=500 candidates each)
# Simplified into 4 categories:
#   3×sure | Mixed (contains sure) | Mixed (no sure) | 3×impossible

# --- Gemini 2.5 batched (depth 0) ---
# Raw: 3×sure 0.2%, 2s+1l 0.2%, 1s+2l 0.6%, 1s+1l+1i 1.2%,
#      2s+1i 1.8%, 1s+2i 22.0%,  2l+1i 1.0%, 1l+2i 7.2%, 3×imp 65.8%
gem25_bat_sure      <- 0.2
gem25_bat_mix_sure  <- 0.2 + 0.6 + 1.2 + 1.8 + 22.0  # 25.8
gem25_bat_mix_none  <- 1.0 + 7.2                        # 8.2
gem25_bat_imp       <- 65.8

# --- Gemini 2.5 un-batched (depth 0) ---
# Raw: 3×sure 23.2%, 2s+1l 21.0%, 1s+2l 16.6%, 1s+1l+1i 3.6%,
#      2s+1i 4.0%, 1s+2i 0.4%,  3×likely 30.8%, 2l+1i 0.4%
gem25_ub_sure      <- 23.2
gem25_ub_mix_sure  <- 21.0 + 16.6 + 3.6 + 4.0 + 0.4  # 45.6
gem25_ub_mix_none  <- 30.8 + 0.4                        # 31.2
gem25_ub_imp       <- 0.0

# --- Gemini 3.1 batched (depth 0) ---
# Raw: 3×sure 62.0%, 2s+1l 17.6%, 2s+1i 6.8%, 1s+2l 6.6%,
#      1s+1l+1i 3.4%, 1s+2i 0.8%,  3×likely 1.2%, 2l+1i 1.6%
gem31_bat_sure      <- 62.0
gem31_bat_mix_sure  <- 17.6 + 6.8 + 6.6 + 3.4 + 0.8  # 35.2
gem31_bat_mix_none  <- 1.2 + 1.6                        # 2.8
gem31_bat_imp       <- 0.0

# --- Gemini 3.1 un-batched (depth 0) ---
# Raw: 3×sure 51.0%, 2s+1l 8.2%, 2s+1i 6.6%, 1s+2l 7.2%,
#      1s+1l+1i 3.2%, 1s+2i 2.8%,  3×likely 17.8%, 2l+1i 3.0%, 3×imp 0.2%
gem31_ub_sure      <- 51.0
gem31_ub_mix_sure  <- 8.2 + 6.6 + 7.2 + 3.2 + 2.8    # 28.0
gem31_ub_mix_none  <- 17.8 + 3.0                        # 20.8
gem31_ub_imp       <- 0.2

p2_df <- data.frame(
  condition = rep(c("Gem 2.5\nBatched", "Gem 2.5\nUn-batched",
                    "Gem 3.1\nBatched", "Gem 3.1\nUn-batched"), each = 4),
  category  = rep(c("3×sure", "Mixed (w/ sure)", "Mixed (no sure)", "3×impossible"), times = 4),
  pct       = c(
    gem25_bat_sure, gem25_bat_mix_sure, gem25_bat_mix_none, gem25_bat_imp,
    gem25_ub_sure,  gem25_ub_mix_sure,  gem25_ub_mix_none,  gem25_ub_imp,
    gem31_bat_sure, gem31_bat_mix_sure, gem31_bat_mix_none, gem31_bat_imp,
    gem31_ub_sure,  gem31_ub_mix_sure,  gem31_ub_mix_none,  gem31_ub_imp
  )
)

p2_df$condition <- factor(p2_df$condition,
  levels = c("Gem 2.5\nBatched", "Gem 2.5\nUn-batched",
             "Gem 3.1\nBatched", "Gem 3.1\nUn-batched"))
p2_df$category <- factor(p2_df$category,
  levels = c("3×sure", "Mixed (w/ sure)", "Mixed (no sure)", "3×impossible"))

# Labels: only show if ≥ 5%
p2_df$label <- ifelse(p2_df$pct >= 5, paste0(round(p2_df$pct), "%"), "")

p2 <- ggplot(p2_df, aes(x = condition, y = pct, fill = category)) +
  geom_col(width = 0.65) +
  geom_text(aes(label = label),
            position = position_stack(vjust = 0.5),
            size = 4, color = "white", fontface = "bold") +
  # Vertical separator between models
  geom_vline(xintercept = 2.5, linetype = "dashed", color = "gray60", linewidth = 0.4) +
  scale_fill_manual(values = c(
    "3×sure"          = MUTED_OLIVE,
    "Mixed (w/ sure)"      = SEAGRASS,
    "Mixed (no sure)"      = CHARCOAL_BLUE,
    "3×impossible"    = PUMPKIN_SPICE
  )) +
  scale_y_continuous(labels = function(x) paste0(x, "%"), expand = c(0, 0)) +
  labs(title    = "Why Batching Hurts: Verdict Distribution at Depth 0",
       subtitle = "ToT (b = 5), 500 candidates per condition. Batching → mass 3×impossible on Gem 2.5",
       x = NULL, y = NULL, fill = NULL) +
  poster_theme +
  theme(legend.key.size = unit(0.6, "cm"))

ggsave("plots/plot2_verdict_distribution.pdf", p2, width = 10, height = 6)
ggsave("plots/plot2_verdict_distribution.png", p2, width = 10, height = 6, dpi = 300)
cat("  -> plots/plot2_verdict_distribution.pdf\n")


# ############################################################
# PLOT 3 — Batched vs +Tiebreaker vs Un-batched
# ############################################################
cat("Building Plot 3...\n")

p3_df <- data.frame(
  condition = rep(c("Batched", "+ Tiebreaker", "Un-batched"), each = 2),
  model     = rep(c("Gemini 2.5", "Gemini 3.1"), times = 3),
  rate      = c(15, 88,    # Batched
                28, 96,    # + Tiebreaker
                87, 96)    # Un-batched
)
p3_df$condition <- factor(p3_df$condition,
  levels = c("Batched", "+ Tiebreaker", "Un-batched"))
p3_df$model <- factor(p3_df$model, levels = c("Gemini 2.5", "Gemini 3.1"))

# Delta annotations (relative to batched baseline)
p3_deltas <- data.frame(
  condition = factor(c("+ Tiebreaker", "+ Tiebreaker", "Un-batched", "Un-batched"),
                     levels = levels(p3_df$condition)),
  model     = factor(c("Gemini 2.5", "Gemini 3.1", "Gemini 2.5", "Gemini 3.1"),
                     levels = levels(p3_df$model)),
  rate      = c(28, 96, 87, 96),
  delta     = c(13, 8, 72, 8)
)

p3 <- ggplot(p3_df, aes(x = condition, y = rate, fill = model)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.65) +
  geom_text(aes(label = paste0(rate, "%")),
            position = position_dodge(width = 0.75),
            vjust = -0.5, size = 4.5, fontface = "bold") +
  scale_fill_manual(values = c(
    "Gemini 2.5" = PUMPKIN_SPICE,
    "Gemini 3.1" = CHARCOAL_BLUE
  )) +
  # Horizontal reference lines at batched baselines
  geom_hline(yintercept = 15, linetype = "dotted", color = PUMPKIN_SPICE, linewidth = 0.5, alpha = 0.6) +
  geom_hline(yintercept = 88, linetype = "dotted", color = CHARCOAL_BLUE, linewidth = 0.5, alpha = 0.6) +
  scale_y_continuous(limits = c(0, 112), breaks = seq(0, 100, 20),
                     labels = function(x) paste0(x, "%"), expand = c(0, 0)) +
  labs(title    = "Recovering from Batching: Two Interventions",
       subtitle = "ToT (b = 5), n = 100 puzzles. Tiebreaker ≈ un-batching on 3.1; un-batching >> tiebreaker on 2.5",
       x = NULL, y = NULL, fill = NULL) +
  poster_theme

ggsave("plots/plot3_recovery.pdf", p3, width = 9, height = 6)
ggsave("plots/plot3_recovery.png", p3, width = 9, height = 6, dpi = 300)
cat("  -> plots/plot3_recovery.pdf\n")


# ############################################################
# PLOT COMBINED — IO, CoT, ToT Batched, +Tiebreaker, Un-batched
# ############################################################
cat("Building Combined Plot...\n")

pc_df <- data.frame(
  method = rep(c("IO", "CoT", "ToT\nBatched", "ToT\n+Tiebreaker", "ToT\nUn-batched"), each = 3),
  model  = rep(c("Paper (GPT-4)", "Gemini 2.5", "Gemini 3.1"), times = 5),
  rate   = c(7.3, 17, 16,       # IO
             4.0, 25, 84,       # CoT
             NA,  15, 88,       # ToT Batched (no GPT-4 batched data)
             NA,  28, 96,       # ToT +Tiebreaker
             74,  87, 96)       # ToT Un-batched
)
pc_df$method <- factor(pc_df$method,
  levels = c("IO", "CoT", "ToT\nBatched", "ToT\n+Tiebreaker", "ToT\nUn-batched"))
pc_df$model  <- factor(pc_df$model,
  levels = c("Paper (GPT-4)", "Gemini 2.5", "Gemini 3.1"))

# Drop NA rows (GPT-4 has no batched/tiebreaker data)
pc_df <- pc_df[!is.na(pc_df$rate), ]

pc <- ggplot(pc_df, aes(x = method, y = rate, fill = model)) +
  geom_col(position = position_dodge(width = 0.78), width = 0.7) +
  geom_text(aes(label = paste0(rate, "%")),
            position = position_dodge(width = 0.78),
            vjust = -0.5, size = 4, fontface = "bold") +
  # Vertical separator between CoT and ToT variants
  geom_vline(xintercept = 2.5, linetype = "dashed", color = "gray60", linewidth = 0.4) +
  scale_fill_manual(values = c(
    "Paper (GPT-4)" = GOLDEN_POLLEN,
    "Gemini 2.5"    = PUMPKIN_SPICE,
    "Gemini 3.1"    = CHARCOAL_BLUE
  )) +
  scale_y_continuous(limits = c(0, 112), breaks = seq(0, 100, 20),
                     labels = function(x) paste0(x, "%"), expand = c(0, 0)) +
  labs(title    = "Game of 24: Success Rate by Method",
       subtitle = "ToT (b = 5), n = 100 puzzles",
       x = NULL, y = NULL, fill = NULL) +
  poster_theme

ggsave("plots/plot_combined_main.pdf", pc, width = 12, height = 6)
ggsave("plots/plot_combined_main.png", pc, width = 12, height = 6, dpi = 300)
cat("  -> plots/plot_combined_main.pdf\n")


# ============================================================
cat("\nAll poster plots saved to plots/\n")
list.files("plots", pattern = "^plot", full.names = TRUE) |> cat(sep = "\n")
cat("\n")
