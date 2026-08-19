#!/usr/bin/env Rscript

# Compare a seed-only benchmark, Ridge, Elastic Net, LASSO, and a regression
# tree for NCAA tournament scoring margin. Model selection uses 2015; the
# 2016-2018 tournaments remain untouched until the final evaluation.

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(glmnet)
  library(rpart)
})

data_path <- file.path("data", "processed", "ncaa_tournament_matchups_2003_2018.csv")
figures_dir <- "figures"
results_dir <- "results"
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(data_path)) {
  stop("Processed data not found. Run Rscript 01_build_dataset.R first.")
}

games <- fread(data_path)
feature_names <- fread(file.path(results_dir, "candidate_features.csv"))$feature

idx_train <- games$Season <= 2014
idx_validation <- games$Season == 2015
idx_final_train <- games$Season <= 2015
idx_test <- games$Season %in% 2016:2018

x_all <- as.matrix(games[, ..feature_names])
y_all <- games$margin

train <- games[idx_train]
validation <- games[idx_validation]
final_train <- games[idx_final_train]
test <- games[idx_test]

# Seed-only benchmark.
seed_tuning_fit <- lm(margin ~ seed_diff, data = train)
seed_validation_prediction <- as.numeric(predict(seed_tuning_fit, newdata = validation))
seed_final_fit <- lm(margin ~ seed_diff, data = final_train)
seed_test_prediction <- as.numeric(predict(seed_final_fit, newdata = test))

# Tune alpha and lambda without consulting the final season.
penalty_candidates <- data.table(
  candidate = c(
    "Ridge", "Elastic Net (alpha=0.25)", "Elastic Net (alpha=0.50)",
    "Elastic Net (alpha=0.75)", "LASSO"
  ),
  family = c("Ridge", "Elastic Net", "Elastic Net", "Elastic Net", "LASSO"),
  prediction_key = c("ridge", "elastic_net", "elastic_net", "elastic_net", "lasso"),
  alpha = c(0, 0.25, 0.50, 0.75, 1)
)

set.seed(444)
candidate_tuning <- rbindlist(lapply(seq_len(nrow(penalty_candidates)), function(i) {
  specification <- penalty_candidates[i]
  fit_path <- glmnet(
    x_all[idx_train, , drop = FALSE],
    y_all[idx_train],
    alpha = specification$alpha,
    nlambda = 120,
    standardize = TRUE
  )
  validation_predictions <- predict(
    fit_path,
    newx = x_all[idx_validation, , drop = FALSE],
    s = fit_path$lambda
  )
  validation_mae <- colMeans(abs(validation_predictions - validation$margin))
  best <- which.min(validation_mae)
  specification[, `:=`(
    lambda = fit_path$lambda[best],
    validation_MAE = validation_mae[best],
    validation_RMSE = sqrt(mean(
      (validation_predictions[, best] - validation$margin)^2
    ))
  )]
  specification
}))

penalty_tuning <- rbindlist(list(
  candidate_tuning[family == "Ridge"],
  candidate_tuning[family == "Elastic Net"][which.min(validation_MAE)],
  candidate_tuning[family == "LASSO"]
))
penalty_tuning[, family_order := match(family, c("Ridge", "Elastic Net", "LASSO"))]
setorder(penalty_tuning, family_order)
penalty_tuning[, family_order := NULL]

penalized_fits <- list()
penalized_test_predictions <- list()
penalized_validation_predictions <- list()

for (i in seq_len(nrow(penalty_tuning))) {
  specification <- penalty_tuning[i]
  key <- specification$prediction_key

  final_fit <- glmnet(
    x_all[idx_final_train, , drop = FALSE],
    y_all[idx_final_train],
    alpha = specification$alpha,
    lambda = specification$lambda,
    standardize = TRUE
  )
  penalized_fits[[key]] <- final_fit
  penalized_test_predictions[[key]] <- as.numeric(predict(
    final_fit,
    newx = x_all[idx_test, , drop = FALSE],
    s = specification$lambda
  ))

  tuning_fit <- glmnet(
    x_all[idx_train, , drop = FALSE],
    y_all[idx_train],
    alpha = specification$alpha,
    lambda = specification$lambda,
    standardize = TRUE
  )
  penalized_validation_predictions[[key]] <- as.numeric(predict(
    tuning_fit,
    newx = x_all[idx_validation, , drop = FALSE],
    s = specification$lambda
  ))

  coefficient_matrix <- as.matrix(coef(final_fit, s = specification$lambda))
  coefficients <- data.table(
    feature = rownames(coefficient_matrix)[-1],
    coefficient = as.numeric(coefficient_matrix[-1, 1])
  )
  training_sd <- apply(x_all[idx_final_train, , drop = FALSE], 2, sd)
  coefficients[, standardized_effect := coefficient * training_sd[feature]]
  coefficients[, absolute_standardized_effect := abs(standardized_effect)]
  setorder(coefficients, -absolute_standardized_effect)
  fwrite(coefficients, file.path(results_dir, paste0(key, "_coefficients.csv")))
}

penalty_tuning[, adjustment_parameters := vapply(
  prediction_key,
  function(key) {
    values <- as.numeric(coef(
      penalized_fits[[key]],
      s = penalty_tuning[prediction_key == key]$lambda[1]
    ))[-1]
    sum(values != 0)
  },
  integer(1)
)]

# A regression tree uses short display names so its branches remain readable.
tree_feature_names <- c(
  seed_diff = "Seed_advantage",
  games_diff = "Games_gap",
  win_pct_diff = "Win_rate_gap",
  avg_margin_diff = "Season_margin_gap",
  points_per_game_diff = "Scoring_gap",
  points_allowed_diff = "Points_allowed_gap",
  offensive_efficiency_diff = "Off_efficiency_gap",
  defensive_efficiency_diff = "Def_efficiency_gap",
  effective_fg_diff = "eFG_gap",
  opponent_effective_fg_diff = "Opponent_eFG_gap",
  turnover_rate_diff = "Turnover_rate_gap",
  forced_turnover_rate_diff = "Forced_turnover_gap",
  offensive_rebound_rate_diff = "Off_rebound_gap",
  defensive_rebound_rate_diff = "Def_rebound_gap",
  free_throw_rate_diff = "Free_throw_rate_gap",
  three_point_rate_diff = "Three_point_rate_gap",
  assist_rate_diff = "Assist_rate_gap",
  steal_rate_diff = "Steal_rate_gap",
  block_rate_diff = "Block_rate_gap",
  foul_rate_diff = "Foul_rate_gap",
  pace_diff = "Pace_gap",
  net_efficiency_diff = "Net_efficiency_gap",
  recent_win_pct_diff = "Recent_win_gap",
  recent_margin_diff = "Recent_margin_gap",
  recent_net_efficiency_diff = "Recent_efficiency_gap",
  schedule_strength_diff = "Schedule_strength_gap"
)
stopifnot(setequal(names(tree_feature_names), feature_names))

tree_data <- games[, c("margin", feature_names), with = FALSE]
setnames(tree_data, names(tree_feature_names), unname(tree_feature_names))
tree_formula <- as.formula(paste(
  "margin ~",
  paste(unname(tree_feature_names), collapse = " + ")
))

set.seed(444)
tree_tuning_full <- rpart(
  tree_formula,
  data = tree_data[idx_train],
  method = "anova",
  control = rpart.control(cp = 0, minsplit = 35, minbucket = 14, maxdepth = 6, xval = 0)
)
candidate_cp <- unique(tree_tuning_full$cptable[, "CP"])
tree_validation_mae <- vapply(candidate_cp, function(cp_value) {
  candidate_tree <- prune(tree_tuning_full, cp = cp_value)
  prediction <- predict(candidate_tree, newdata = tree_data[idx_validation])
  mean(abs(validation$margin - prediction))
}, numeric(1))
best_cp <- candidate_cp[which.min(tree_validation_mae)]

tree_final_full <- rpart(
  tree_formula,
  data = tree_data[idx_final_train],
  method = "anova",
  control = rpart.control(cp = 0, minsplit = 35, minbucket = 14, maxdepth = 6, xval = 0)
)
tree_final <- prune(tree_final_full, cp = best_cp)
tree_test_prediction <- as.numeric(predict(tree_final, newdata = tree_data[idx_test]))
tree_validation_prediction <- as.numeric(predict(
  prune(tree_tuning_full, cp = best_cp),
  newdata = tree_data[idx_validation]
))

tree_importance <- data.table(
  feature = names(tree_final$variable.importance),
  importance = as.numeric(tree_final$variable.importance)
)
tree_importance[, relative_importance := importance / sum(importance)]
setorder(tree_importance, -importance)
fwrite(tree_importance, file.path(results_dir, "tree_variable_importance.csv"))

# Validation comparison. Selection is locked before the 2018 test is scored.
validation_results <- rbindlist(list(
  data.table(
    model = "Seed-only linear",
    validation_MAE = mean(abs(validation$margin - seed_validation_prediction)),
    validation_RMSE = sqrt(mean((validation$margin - seed_validation_prediction)^2))
  ),
  rbindlist(lapply(seq_len(nrow(penalty_tuning)), function(i) {
    key <- penalty_tuning$prediction_key[i]
    prediction <- penalized_validation_predictions[[key]]
    data.table(
      model = penalty_tuning$family[i],
      validation_MAE = mean(abs(validation$margin - prediction)),
      validation_RMSE = sqrt(mean((validation$margin - prediction)^2))
    )
  })),
  data.table(
    model = "Regression tree",
    validation_MAE = mean(abs(validation$margin - tree_validation_prediction)),
    validation_RMSE = sqrt(mean((validation$margin - tree_validation_prediction)^2))
  )
))
validation_results[, selected := validation_MAE == min(validation_MAE)]
selected_model <- validation_results[selected == TRUE]$model[1]

prediction_lookup <- list(
  "Seed-only linear" = seed_test_prediction,
  "Ridge" = penalized_test_predictions$ridge,
  "Elastic Net" = penalized_test_predictions$elastic_net,
  "LASSO" = penalized_test_predictions$lasso,
  "Regression tree" = tree_test_prediction
)
selected_test_prediction <- prediction_lookup[[selected_model]]

test_predictions <- test[, .(
  Season, DayNum, Team1ID, Team1, Team2ID, Team2, seed_1, seed_2,
  margin, favorite_team1, favorite_margin, upset
)]
test_predictions[, `:=`(
  seed_only = seed_test_prediction,
  ridge = penalized_test_predictions$ridge,
  elastic_net = penalized_test_predictions$elastic_net,
  lasso = penalized_test_predictions$lasso,
  regression_tree = tree_test_prediction,
  selected_prediction = selected_test_prediction,
  selected_model = selected_model
)]
test_predictions[, predicted_favorite_margin := fifelse(
  favorite_team1,
  selected_prediction,
  -selected_prediction
)]

score_model <- function(actual, predicted, adjustment_parameters) {
  error <- actual - predicted
  r_squared <- 1 - sum(error^2) / sum((actual - mean(actual))^2)
  n <- length(actual)
  adjusted_r_squared <- 1 - (1 - r_squared) *
    (n - 1) / (n - adjustment_parameters - 1)
  data.table(
    MAE_points = mean(abs(error)),
    Median_AE_points = median(abs(error)),
    RMSE_points = sqrt(mean(error^2)),
    R_squared = r_squared,
    Adjusted_R_squared = adjusted_r_squared,
    Winner_accuracy_percent = 100 * mean((predicted > 0) == (actual > 0))
  )
}

model_complexity <- data.table(
  model = c("Seed-only linear", penalty_tuning$family, "Regression tree"),
  adjustment_parameters = c(
    1L,
    penalty_tuning$adjustment_parameters,
    sum(tree_final$frame$var == "<leaf>") - 1L
  )
)

model_metrics <- rbindlist(lapply(model_complexity$model, function(model_name) {
  cbind(
    model = model_name,
    score_model(
      test$margin,
      prediction_lookup[[model_name]],
      model_complexity[model == model_name]$adjustment_parameters
    )
  )
}))
model_metrics[, selected := model == selected_model]

# Season-by-season diagnostics show whether the selected model's pooled test
# result is driven by one unusually easy tournament.
annual_test_metrics <- rbindlist(lapply(sort(unique(test_predictions$Season)), function(season_value) {
  season_games <- test_predictions[Season == season_value]
  rbindlist(lapply(c("Seed-only linear", selected_model), function(model_name) {
    prediction <- if (model_name == "Seed-only linear") {
      season_games$seed_only
    } else {
      season_games$selected_prediction
    }
    error <- season_games$margin - prediction
    data.table(
      Season = season_value,
      model = model_name,
      games = nrow(season_games),
      MAE_points = mean(abs(error)),
      RMSE_points = sqrt(mean(error^2)),
      R_squared = 1 - sum(error^2) /
        sum((season_games$margin - mean(season_games$margin))^2),
      Winner_accuracy_percent = 100 * mean(
        (prediction > 0) == (season_games$margin > 0)
      )
    )
  }))
}))

# Rank unequal-seed games by the selected model's predicted favorite margin.
# The most vulnerable quartile is the group with the smallest predicted margin.
upset_games <- test_predictions[!is.na(upset)]
upset_games[, vulnerability_rank := frank(
  predicted_favorite_margin,
  ties.method = "first"
)]
flag_count <- ceiling(0.25 * nrow(upset_games))
upset_games[, flagged_vulnerable := vulnerability_rank <= flag_count]
base_upset_rate <- mean(upset_games$upset)
upset_triage <- data.table(
  model = selected_model,
  eligible_games = nrow(upset_games),
  flagged_games = sum(upset_games$flagged_vulnerable),
  flagged_share_percent = 100 * mean(upset_games$flagged_vulnerable),
  upset_base_rate_percent = 100 * base_upset_rate,
  upset_precision_percent = 100 * mean(upset_games[flagged_vulnerable == TRUE]$upset),
  upset_recall_percent = 100 * sum(upset_games[flagged_vulnerable == TRUE]$upset) / sum(upset_games$upset),
  lift = mean(upset_games[flagged_vulnerable == TRUE]$upset) / base_upset_rate
)

# Five equal-sized vulnerability groups for a stable test-season diagnostic.
upset_games[, vulnerability_group := cut(
  frank(predicted_favorite_margin, ties.method = "first") / .N,
  breaks = seq(0, 1, by = 0.2),
  include.lowest = TRUE,
  labels = c("Most vulnerable", "2", "3", "4", "Safest")
)]
upset_calibration <- upset_games[, .(
  games = .N,
  mean_predicted_favorite_margin = mean(predicted_favorite_margin),
  actual_upset_percent = 100 * mean(upset)
), by = vulnerability_group]
upset_calibration[, group_order := as.integer(vulnerability_group)]
setorder(upset_calibration, group_order)
upset_calibration[, group_order := NULL]

fwrite(candidate_tuning, file.path(results_dir, "penalty_candidate_tuning.csv"))
fwrite(penalty_tuning, file.path(results_dir, "penalty_family_tuning.csv"))
fwrite(validation_results, file.path(results_dir, "validation_model_selection.csv"))
fwrite(model_complexity, file.path(results_dir, "model_complexity.csv"))
fwrite(model_metrics, file.path(results_dir, "model_metrics_2016_2018.csv"))
fwrite(annual_test_metrics, file.path(results_dir, "annual_test_metrics.csv"))
fwrite(test_predictions, file.path(results_dir, "test_predictions_2016_2018.csv"))
fwrite(upset_triage, file.path(results_dir, "upset_triage_2016_2018.csv"))
fwrite(upset_calibration, file.path(results_dir, "upset_calibration_2016_2018.csv"))

# Figure 1: seed advantage is useful but leaves substantial unexplained spread.
p1 <- ggplot(games, aes(seed_diff, margin)) +
  geom_point(alpha = 0.18, size = 1.4, colour = "#377eb8", position = position_jitter(width = 0.12)) +
  geom_smooth(method = "lm", formula = y ~ x, se = TRUE, colour = "#d7301f", linewidth = 1) +
  geom_hline(yintercept = 0, colour = "#666666", linewidth = 0.35) +
  labs(
    title = "Tournament seeds predict direction, not certainty",
    subtitle = "Men's NCAA tournament games, 2003-2018",
    x = "Seed advantage for Team 1 (opponent seed minus Team 1 seed)",
    y = "Team 1 scoring margin (points)"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title.position = "plot")
ggsave(file.path(figures_dir, "01_seed_advantage_and_margin.png"), p1,
  width = 9, height = 5.6, dpi = 200, bg = "white"
)

# Figure 2: honest test-season error.
performance_plot <- melt(
  model_metrics,
  id.vars = "model",
  measure.vars = c("MAE_points", "RMSE_points"),
  variable.name = "metric",
  value.name = "error"
)
performance_plot[, metric := factor(
  metric,
  levels = c("MAE_points", "RMSE_points"),
  labels = c("MAE", "RMSE")
)]
performance_plot[, model := factor(model, levels = model_complexity$model)]
p2 <- ggplot(performance_plot, aes(model, error, fill = metric)) +
  geom_col(position = "dodge", width = 0.74) +
  geom_text(
    aes(label = sprintf("%.1f", error)),
    position = position_dodge(width = 0.74),
    vjust = -0.25,
    size = 3.2
  ) +
  scale_fill_manual(values = c(MAE = "#3182bd", RMSE = "#e6550d")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(
    title = "Scoring-margin error on the untouched 2016-2018 tournaments",
    x = NULL,
    y = "Prediction error (points)",
    fill = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "top",
    plot.title.position = "plot",
    axis.text.x = element_text(angle = 18, hjust = 1)
  )
ggsave(file.path(figures_dir, "02_model_performance.png"), p2,
  width = 9.5, height = 5.4, dpi = 200, bg = "white"
)

# Figure 3: selected model calibration and residual spread.
plot_test <- copy(test_predictions)
plot_test[, result := fifelse(margin > 0, paste0(Team1, " win"), paste0(Team2, " win"))]
calibration_limit <- 5 * ceiling(max(abs(c(
  plot_test$selected_prediction,
  plot_test$margin
))) / 5)
p3 <- ggplot(plot_test, aes(selected_prediction, margin)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "#777777") +
  geom_hline(yintercept = 0, linewidth = 0.3, colour = "#999999") +
  geom_vline(xintercept = 0, linewidth = 0.3, colour = "#999999") +
  geom_point(size = 2.4, alpha = 0.75, colour = "#2166ac") +
  coord_equal(
    xlim = c(-calibration_limit, calibration_limit),
    ylim = c(-calibration_limit, calibration_limit)
  ) +
  labs(
    title = sprintf("%s captures team strength but not every upset", selected_model),
    subtitle = "Each point is one NCAA tournament game from 2016-2018",
    x = "Predicted Team 1 margin",
    y = "Observed Team 1 margin"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title.position = "plot")
ggsave(file.path(figures_dir, "03_selected_model_calibration.png"), p3,
  width = 7.2, height = 6.1, dpi = 200, bg = "white"
)

# Figure 4: practical upset-risk ranking.
p4 <- ggplot(upset_calibration, aes(vulnerability_group, actual_upset_percent)) +
  geom_hline(
    yintercept = 100 * base_upset_rate,
    linetype = "dashed",
    colour = "#555555"
  ) +
  geom_col(fill = "#756bb1", width = 0.72) +
  geom_text(aes(label = sprintf("%.0f%%", actual_upset_percent)), vjust = -0.3, size = 3.6) +
  scale_y_continuous(
    limits = c(0, max(60, 1.15 * max(upset_calibration$actual_upset_percent))),
    expand = expansion(mult = c(0, 0.03))
  ) +
  labs(
    title = "Predicted vulnerability concentrates actual 2016-2018 upsets",
    subtitle = "Unequal-seed games grouped by the selected model's predicted favorite margin",
    x = "Favorite vulnerability group",
    y = "Games won by the underdog (%)",
    caption = sprintf("Dashed line: %.1f%% overall upset rate among unequal-seed games.", 100 * base_upset_rate)
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title.position = "plot")
ggsave(file.path(figures_dir, "04_upset_risk_groups.png"), p4,
  width = 8.5, height = 5.2, dpi = 200, bg = "white"
)

# Figure 5: readable pruning of the fitted regression tree. Metrics continue to
# use the validation-selected tree; this smaller tree is explanatory only.
explanatory_candidates <- unique(tree_final_full$cptable[, "CP"])
explanatory_sizes <- rbindlist(lapply(explanatory_candidates, function(cp_value) {
  candidate <- prune(tree_final_full, cp = cp_value)
  data.table(cp = cp_value, leaves = sum(candidate$frame$var == "<leaf>"))
}))
eligible_explanatory <- explanatory_sizes[leaves <= 5]
if (!nrow(eligible_explanatory)) eligible_explanatory <- explanatory_sizes[which.min(leaves)]
explanatory_cp <- eligible_explanatory[which.min(cp)]$cp
tree_explanatory <- prune(tree_final_full, cp = explanatory_cp)

png(
  file.path(figures_dir, "05_regression_tree_explanatory.png"),
  width = 2400,
  height = 1450,
  res = 220,
  bg = "white"
)
par(mar = c(1.2, 1.0, 5.0, 1.0), xpd = NA)
plot(tree_explanatory, uniform = TRUE, branch = 0.45, compress = TRUE, margin = 0.12)
text(
  tree_explanatory,
  use.n = TRUE,
  all = TRUE,
  fancy = TRUE,
  digits = 2,
  cex = 1.05,
  pretty = 0
)
title(
  main = "A readable regression tree for tournament scoring margin",
  sub = "Terminal nodes report predicted Team 1 margin and number of historical matchups",
  cex.main = 1.35,
  cex.sub = 0.95,
  line = 3.1
)
dev.off()

tree_explanatory_nodes <- data.table(
  node = as.integer(row.names(tree_explanatory$frame)),
  split_variable = tree_explanatory$frame$var,
  observations = tree_explanatory$frame$n,
  predicted_margin = tree_explanatory$frame$yval,
  is_leaf = tree_explanatory$frame$var == "<leaf>"
)
fwrite(tree_explanatory_nodes, file.path(results_dir, "tree_explanatory_nodes.csv"))

# Figure 6: standardized effects from the best penalized model.
best_penalized <- penalty_tuning[which.min(validation_MAE)]
best_penalized_coefficients <- fread(file.path(
  results_dir,
  paste0(best_penalized$prediction_key, "_coefficients.csv")
))[absolute_standardized_effect > 0]
best_penalized_coefficients <- head(best_penalized_coefficients, 12)
best_penalized_coefficients[, feature_label := gsub("_diff$", "", feature)]
best_penalized_coefficients[, feature_label := gsub("_", " ", feature_label)]
best_penalized_coefficients[, feature_label := reorder(feature_label, standardized_effect)]

p6 <- ggplot(
  best_penalized_coefficients,
  aes(standardized_effect, feature_label, fill = standardized_effect > 0)
) +
  geom_col(width = 0.72, show.legend = FALSE) +
  geom_vline(xintercept = 0, linewidth = 0.35, colour = "#555555") +
  scale_fill_manual(values = c(`FALSE` = "#d73027", `TRUE` = "#2171b5")) +
  labs(
    title = sprintf("Largest standardized effects in the best penalized model (%s)", best_penalized$family),
    subtitle = "Effect on Team 1 margin for a one-standard-deviation increase in the matchup feature",
    x = "Expected margin change (points)",
    y = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title.position = "plot")
ggsave(file.path(figures_dir, "06_penalized_standardized_effects.png"), p6,
  width = 8.6, height = 5.8, dpi = 200, bg = "white"
)

# Figure 7: the held-out 2015 comparison that actually selected the model.
validation_plot <- copy(validation_results)
validation_plot[, model := factor(model, levels = rev(model[order(validation_MAE)]))]
p7 <- ggplot(validation_plot, aes(validation_MAE, model, fill = selected)) +
  geom_col(width = 0.68, show.legend = FALSE) +
  geom_text(
    aes(label = sprintf("%.2f", validation_MAE)),
    hjust = -0.18,
    size = 3.7
  ) +
  scale_fill_manual(values = c(`FALSE` = "#9ecae1", `TRUE` = "#2171b5")) +
  scale_x_continuous(
    limits = c(0, 10.8),
    expand = expansion(mult = c(0, 0.02))
  ) +
  labs(
    title = "Ridge narrowly wins the held-out 2015 validation tournament",
    subtitle = "Lower mean absolute error is better; the three penalized models are nearly tied",
    x = "Validation MAE (points)",
    y = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title.position = "plot")
ggsave(file.path(figures_dir, "07_validation_model_comparison.png"), p7,
  width = 8.6, height = 4.8, dpi = 200, bg = "white"
)

# Figure 8: compare the selected model with the seed-only benchmark in each
# untouched tournament rather than relying only on the pooled test result.
annual_plot <- copy(annual_test_metrics)
annual_plot[, model := factor(model, levels = c("Seed-only linear", selected_model))]
annual_plot[, label_y := MAE_points + fifelse(model == "Seed-only linear", 0.35, -0.35)]
p8 <- ggplot(annual_plot, aes(factor(Season), MAE_points, group = model, colour = model)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  geom_text(
    aes(y = label_y, label = sprintf("%.1f", MAE_points)),
    size = 3.2, show.legend = FALSE
  ) +
  scale_colour_manual(values = c("Seed-only linear" = "#969696", "Ridge" = "#2171b5")) +
  scale_y_continuous(limits = c(0, 12), expand = expansion(mult = c(0, 0.04))) +
  labs(
    title = "Ridge's advantage is checked tournament by tournament",
    subtitle = "Mean absolute scoring-margin error on three untouched seasons",
    x = "Tournament season",
    y = "MAE (points)",
    colour = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top", plot.title.position = "plot")
ggsave(file.path(figures_dir, "08_test_mae_by_season.png"), p8,
  width = 8.2, height = 4.8, dpi = 200, bg = "white"
)

summary_lines <- c(
  sprintf("Selected model from 2015 validation: %s", selected_model),
  sprintf("Selected validation MAE: %.3f points", validation_results[selected == TRUE]$validation_MAE),
  sprintf("Selected 2016-2018 test MAE: %.3f points", model_metrics[selected == TRUE]$MAE_points),
  sprintf("Selected 2016-2018 test R-squared: %.3f", model_metrics[selected == TRUE]$R_squared),
  sprintf("Selected 2016-2018 adjusted R-squared: %.3f", model_metrics[selected == TRUE]$Adjusted_R_squared),
  sprintf("Selected 2016-2018 winner accuracy: %.1f%%", model_metrics[selected == TRUE]$Winner_accuracy_percent),
  sprintf("Vulnerable-quartile upset precision: %.1f%%", upset_triage$upset_precision_percent),
  sprintf("Vulnerable-quartile upset recall: %.1f%%", upset_triage$upset_recall_percent),
  sprintf("Vulnerable-quartile lift: %.2f", upset_triage$lift),
  sprintf("Validation-selected tree CP: %.6f", best_cp),
  sprintf("Selected tree leaves: %d", sum(tree_final$frame$var == "<leaf>")),
  sprintf("Explanatory tree leaves: %d", sum(tree_explanatory$frame$var == "<leaf>"))
)
writeLines(summary_lines, file.path(results_dir, "model_summary.txt"))
cat(paste(summary_lines, collapse = "\n"), "\n")
