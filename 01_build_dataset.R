#!/usr/bin/env Rscript

# Build a leakage-safe NCAA tournament matchup dataset from the historical
# Kaggle March Madness files. Every predictor summarizes regular-season games
# completed before the NCAA tournament begins.

suppressPackageStartupMessages(library(data.table))

raw_dir <- file.path("data", "raw")
processed_dir <- file.path("data", "processed")
results_dir <- "results"
dir.create(processed_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

required_files <- c(
  "MRegularSeasonDetailedResults.csv",
  "MNCAATourneyDetailedResults.csv",
  "MNCAATourneySeeds.csv",
  "MTeams.csv"
)
missing_files <- required_files[!file.exists(file.path(raw_dir, required_files))]
if (length(missing_files)) {
  stop("Missing raw files: ", paste(missing_files, collapse = ", "))
}

regular <- fread(file.path(raw_dir, required_files[1]))
tourney <- fread(file.path(raw_dir, required_files[2]))
seeds <- fread(file.path(raw_dir, required_files[3]))
teams <- fread(file.path(raw_dir, required_files[4]))

invert_location <- function(x) {
  fifelse(x == "H", "A", fifelse(x == "A", "H", "N"))
}

# Express each regular-season game from both teams' perspectives.
winner <- regular[, .(
  Season, DayNum, TeamID = WTeamID, OppTeamID = LTeamID, location = WLoc,
  win = 1L, points_for = WScore, points_against = LScore,
  FGM = WFGM, FGA = WFGA, FGM3 = WFGM3, FGA3 = WFGA3,
  FTM = WFTM, FTA = WFTA, OR = WOR, DR = WDR, Ast = WAst,
  TO = WTO, Stl = WStl, Blk = WBlk, PF = WPF,
  OppFGM = LFGM, OppFGA = LFGA, OppFGM3 = LFGM3, OppFGA3 = LFGA3,
  OppFTM = LFTM, OppFTA = LFTA, OppOR = LOR, OppDR = LDR,
  OppAst = LAst, OppTO = LTO, OppStl = LStl, OppBlk = LBlk, OppPF = LPF
)]

loser <- regular[, .(
  Season, DayNum, TeamID = LTeamID, OppTeamID = WTeamID,
  location = invert_location(WLoc), win = 0L,
  points_for = LScore, points_against = WScore,
  FGM = LFGM, FGA = LFGA, FGM3 = LFGM3, FGA3 = LFGA3,
  FTM = LFTM, FTA = LFTA, OR = LOR, DR = LDR, Ast = LAst,
  TO = LTO, Stl = LStl, Blk = LBlk, PF = LPF,
  OppFGM = WFGM, OppFGA = WFGA, OppFGM3 = WFGM3, OppFGA3 = WFGA3,
  OppFTM = WFTM, OppFTA = WFTA, OppOR = WOR, OppDR = WDR,
  OppAst = WAst, OppTO = WTO, OppStl = WStl, OppBlk = WBlk, OppPF = WPF
)]

team_games <- rbindlist(list(winner, loser), use.names = TRUE)
setorder(team_games, Season, TeamID, DayNum)
team_games[, `:=`(
  margin = points_for - points_against,
  possessions = FGA - OR + TO + 0.475 * FTA,
  opp_possessions = OppFGA - OppOR + OppTO + 0.475 * OppFTA
)]

safe_ratio <- function(numerator, denominator) {
  fifelse(denominator > 0, numerator / denominator, NA_real_)
}

team_summary <- team_games[, .(
  games = .N,
  win_pct = mean(win),
  avg_margin = mean(margin),
  points_per_game = mean(points_for),
  points_allowed = mean(points_against),
  offensive_efficiency = 100 * safe_ratio(sum(points_for), sum(possessions)),
  defensive_efficiency = 100 * safe_ratio(sum(points_against), sum(opp_possessions)),
  effective_fg = safe_ratio(sum(FGM) + 0.5 * sum(FGM3), sum(FGA)),
  opponent_effective_fg = safe_ratio(sum(OppFGM) + 0.5 * sum(OppFGM3), sum(OppFGA)),
  turnover_rate = safe_ratio(sum(TO), sum(possessions)),
  forced_turnover_rate = safe_ratio(sum(OppTO), sum(opp_possessions)),
  offensive_rebound_rate = safe_ratio(sum(OR), sum(OR) + sum(OppDR)),
  defensive_rebound_rate = safe_ratio(sum(DR), sum(DR) + sum(OppOR)),
  free_throw_rate = safe_ratio(sum(FTA), sum(FGA)),
  three_point_rate = safe_ratio(sum(FGA3), sum(FGA)),
  assist_rate = safe_ratio(sum(Ast), sum(FGM)),
  steal_rate = safe_ratio(sum(Stl), sum(possessions)),
  block_rate = safe_ratio(sum(Blk), sum(OppFGA) - sum(OppFGA3)),
  foul_rate = safe_ratio(sum(PF), sum(possessions)),
  pace = mean(0.5 * (possessions + opp_possessions))
), by = .(Season, TeamID)]
team_summary[, net_efficiency := offensive_efficiency - defensive_efficiency]

# Recent form uses only the final ten regular-season games, still before the
# tournament. The complete season is retained for every other statistic.
recent_summary <- team_games[, tail(.SD, 10L), by = .(Season, TeamID)][, .(
  recent_win_pct = mean(win),
  recent_margin = mean(margin),
  recent_net_efficiency = 100 * safe_ratio(sum(points_for), sum(possessions)) -
    100 * safe_ratio(sum(points_against), sum(opp_possessions))
), by = .(Season, TeamID)]
team_summary <- merge(team_summary, recent_summary, by = c("Season", "TeamID"))

# Strength of schedule is the average full-season win percentage of opponents.
opponent_strength <- team_summary[, .(
  Season, OppTeamID = TeamID, opponent_win_pct = win_pct
)]
games_with_strength <- merge(
  team_games[, .(Season, TeamID, OppTeamID)],
  opponent_strength,
  by = c("Season", "OppTeamID"),
  all.x = TRUE
)
schedule_strength <- games_with_strength[, .(
  schedule_strength = mean(opponent_win_pct, na.rm = TRUE)
), by = .(Season, TeamID)]
team_summary <- merge(team_summary, schedule_strength, by = c("Season", "TeamID"))

seeds[, seed_number := as.integer(substr(Seed, 2, 3))]
seed_lookup <- seeds[, .(Season, TeamID, seed_number)]

# Orient matchups by team ID rather than winner so the response is not encoded
# in row construction. Positive margin means Team 1 (the lower ID) won.
matchups <- tourney[, .(
  Season,
  DayNum,
  Team1ID = pmin(WTeamID, LTeamID),
  Team2ID = pmax(WTeamID, LTeamID),
  margin = fifelse(WTeamID < LTeamID, WScore - LScore, LScore - WScore),
  NumOT
)]

feature_columns <- setdiff(names(team_summary), c("Season", "TeamID"))

team1 <- copy(team_summary)
setnames(team1, c("TeamID", feature_columns), c("Team1ID", paste0(feature_columns, "_1")))
team2 <- copy(team_summary)
setnames(team2, c("TeamID", feature_columns), c("Team2ID", paste0(feature_columns, "_2")))

matchups <- merge(matchups, team1, by = c("Season", "Team1ID"), all.x = TRUE)
matchups <- merge(matchups, team2, by = c("Season", "Team2ID"), all.x = TRUE)

seed1 <- copy(seed_lookup)
setnames(seed1, c("TeamID", "seed_number"), c("Team1ID", "seed_1"))
seed2 <- copy(seed_lookup)
setnames(seed2, c("TeamID", "seed_number"), c("Team2ID", "seed_2"))
matchups <- merge(matchups, seed1, by = c("Season", "Team1ID"), all.x = TRUE)
matchups <- merge(matchups, seed2, by = c("Season", "Team2ID"), all.x = TRUE)

name_lookup <- teams[, .(TeamID, TeamName)]
name1 <- copy(name_lookup)
setnames(name1, c("TeamID", "TeamName"), c("Team1ID", "Team1"))
name2 <- copy(name_lookup)
setnames(name2, c("TeamID", "TeamName"), c("Team2ID", "Team2"))
matchups <- merge(matchups, name1, by = "Team1ID", all.x = TRUE)
matchups <- merge(matchups, name2, by = "Team2ID", all.x = TRUE)

for (feature in feature_columns) {
  matchups[, paste0(feature, "_diff") :=
    get(paste0(feature, "_1")) - get(paste0(feature, "_2"))]
}
matchups[, `:=`(
  seed_diff = seed_2 - seed_1,
  team1_win = as.integer(margin > 0),
  seed_gap = abs(seed_1 - seed_2),
  favorite_team1 = seed_1 < seed_2,
  favorite_margin = fifelse(seed_1 < seed_2, margin, -margin),
  split = fifelse(
    Season <= 2014, "Training (2003-2014)",
    fifelse(Season == 2015, "Validation (2015)", "Test (2016-2018)")
  )
)]
matchups[, upset := fifelse(
  seed_1 == seed_2,
  NA_integer_,
  as.integer(favorite_margin < 0)
)]

model_features <- c("seed_diff", paste0(feature_columns, "_diff"))
keep_columns <- c(
  "Season", "DayNum", "Team1ID", "Team1", "Team2ID", "Team2",
  "seed_1", "seed_2", "margin", "team1_win", "seed_gap",
  "favorite_team1", "favorite_margin", "upset", "NumOT", "split",
  model_features
)
model_data <- matchups[, ..keep_columns]

complete_rows <- complete.cases(model_data[, c("margin", model_features), with = FALSE])
dropped <- sum(!complete_rows)
model_data <- model_data[complete_rows]
setorder(model_data, Season, DayNum, Team1ID)

fwrite(
  model_data,
  file.path(processed_dir, "ncaa_tournament_matchups_2003_2018.csv")
)
fwrite(
  data.table(feature = c("seed_diff", paste0(feature_columns, "_diff"))),
  file.path(results_dir, "candidate_features.csv")
)

quality_lines <- c(
  "NCAA tournament matchup data quality summary",
  sprintf("Regular-season detailed games: %s", format(nrow(regular), big.mark = ",")),
  sprintf("Tournament games before joins: %s", format(nrow(tourney), big.mark = ",")),
  sprintf("Complete tournament matchups retained: %s", format(nrow(model_data), big.mark = ",")),
  sprintf("Rows dropped for missing predictors: %s", format(dropped, big.mark = ",")),
  sprintf("Seasons represented: %d-%d", min(model_data$Season), max(model_data$Season)),
  sprintf("Candidate continuous predictors: %d", length(model_features)),
  sprintf("Training games: %d", nrow(model_data[Season <= 2014])),
  sprintf("Validation games: %d", nrow(model_data[Season == 2015])),
  sprintf("Untouched test games: %d", nrow(model_data[Season %in% 2016:2018])),
  sprintf("Overall mean absolute tournament margin: %.2f points", mean(abs(model_data$margin))),
  sprintf("Unequal-seed upset rate: %.1f%%", 100 * mean(model_data$upset, na.rm = TRUE))
)
writeLines(quality_lines, file.path(results_dir, "data_quality.txt"))
cat(paste(quality_lines, collapse = "\n"), "\n")
