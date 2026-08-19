# STAT 444 NCAA Tournament Project

Author: Adit Sinsinwal (Student ID 20988864)

This project predicts men's NCAA tournament scoring margin using information available before the tournament. It compares a seed-only benchmark, Ridge, Elastic Net, LASSO, and a regression tree, then converts the selected margin forecast into an upset-risk ranking.

## Main result

Ridge was selected using the 2015 validation tournament. On the untouched 2016-2018 tournaments it achieved:

- MAE: 8.91 points
- RMSE: 11.89 points
- R-squared: 0.337
- Adjusted R-squared: 0.238
- Winner accuracy: 74.1%

Ridge also improved on the seed-only MAE separately in 2016, 2017, and 2018. The report includes the held-out validation comparison, predicted-versus-observed test calibration, and the season-by-season stability check.

Among unequal-seed test games, the most vulnerable fifth of favourites had a 51.4% upset rate, compared with 7.9% for the safest fifth.

## Data

The raw CSV files are a public archive of the historical men's files used in Kaggle's March Machine Learning Mania competition:

- `MRegularSeasonDetailedResults.csv`
- `MNCAATourneyDetailedResults.csv`
- `MNCAATourneySeeds.csv`
- `MTeams.csv`

The public archive is at <https://github.com/kjaisingh/Forecasting-March-Madness>. The detailed tournament file ends in 2018, so this is a historical evaluation rather than a live 2026 forecast.

## Reproduce the analysis

From this directory, run:

```bash
Rscript 01_build_dataset.R
Rscript 02_fit_models.R
```

Then render the report with R Markdown:

```r
rmarkdown::render("final_report.Rmd", output_format = "pdf_document")
```

Required R packages are `data.table`, `ggplot2`, `glmnet`, `rpart`, `knitr`, and `rmarkdown`. A LaTeX distribution and Pandoc are required for PDF rendering.

## Files

- `01_build_dataset.R`: converts game results to leakage-safe team-season summaries and tournament matchup differences.
- `02_fit_models.R`: tunes, fits, evaluates, and plots all models.
- `final_report.Rmd`: reproducible report source.
- `final_report.pdf`: polished submission-ready report.
- `data/processed/ncaa_tournament_matchups_2003_2018.csv`: analysis-ready matchup data.
- `results/`: validation choices, pooled and annual test metrics, coefficients, tree summaries, predictions, and upset diagnostics.
- `figures/`: eight report-ready graphics, including validation, calibration, season stability, and the explanatory regression tree.

## Interpretation warning

All matchup predictors are computed from games completed before the NCAA tournament. Do not add tournament box scores, rounds won, or later-season rankings to pre-game predictions. Adjusted R-squared is reported as a descriptive complexity check; coefficient count is only an approximation to effective degrees of freedom for penalized models.
