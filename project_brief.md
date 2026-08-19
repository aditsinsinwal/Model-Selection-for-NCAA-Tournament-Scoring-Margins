# Project brief: Can the Favourite Be Trusted?

**Author:** Adit Sinsinwal (Student ID 20988864)

## Research question

Can regular-season team statistics and tournament seeds predict NCAA tournament scoring margin and identify favourites with elevated upset risk?

## Why it matters

A binary bracket pick hides uncertainty. A margin model can show how much separation exists between teams, while a vulnerability ranking can direct scouting or bracket analysis toward a manageable group of games.

## Data and response

- Source: historical men's March Machine Learning Mania files
- Regular-season games: 87,366
- NCAA tournament games: 1,048 from 2003-2018
- Response: Team 1 score minus Team 2 score
- Candidate predictors: 26 pre-tournament matchup differences
- Training: 2003-2014 (780 games)
- Validation: 2015 (67 games)
- Untouched test: 2016-2018 (201 games)

## Predictor groups

- Tournament seed
- Win rate, average margin, and recent form
- Offensive, defensive, and net efficiency
- Effective field-goal percentage and three-point rate
- Turnover and forced-turnover rates
- Offensive and defensive rebounding rates
- Free-throw, assist, steal, block, and foul rates
- Pace and opponent strength

## Methods

The project treats Ridge, LASSO, and Elastic Net as one penalized-regression framework and compares them with a cost-complexity-pruned regression tree. Hyperparameters are selected by validation MAE without consulting the 2016-2018 test tournaments.

## Main findings

| Model | Test MAE | Test R-squared | Adjusted R-squared | Winner accuracy |
|---|---:|---:|---:|---:|
| Seed-only linear | 9.39 | 0.272 | 0.268 | 70.6% |
| Ridge | **8.91** | **0.337** | 0.238 | **74.1%** |
| Elastic Net | 9.04 | 0.330 | 0.309 | 71.6% |
| LASSO | 9.03 | 0.329 | **0.312** | 72.1% |
| Regression tree | 9.76 | 0.223 | 0.191 | 68.2% |

Ridge was selected on validation MAE and also achieved the lowest pooled test MAE. It beat the seed-only MAE separately in 2016, 2017, and 2018, providing a simple check that the pooled gain was not created by one favourable tournament. The larger adjusted R-squared values for Elastic Net and LASSO reflect their much sparser solutions, not a change to the pre-specified selection rule.

Among 186 unequal-seed test games, the upset base rate was 29.0%. The most vulnerable fifth identified by Ridge had a 51.4% upset rate and the safest fifth had a 7.9% rate. Flagging the most vulnerable quarter produced 46.8% precision, 40.7% recall, and 1.61 lift.

## Defensible conclusion

Pre-tournament statistics explain about one-third of later scoring-margin variation and improve materially on seeds alone. The model is useful for ranking uncertainty, not guaranteeing a winner or establishing that any statistic causes an upset.
