# NBA 2K ratings → team win percentage

This project builds a 300-row matrix (30 NBA teams × 10 completed seasons,
2016-17 through 2025-26). Each row contains the team's regular-season win
percentage and its 10 highest HoopsHype NBA 2K ratings, player names, and
corresponding positions.

## NBA scouting dashboard

The Next.js app also includes a scouting report at `/scout` with:

- all 30 current roster snapshots, ESPN team logos, and player headshots;
- rating-ranked projected starters and second units (10 players per team);
- player shooting-zone volume and accuracy from the 2025-26 regular season;
- team defensive shooting concessions by zone;
- player-level on-ball matchup results where the source has a qualifying sample.

The shooting and matchup summaries are built from the `shotdetail_2025` and
`matchups_2025` archives in
[`shufinskiy/nba_data`](https://github.com/shufinskiy/nba_data). Defensive shot
coordinates in that source are team-level; player defense is therefore shown
separately from the court map rather than inferred. Live day-by-day injury data
is intentionally not connected in this version.

Regenerate the compact browser data file with:

```bash
cd dashboard-app
npm run build:scout-data
```

## Data sources

- **Player names and ratings:** HoopsHype's historical NBA 2K player pages.
- **Season-specific positions:** the roster table on each Wikipedia team-season
  page. When a player is absent from that roster snapshot (usually because of a
  trade), the scraper uses the position on the player's HoopsHype profile and
  records that fact in `position_source_1` … `position_source_10`.
- **Wins, losses, and win percentage:** ESPN's public NBA standings endpoint.

The generated `data/nba_2k_team_seasons.csv` is the only input read by model
training. Network access is not used by `train_models.py` except when TabFM must
download its pretrained checkpoint.

## Run

```bash
python3 scrape_dataset.py
python3 train_models.py --tabfm auto
```

The test is time-aware: 2016-17 through 2023-24 are training data; 2024-25 and
2025-26 are held out. Classical model selection uses grouped cross-validation
by season on the training portion, not the test set.

Outputs:

- `data/nba_2k_team_seasons.csv` — complete matrix CSV
- `artifacts/model_metrics.csv` — MAE, RMSE, and R² for all completed models
- `artifacts/test_predictions.csv` — held-out predictions by team-season
- `artifacts/best_classical_model.joblib` — reusable selected sklearn pipeline
- `artifacts/run_summary.json` — exact split and TabFM status

## Historical win-total backtest

The browser-pulled Basketball Reference preseason lines are stored in
`data/nba_historical_win_totals.csv`. A walk-forward backtest uses 2013-14
through 2015-16 as burn-in data, then retrains the selected model using only
prior seasons before predicting each season from 2016-17 through 2025-26:

```bash
python3 backtest_win_totals.py --model extra_trees --stake 10 --american-odds -110
```

For the strictest check, select the best classical model separately before each
season using grouped cross-validation on prior seasons only:

```bash
python3 backtest_win_totals.py --model walk_forward_cv --stake 10 \
  --american-odds=-110 --output-dir artifacts/win_total_backtest_walkforward_cv
```

The historical tables do not include side-specific Over/Under prices, so the
simulation uses a transparent standard price of -110. The unexpectedly
shortened 2019-20 season is voided and all stakes are returned, consistent with
the common sportsbook minimum-games rule. Detailed bets and yearly summaries
are written under `artifacts/win_total_backtest/`.

TabFM is Google's zero-shot tabular foundation model. Its source is Apache-2.0,
but its default pretrained weights have a separate non-commercial,
non-production license. Install it with a backend if needed:

```bash
pip install "tabfm[jax]>=1.0.0"
```
