# Data Directory

This repository intentionally ships only tiny toy CSVs to illustrate the schema.

Included samples:
- `data/raw/sample_pbp.csv`: minimal play-by-play events with scores and event types.
- `data/processed/sample_features.csv`: engineered rows with cumulative features, advanced efficiency/momentum metrics, and a dummy `home_win` label at each action (matching the columns produced by notebook 02).

For your own runs:
- Place full raw play-by-play exports (e.g., `play_by_play_combined.csv`) in `data/raw/`. Keep the column names consistent or adjust `src/` accordingly.
- Write processed feature tables (e.g., `features_full.csv`) to `data/processed/`. The notebooks expect to find the real feature matrix there, while the git-tracked sample remains a lightweight placeholder.