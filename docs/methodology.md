# Methodology Notes

These notes document the full workflow so that anyone unfamiliar with the original internal project can quickly understand (and rerun) the public-ready version in this repo.

## Data pipeline
1. **Ingest:** Read the combined play-by-play export (`data/raw/play_by_play_combined.csv`). Validate that each row contains the canonical columns (`season_id`, `game_id`, scores, timing, event type, team identifiers).
2. **Augment raw columns:** Parse the `game_id` string to recover `home_team_name`/`away_team_name`, and infer a `team_side` flag (`HOME`, `AWAY`, `NA` for neutral events). Cast action numbers, scores, and clocks to numeric dtypes.
3. **Feature build (`src/features.build_feature_dataframe`):**
   - Compute remaining absolute seconds (`remaining_time`) and the running score differential (`home_point_diff`).
   - Track cumulative counts for 2FG/3FG attempts/makes, rebounds (O/D), personal fouls drawn/committed, turnovers, assists, steals, blocks, block-against, and time-outs for both home and away teams.
   - Create engineered interactions: `remaining_time_transformed = 1 / (remaining_time + 1e-6)`, `time_point_diff_interaction`, and a neutral `home_prior`.
   - Derive advanced efficiency and momentum features (eFG%, TS%, offensive/defensive rebounding rates, turnover rates, free-throw rate, short-window score differential, and lead-change counts) that feed the second XGBoost variant.
   - Label every action in a game with the final outcome (`home_win`).
4. **Persist features:** Notebook 02 writes `data/processed/features_full.csv`, which becomes the single source-of-truth for modeling notebooks.

## Modeling
1. **Standardize + split:** Notebook 03 standardizes both the base and advanced feature matrices with `StandardScaler`, then applies `GroupShuffleSplit` (`test_size=0.2`, `random_state=42`) to keep each `game_id` in a single fold.
2. **Rebalancing:** Apply SMOTE on the training fold (per feature set) to balance the slight class imbalance before fitting models.
3. **Baseline model:** Logistic Regression (`sklearn`) trained on the scaled, balanced base feature set.
4. **Alternative models:** Two `xgboost.XGBClassifier` runs—one on the base features and one on the advanced feature set. Each candidate is tuned over a compact grid (`n_estimators`, `learning_rate`, `max_depth`, sampling ratios, `reg_lambda`) with group-aware CV (3 splits), using Brier score to pick the best configuration.
5. **Artifacts:** Notebook 03 saves both scalers, the logistic model, the two tuned XGBoost models, their CV results, and metadata (`models/training_metadata.json`) so downstream steps do not retrain from scratch.

## Evaluation & reporting
1. **Reload artifacts:** Notebook 04 loads `features_full.csv`, the persisted scaler, metadata, and both trained models.
2. **Recreate the split:** Using the same `GroupShuffleSplit` settings ensures the hold-out set matches the training notebook exactly.
3. **Metrics:** Compute and display Brier score and ROC AUC for logistic, base XGBoost, and advanced-feature XGBoost on the hold-out set.
4. **Plots:** Generate calibration and ROC curves (three lines per plot) and save them under `reports/figures/`.
5. **Reproducibility:** Because all random seeds are fixed (split + SMOTE + model initialization), rerunning the notebooks reproduces identical metrics/plots as long as the raw CSV does not change.

## Folder organisation
- `data/raw/` contains the combined PBP dataset, following the structure of the sample one.
- `data/processed/features_full.csv` is produced by notebook 02, using the dataset present in `data/raw/` folder.
- `models/` stores the serialized scaler, trained models, CV results, and metadata.
- `reports/figures/` holds calibration + ROC images.

Following this structure makes it easy to substitute your own play-by-play export while keeping the training/evaluation experience identical.