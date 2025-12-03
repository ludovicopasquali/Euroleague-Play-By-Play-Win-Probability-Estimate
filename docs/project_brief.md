# Project Brief

## Overview
- **Problem framing:** Estimate the probability that the home team wins a Euroleague game at each play-by-play action.
- **Inputs:** Anonymized play-by-play (PBP) logs with timestamps, running scores, event types, and the team who triggered each action (no player-identifying data).
- **Outputs:** Trained models plus evaluation notebooks/figures that quantify calibration and discrimination (Brier score, ROC AUC).

## Data scope
- Raw CSV: one row per PBP action with `season_id`, `game_id`, `period`, `remaining_period_time`, `home_score`, `away_score`, `type`, and the triggering `team_id`.
- Feature matrix (`features_full.csv`): enriches the raw feed with
  - Remaining absolute game time and running score differential.
  - Cumulative counts for 2FG/3FG attempts/makes, rebounds, fouls, turnovers, steals, assists, time-outs, etc. from the home and away perspectives.
  - Interaction features (`remaining_time_transformed`, `time_point_diff_interaction`, `home_prior`) plus the advanced efficiency/momentum metrics derived from those stats (eFG%, TS%, rebounding rates, turnover rates, free-throw rate, `score_diff_rolling`, `lead_changes`).
  - A `home_win` label broadcast to every action within a game.

## Modeling plan
- **Baseline:** Logistic regression trained on the standardized feature matrix. We rebalance the training fold with SMOTE to address the slight class imbalance.
- **Alternative (raw features):** XGBoost classifier tuned over a compact grid (`n_estimators`, `learning_rate`, `max_depth`, sampling ratios, `reg_lambda`). Group-aware splits ensure each game stays in a single fold.
- **Alternative (advanced features):** A second XGBoost model trained on the advanced efficiency/momentum feature set to see if higher-order transformations boost calibration/AUC.
- **Training flow:** Notebook 03 performs the hyperparameter search for both XGBoost variants, fits all models on the resampled training fold, and saves the fitted scalers + models + metadata under `models/`.
- **Evaluation:** Notebook 04 reloads the saved artifacts, regenerates the same train/test split by `game_id`, and reports Brier/AUC plus calibration and ROC plots without retraining.

## Validation & reporting
- Splitting strategy: `GroupShuffleSplit` with `test_size=0.2`, `random_state=42`, grouped by `game_id`.
- Metrics: Brier score (probability calibration) and ROC AUC (ranking quality) for logistic, tuned XGBoost (base features), and tuned XGBoost (advanced features).
- Visuals: Calibration curves and ROC curves stored under `reports/figures/`.
