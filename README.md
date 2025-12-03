# Group Project 4: In-Game Win Probability (Euroleague Example)

This repository is a public-ready example for modeling in-game win probability from Euroleague play-by-play data. It keeps the code and notebooks compact so students can follow a supervised ML workflow without exposing proprietary datasets.

## Quickstart
1. Create an environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Drop your raw play-by-play CSVs into `data/raw/` (schema shown in `data/raw/sample_pbp.csv`).
3. Run notebooks in order to reproduce the workflow on your data:
   - `notebooks/01_data_overview.ipynb`
   - `notebooks/02_feature_engineering.ipynb`
   - `notebooks/03_modeling.ipynb` (trains and persists the logistic + XGBoost models)
   - `notebooks/04_model_evaluation.ipynb` (reloads the saved models and reports metrics)

## Repository Map
- `data/` – placeholder CSVs; add real PBP here (gitignored by default).
- `docs/` – project brief and methodology describing the modeling workflow.
- `models/` - artifacts stored here
- `notebooks/` – EDA → features → modeling → evaluation.
- `reports/` – generated figures/tables for the write-up.
- `src/` – reusable helpers for ingest, feature building, model training/evaluation.
- `requirements.txt` – minimal Python dependencies.
- `LICENSE` – MIT.

## Notes on Scope
- Focus is team/game-level win probability from play-by-play (no player-level compensation or sensitive stats).
- Baseline: logistic regression on engineered cumulative features.
- Alternatives: two XGBoost classifiers – one on the same engineered features, and one on the extended “advanced” feature set (efficiency + momentum metrics).
- Replace toy data with your own and adjust feature definitions as needed.
- Current executed run (for figures) uses the complete dataset for seasons: 2019-2020, 2020-2021, 2021-2022, 2022-2023, 2023-2024.

## Artifacts
- `models/` – serialized scalers/models/metadata produced by `03_modeling.ipynb`.
- `reports/figures/` – calibration and ROC plots saved by `04_model_evaluation.ipynb`.
