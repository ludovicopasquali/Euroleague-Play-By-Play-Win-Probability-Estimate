from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def build_baseline_model(preprocessor) -> Pipeline:
    """Baseline pipeline using Logistic Regression for home win prediction."""
    estimator = LogisticRegression(max_iter=200, class_weight="balanced")
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
    return pipeline


def baseline_param_grid() -> Dict[str, list]:
    return {
        "model__C": [0.1, 1.0, 5.0],
        "model__penalty": ["l2"],
    }


def build_candidate_models(preprocessor, cv: int = 3) -> Dict[str, GridSearchCV]:
    """Set up baseline and alternative ML models with small grids suitable for class demos."""
    gbr = Pipeline(
        steps=[("preprocess", preprocessor), ("model", GradientBoostingClassifier(random_state=42))]
    )

    candidates = {
        "logistic": GridSearchCV(
            build_baseline_model(preprocessor),
            param_grid=baseline_param_grid(),
            cv=cv,
            scoring="neg_brier_score",
        ),
        "gradient_boosting": GridSearchCV(
            gbr,
            param_grid={
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
            cv=cv,
            scoring="neg_brier_score",
        ),
    }
    return candidates


def fit_models(models: Dict[str, GridSearchCV], X, y, groups=None) -> Dict[str, GridSearchCV]:
    """Fit each model in the provided dictionary and return the fitted estimators."""
    fitted = {}
    for name, model in models.items():
        if groups is None:
            fitted[name] = model.fit(X, y)
        else:
            fitted[name] = model.fit(X, y, groups=groups)
    return fitted


def summarize_best_params(fitted_models: Dict[str, GridSearchCV]) -> Dict[str, Tuple[float, dict]]:
    """Return best scores and params for fitted GridSearchCV models."""
    summary: Dict[str, Tuple[float, dict]] = {}
    for name, model in fitted_models.items():
        best_score = float(model.best_score_) if hasattr(model, "best_score_") else np.nan
        best_params = model.best_params_ if hasattr(model, "best_params_") else {}
        summary[name] = (best_score, best_params)
    return summary


__all__ = [
    "build_baseline_model",
    "baseline_param_grid",
    "build_candidate_models",
    "fit_models",
    "summarize_best_params",
]
