from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score, accuracy_score
from sklearn.model_selection import cross_val_score


def classification_metrics(y_true, y_pred_proba, threshold: float = 0.5) -> Dict[str, float]:
    """Compute common classification metrics for win probability."""
    preds = (y_pred_proba >= threshold).astype(int)
    return {
        "brier": brier_score_loss(y_true, y_pred_proba),
        "auc": roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else np.nan,
        "accuracy": accuracy_score(y_true, preds),
    }


def cross_validate_model(model, X, y, cv: int = 5, scoring: str = "neg_brier_score") -> np.ndarray:
    """Run cross-validation on a model and return scores."""
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return scores


def extract_feature_importance(
    estimator, feature_names: Iterable[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return feature importance or coefficients if available."""
    names = np.array(list(feature_names))
    if hasattr(estimator, "feature_importances_"):
        importances = np.array(estimator.feature_importances_)
    elif hasattr(estimator, "coef_"):
        importances = np.array(estimator.coef_).ravel()
    else:
        importances = np.zeros(len(names))
    return names, importances


__all__ = [
    "classification_metrics",
    "cross_validate_model",
    "extract_feature_importance",
]
