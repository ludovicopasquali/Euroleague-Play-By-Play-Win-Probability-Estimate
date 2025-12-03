from typing import Iterable, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_TARGET = "home_win"


def split_features_target(
    df: pd.DataFrame, target: str = DEFAULT_TARGET
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into feature matrix X and target vector y."""
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in DataFrame")
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def build_preprocessor(
    categorical_cols: Iterable[str], numeric_cols: Iterable[str]
) -> ColumnTransformer:
    """Construct a preprocessing pipeline for categorical and numeric features."""
    categorical = OneHotEncoder(handle_unknown="ignore")
    numeric = StandardScaler()

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical, list(categorical_cols)),
            ("numeric", numeric, list(numeric_cols)),
        ]
    )


def make_train_val_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Simple train/validation split wrapper with a fixed random seed."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


__all__ = [
    "DEFAULT_TARGET",
    "split_features_target",
    "build_preprocessor",
    "make_train_val_split",
]
