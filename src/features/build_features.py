from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

PERIOD_LENGTH = 600.0
EVENT_SPECS: Dict[str, Tuple[str, ...]] = {
    "2FGA": ("2FGA", "2FGM"),
    "2FGM": ("2FGM",),
    "3FGA": ("3FGA", "3FGM"),
    "3FGM": ("3FGM",),
    "OREB": ("OREB",),
    "DREB": ("DREB",),
    "PF": ("PF",),
    "PFD": ("PFD",),
    "BLK": ("BLK",),
    "BLKA": ("BLKA",),
    "TOV": ("TOV",),
    "FTM": ("FTM",),
    "FTA": ("FTA", "FTM"),
    "TOUT": ("TOUT",),
    "AST": ("AST",),
    "STL": ("STL",),
}
EVENT_FEATURE_COLUMNS: Tuple[str, ...] = tuple(
    f"{metric}_{suffix}"
    for metric in (
        "2FGA",
        "2FGM",
        "3FGA",
        "3FGM",
        "OREB",
        "DREB",
        "PF",
        "PFD",
        "BLK",
        "BLKA",
        "TOV",
        "FTM",
        "FTA",
        "TOUT",
        "AST",
        "STL",
    )
    for suffix in ("h", "a")
)
MODEL_FEATURE_COLUMNS: Tuple[str, ...] = (
    "home_point_diff",
    *EVENT_FEATURE_COLUMNS,
    "remaining_time_transformed",
    "time_point_diff_interaction",
    "home_prior",
)

ADVANCED_FEATURE_COLUMNS: Tuple[str, ...] = (
    "eFG%_h",
    "eFG%_a",
    "TS%_h",
    "TS%_a",
    "OREB%_h",
    "OREB%_a",
    "DREB%_h",
    "DREB%_a",
    "TOV%_h",
    "TOV%_a",
    "FTr_h",
    "FTr_a",
    "score_diff_rolling",
    "lead_changes",
)


def compute_remaining_time(row: pd.Series) -> float:
    """Compute absolute remaining game time from period and remaining period time."""
    return (PERIOD_LENGTH * (4 - row["period"])) + float(row["remaining_period_time"])


def label_home_win(df: pd.DataFrame) -> pd.DataFrame:
    """Broadcast the final game result (home_win) across all rows per game."""
    features = df.copy()
    features["home_win"] = 0
    for game_id, game_df in features.groupby("game_id"):
        final_row = game_df.iloc[-1]
        home_win = int(final_row["home_score"] > final_row["away_score"])
        features.loc[features["game_id"] == game_id, "home_win"] = home_win
    return features


def _resolve_type_suffix(series: pd.Series) -> pd.Series:
    suffix = series.map({"HOME": "h", "AWAY": "a"})
    return suffix.fillna("na")


def _add_event_counts(features: pd.DataFrame, side_col: str) -> pd.DataFrame:
    for metric, raw_events in EVENT_SPECS.items():
        for side, suffix in (("HOME", "h"), ("AWAY", "a")):
            column_name = f"{metric}_{suffix}"
            indicator = (
                (features["type"].isin(raw_events)) & (features[side_col] == side)
            ).astype(int)
            cumulative = indicator.groupby(features["game_id"]).cumsum()
            features[column_name] = cumulative
            features[column_name] = (
                features.groupby("game_id")[column_name].transform("ffill").fillna(0).astype(int)
            )
    return features


def _add_interactions(features: pd.DataFrame) -> pd.DataFrame:
    epsilon = 1e-6
    features["remaining_time_transformed"] = 1.0 / (features["remaining_time"] + epsilon)
    features["time_point_diff_interaction"] = (
        features["remaining_time_transformed"] * features["home_point_diff"]
    )
    features["home_prior"] = 0.5
    return features


def build_feature_dataframe(df: pd.DataFrame, side_col: str = "team_side") -> pd.DataFrame:
    """Construct the full feature set (cumulative stats + interactions) used across notebooks."""
    if side_col not in df.columns:
        raise KeyError(f"Expected '{side_col}' column in play-by-play DataFrame.")

    features = df.copy()
    features["type_h"] = features["type"] + "_" + _resolve_type_suffix(features[side_col])
    features["home_point_diff"] = features["home_score"] - features["away_score"]
    features["remaining_time"] = features.apply(compute_remaining_time, axis=1)
    features = _add_event_counts(features, side_col=side_col)
    features = label_home_win(features)
    features = _add_interactions(features)
    features = _add_advanced_metrics(features)

    desired_order: Iterable[str] = list(df.columns) + [
        "home_win",
        "type_h",
        "home_point_diff",
        "remaining_time",
        *EVENT_FEATURE_COLUMNS,
        "remaining_time_transformed",
        "time_point_diff_interaction",
        "home_prior",
        *ADVANCED_FEATURE_COLUMNS,
    ]
    ordered_columns = [col for col in desired_order if col in features.columns]
    return features[ordered_columns]


def separate_categorical_numeric(df: pd.DataFrame, target_column: str = "home_win"):
    """Identify categorical and numeric columns for preprocessing."""
    categorical_cols = [col for col in df.columns if df[col].dtype == "object" and col != target_column]
    numeric_cols = [col for col in df.columns if df[col].dtype != "object" and col != target_column]
    return categorical_cols, numeric_cols


__all__ = [
    "build_feature_dataframe",
    "separate_categorical_numeric",
    "label_home_win",
    "EVENT_FEATURE_COLUMNS",
    "MODEL_FEATURE_COLUMNS",
    "ADVANCED_FEATURE_COLUMNS",
]
def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator.astype(float)
    denom = denominator.replace(0, np.nan)
    return result.divide(denom).fillna(0.0)


def _add_advanced_metrics(features: pd.DataFrame) -> pd.DataFrame:
    feats = features.copy()
    for prefix in ("h", "a"):
        fg_attempts = feats[f"2FGA_{prefix}"] + feats[f"3FGA_{prefix}"]
        feats[f"eFG%_{prefix}"] = _safe_divide(
            feats[f"2FGM_{prefix}"] + 0.5 * feats[f"3FGM_{prefix}"],
            fg_attempts,
        )
        scoring = feats["home_score"] if prefix == "h" else feats["away_score"]
        feats[f"TS%_{prefix}"] = _safe_divide(
            scoring * 1.0,
            2 * (fg_attempts + 0.44 * feats[f"FTA_{prefix}"]),
        )
        feats[f"OREB%_{prefix}"] = _safe_divide(
            feats[f"OREB_{prefix}"],
            feats[f"OREB_{prefix}"] + feats[f"DREB_{'a' if prefix == 'h' else 'h'}"],
        )
        feats[f"DREB%_{prefix}"] = _safe_divide(
            feats[f"DREB_{prefix}"],
            feats[f"DREB_{prefix}"] + feats[f"OREB_{'a' if prefix == 'h' else 'h'}"],
        )
        possession_terms = (
            feats[f"2FGA_{prefix}"]
            + feats[f"3FGA_{prefix}"]
            + 0.44 * feats[f"FTA_{prefix}"]
            + feats[f"TOV_{prefix}"]
        )
        feats[f"TOV%_{prefix}"] = _safe_divide(feats[f"TOV_{prefix}"], possession_terms)
        feats[f"FTr_{prefix}"] = _safe_divide(feats[f"FTA_{prefix}"], fg_attempts)
    feats["score_diff_rolling"] = _compute_score_diff_rolling(feats)
    feats["lead_changes"] = _compute_lead_changes(feats)
    return feats


def _compute_score_diff_rolling(features: pd.DataFrame, window_seconds: float = 120.0) -> pd.Series:
    result = pd.Series(index=features.index, dtype=float)
    for _, game_df in features.groupby("game_id", sort=False):
        ordered = game_df.sort_values("remaining_time", ascending=False).copy()
        elapsed = ordered["remaining_time"].max() - ordered["remaining_time"]
        values = ordered["home_point_diff"].values
        rolling = np.zeros_like(values, dtype=float)
        for idx in range(len(values)):
            start_time = elapsed.iloc[idx] - window_seconds
            mask = elapsed.between(max(start_time, 0), elapsed.iloc[idx])
            window_vals = values[mask.values]
            if len(window_vals) > 1:
                rolling[idx] = window_vals[-1] - window_vals[0]
            else:
                rolling[idx] = values[idx]
        result.loc[ordered.index] = rolling
    return result.fillna(0.0)


def _compute_lead_changes(features: pd.DataFrame) -> pd.Series:
    lead_changes = np.zeros(len(features), dtype=int)
    offset = 0
    for _, game_df in features.groupby("game_id", sort=False):
        count = 0
        last_sign = 0
        for idx, value in enumerate(game_df["home_point_diff"].values):
            sign = 0 if value == 0 else (1 if value > 0 else -1)
            if sign != 0:
                if last_sign == 0 or sign != last_sign:
                    if last_sign != 0:
                        count += 1
                    else:
                        count += 1
                last_sign = sign
            lead_changes[offset + idx] = count
        offset += len(game_df)
    return pd.Series(lead_changes, index=features.index)
