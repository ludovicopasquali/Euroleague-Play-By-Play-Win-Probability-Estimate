from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from src.utils.paths import RAW_DATA_DIR

REQUIRED_COLUMNS: Sequence[str] = (
    "season_id",
    "game_id",
    "action_number",
    "period",
    "home_score",
    "away_score",
    "remaining_period_time",
    "type",
    "team_id",
    "opponent_id",
)


def load_play_by_play(
    path: Path | str,
    required_columns: Iterable[str] | None = None,
    dtype_overrides: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Load play-by-play CSV, validate expected columns, and keep original schema."""
    csv_path = Path(path)
    df = pd.read_csv(csv_path)

    if dtype_overrides:
        # Never coerce identifier columns so notebook truth tables reflect the exact CSV contents.
        overrides = {
            column: dtype
            for column, dtype in dtype_overrides.items()
            if column not in {"game_id", "home_team_id", "away_team_id"}
        }
        if overrides:
            df = df.astype(overrides)

    columns_to_check = list(required_columns) if required_columns else list(REQUIRED_COLUMNS)
    missing = [col for col in columns_to_check if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in {csv_path}: {missing}")

    return df


def example_path() -> Path:
    """Return the bundled sample raw CSV path."""
    return RAW_DATA_DIR / "play_by_play_combined.csv"


__all__ = ["load_play_by_play", "example_path", "REQUIRED_COLUMNS"]
