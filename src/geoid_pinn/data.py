"""Louisiana study-data loading helpers.

The loader is included for reproducibility, but the underlying tract-level
file is not redistributed until its data-use terms are documented.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


LOUISIANA_FILENAMES = (
    "LousianaWeeklyCasesByCensusTract2022_date_20231001.dta",
    "LouisianaWeeklyCasesByCensusTract2022_date_20231001.dta",
)


def find_louisiana_dta(root: Path) -> Path:
    """Find the original Louisiana weekly tract-level Stata file."""
    root = Path(root).expanduser().resolve()
    search_dirs = (root, root / "Raw_data", root / "raw", root / "Raw Data")
    for directory in search_dirs:
        for filename in LOUISIANA_FILENAMES:
            candidate = directory / filename
            if candidate.is_file():
                return candidate
    for filename in LOUISIANA_FILENAMES:
        matches = list(root.rglob(filename))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Louisiana weekly case file was not found under {root}. "
        "Set LOUISIANA_DATA_ROOT to the local data directory."
    )


def load_parish_weekly(dta_path: Path) -> pd.DataFrame:
    """Aggregate tract-level weekly cases and tests to parish-week rows."""
    raw = pd.read_stata(dta_path, convert_categoricals=False)
    required = {
        "tract_fips10",
        "dateforstartofweek",
        "weeklycasecount",
        "weeklytestcount",
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Raw data missing required columns: {missing}")

    raw["parish_fips"] = raw["tract_fips10"].astype(str).str[:5].str.zfill(5)
    raw["week"] = pd.to_datetime(raw["dateforstartofweek"])
    raw["cases"] = pd.to_numeric(raw["weeklycasecount"], errors="coerce").fillna(0)
    raw["tests"] = pd.to_numeric(raw["weeklytestcount"], errors="coerce").fillna(0)
    raw[["cases", "tests"]] = raw[["cases", "tests"]].clip(lower=0)

    group_columns: list[str] = ["parish_fips", "week"]
    if "parish" in raw.columns:
        group_columns.insert(1, "parish")
    return (
        raw.groupby(group_columns, as_index=False)
        .agg(cases=("cases", "sum"), tests=("tests", "sum"))
        .sort_values(["week", "parish_fips"])
        .reset_index(drop=True)
    )


def require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise a clear error when a result or input table is incomplete."""
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
