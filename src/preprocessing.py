# src/preprocessing.py
from __future__ import annotations
import pandas as pd
import numpy as np
import re

from .config import TARGET_COL, DATE_COL, DAY_COL

def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mirror notebook behavior:
    - require DATE_COL and TARGET_COL
    - do NOT pre-fill categorical NaNs (because missing flags use isna())
    """
    df = df.copy()
    df = df.dropna(subset=[DATE_COL, TARGET_COL])
    return df

def time_split_last_days(df: pd.DataFrame, test_days: int = 7, val_days: int = 7):
    """
    EXACT notebook split:
    - sort by INSTANCE_DATE
    - DATE = dt.date
    - test = last 7 unique days
    - val  = previous 7 unique days
    - train = rest
    """
    df = df.sort_values(DATE_COL).reset_index(drop=True).copy()
    df[DAY_COL] = df[DATE_COL].dt.date

    unique_days = sorted(df[DAY_COL].dropna().unique())
    if len(unique_days) < (test_days + val_days + 1):
        raise ValueError(f"Not enough unique days for split. Have {len(unique_days)} days.")

    test_set = set(unique_days[-test_days:])
    val_set  = set(unique_days[-(test_days + val_days):-test_days])

    test_df  = df[df[DAY_COL].isin(test_set)].copy()
    val_df   = df[df[DAY_COL].isin(val_set)].copy()
    train_df = df[~df[DAY_COL].isin(test_set | val_set)].copy()

    return train_df, val_df, test_df

def parse_rooms(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if "studio" in s:
        return 0.0
    m = re.search(r"(\d+)", s)
    return float(m.group(1)) if m else np.nan

def parse_parking(x):
    if pd.isna(x):
        return np.nan
    m = re.search(r"(\d+)", str(x))
    return float(m.group(1)) if m else np.nan

def add_features(d: pd.DataFrame) -> pd.DataFrame:
    """
    EXACT notebook add_features():
    - numeric coercion for ACTUAL_AREA / PROCEDURE_AREA
    - ROOMS_COUNT / PARKING_COUNT parsing
    - AREA_RATIO
    - MONTH/DOW + cyclical encoding
    - missing flags (isna()) for location fields
    """
    d = d.copy()

    d["ACTUAL_AREA"] = pd.to_numeric(d.get("ACTUAL_AREA"), errors="coerce")
    d["PROCEDURE_AREA"] = pd.to_numeric(d.get("PROCEDURE_AREA"), errors="coerce")

    d["ROOMS_COUNT"] = d.get("ROOMS_EN").apply(parse_rooms) if "ROOMS_EN" in d.columns else np.nan
    d["PARKING_COUNT"] = d.get("PARKING").apply(parse_parking) if "PARKING" in d.columns else np.nan

    d["AREA_RATIO"] = d["ACTUAL_AREA"] / (d["PROCEDURE_AREA"] + 1e-6)

    dt = d[DATE_COL]
    d["MONTH"] = dt.dt.month
    d["DOW"] = dt.dt.dayofweek
    d["MONTH_SIN"] = np.sin(2*np.pi*d["MONTH"]/12)
    d["MONTH_COS"] = np.cos(2*np.pi*d["MONTH"]/12)
    d["DOW_SIN"] = np.sin(2*np.pi*d["DOW"]/7)
    d["DOW_COS"] = np.cos(2*np.pi*d["DOW"]/7)

    for c in ["NEAREST_METRO_EN", "NEAREST_MALL_EN", "NEAREST_LANDMARK_EN", "PROJECT_EN"]:
        if c in d.columns:
            d[f"{c}_MISSING"] = d[c].isna().astype(int)

    return d
