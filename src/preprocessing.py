# src/preprocessing.py
from __future__ import annotations
import pandas as pd
import numpy as np
from .config import TARGET_COL, DATE_COL, DROP_COLS_ALWAYS

def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # parse date
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # drop rows missing date or target (for training)
    df = df.dropna(subset=[DATE_COL, TARGET_COL])

    # numeric sanity
    if "ACTUAL_AREA" in df.columns:
        df.loc[df["ACTUAL_AREA"] <= 0, "ACTUAL_AREA"] = np.nan

    # fill categorical NaNs (prevents CatBoost "bad object for id: nan")
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df[c] = df[c].fillna("UNKNOWN")

    # fill numeric NaNs with median (simple, robust default)
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        if c != TARGET_COL:
            med = df[c].median()
            df[c] = df[c].fillna(med)

    return df

def time_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15):
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df.iloc[:train_end].copy()
    val_df   = df.iloc[train_end:val_end].copy()
    test_df  = df.iloc[val_end:].copy()
    return train_df, val_df, test_df

def make_Xy(df: pd.DataFrame):
    drop_cols = [c for c in DROP_COLS_ALWAYS if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")
    y = np.log1p(df[TARGET_COL].values)  # train in log space
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    return X, y, cat_cols
