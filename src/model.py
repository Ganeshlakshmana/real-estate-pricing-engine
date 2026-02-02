# src/model.py
from __future__ import annotations

import argparse
import json
import os
import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error

from .config import (
    TARGET_COL, SEGMENT_COL, LAND_VALUE, TRAIN_FRAC, VAL_FRAC, SEED,
    FEATURE_COLS, API_BASE_FEATURES
)
from .preprocessing import load_csv, basic_clean, time_split


# ---------------- Metrics ----------------
def rmse(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), 1e-6)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)

def eval_metrics(y_true_log, y_pred_log):
    out = {
        "R2_log": float(r2_score(y_true_log, y_pred_log)),
        "RMSE_log": rmse(y_true_log, y_pred_log),
    }
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    out.update({
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MedAE": float(median_absolute_error(y_true, y_pred)),
        "MAPE_%": float(mape(y_true, y_pred)),
    })
    return out


# ---------------- Priors (Notebook-equivalent) ----------------
def _make_agg_map(train_df: pd.DataFrame, group_cols: list[str]) -> dict:
    """
    Train-only aggregation map: key -> {mean_log, median_log, count, logcount}
    """
    g = train_df.groupby(group_cols)[TARGET_COL].agg(["mean", "median", "count"]).reset_index()
    g["mean_log"] = np.log1p(g["mean"].astype(float))
    g["median_log"] = np.log1p(g["median"].astype(float))
    g["logcount"] = np.log1p(g["count"].astype(float))

    out = {}
    for _, r in g.iterrows():
        key = tuple(r[c] for c in group_cols)
        out[key] = {
            "mean_log": float(r["mean_log"]),
            "median_log": float(r["median_log"]),
            "count": float(r["count"]),
            "logcount": float(r["logcount"]),
        }
    return out

def _apply_agg_features(df: pd.DataFrame, group_cols: list[str], agg_map: dict, prefix: str, fallback: dict) -> pd.DataFrame:
    df = df.copy()

    def lookup(row):
        key = tuple(row.get(c, "UNKNOWN") for c in group_cols)
        return agg_map.get(key, fallback)

    vals = df.apply(lookup, axis=1)
    df[f"{prefix}_mean_log"]   = [v["mean_log"] for v in vals]
    df[f"{prefix}_median_log"] = [v["median_log"] for v in vals]
    df[f"{prefix}_count"]      = [v["count"] for v in vals]
    df[f"{prefix}_logcount"]   = [v["logcount"] for v in vals]
    return df


# ---------------- Feature builder ----------------
def make_Xy(df: pd.DataFrame):
    df = df.copy()

    # Ensure base API fields exist (so priors can compute and model always has stable schema)
    for c in API_BASE_FEATURES:
        if c not in df.columns:
            df[c] = "UNKNOWN"

    # Ensure full feature schema exists (includes prior cols; these may be added earlier)
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0 if c.endswith(("mean_log", "median_log", "count", "logcount")) else "UNKNOWN"

    X = df[FEATURE_COLS].copy()

    # Fill cat columns safely
    for c in X.select_dtypes(include=["object", "string"]).columns:
        X[c] = X[c].fillna("UNKNOWN").astype(str)

    # Fill numeric columns
    for c in X.select_dtypes(include=[np.number]).columns:
        if X[c].notna().any():
            X[c] = X[c].fillna(X[c].median())
        else:
            X[c] = X[c].fillna(0.0)

    y = np.log1p(df[TARGET_COL].values)
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    return X, y, cat_cols


# ---------------- Segmentation ----------------
def segment_masks(df: pd.DataFrame):
    is_land = (df[SEGMENT_COL].astype(str) == str(LAND_VALUE))
    is_main = ~is_land
    return is_main, is_land


# ---------------- Model training ----------------
def train_catboost(X_train, y_train, X_val, y_val, cat_cols, params: dict):
    model = CatBoostRegressor(**params)
    model.fit(
        X_train, y_train,
        cat_features=cat_cols,
        eval_set=(X_val, y_val),
        use_best_model=True,
        verbose=200
    )
    return model


def train_segmented(df: pd.DataFrame, reports_dir: str):
    train_df, val_df, test_df = time_split(df, TRAIN_FRAC, VAL_FRAC)

    train_main, train_land = segment_masks(train_df)
    val_main, val_land     = segment_masks(val_df)
    test_main, test_land   = segment_masks(test_df)

    # ---------- Build priors for MAIN only (leakage-safe) ----------
    train_main_df = train_df.loc[train_main].copy()
    val_main_df   = val_df.loc[val_main].copy()
    test_main_df  = test_df.loc[test_main].copy()

    # fallback from MAIN train
    global_mean = float(train_main_df[TARGET_COL].mean())
    global_median = float(train_main_df[TARGET_COL].median())
    global_count = float(len(train_main_df))
    fallback = {
        "mean_log": float(np.log1p(global_mean)),
        "median_log": float(np.log1p(global_median)),
        "count": float(global_count),
        "logcount": float(np.log1p(global_count)),
    }

    agg_specs = [
        (["AREA_EN"], "agg_area"),
        (["PROJECT_EN"], "agg_project"),
        (["AREA_EN", "PROP_TYPE_EN"], "agg_area_prop"),
    ]

    agg_maps = {}
    for cols, prefix in agg_specs:
        agg_map = _make_agg_map(train_main_df, cols)
        agg_maps[prefix] = {"cols": cols, "map": agg_map}

        train_main_df = _apply_agg_features(train_main_df, cols, agg_map, prefix, fallback)
        val_main_df   = _apply_agg_features(val_main_df, cols, agg_map, prefix, fallback)
        test_main_df  = _apply_agg_features(test_main_df, cols, agg_map, prefix, fallback)

    print("Aggregate priors added for MAIN only.")

    # ---------- CatBoost params ----------
    common = dict(
        loss_function="RMSE",
        iterations=1500,
        learning_rate=0.03,
        depth=8,
        random_seed=SEED,
        eval_metric="RMSE",
        od_type="Iter",
        od_wait=80
    )

    # MAIN model trains on MAIN with priors
    X_tr_m, y_tr_m, cat_m = make_Xy(train_main_df)
    X_va_m, y_va_m, _     = make_Xy(val_main_df)
    model_main = train_catboost(X_tr_m, y_tr_m, X_va_m, y_va_m, cat_m, common)

    # LAND model (no priors boost required; schema still includes prior cols but they will be defaults)
    land_params = common | dict(depth=6, learning_rate=0.05, iterations=1200)
    train_land_df = train_df.loc[train_land].copy()
    val_land_df   = val_df.loc[val_land].copy()
    test_land_df  = test_df.loc[test_land].copy()

    X_tr_l, y_tr_l, cat_l = make_Xy(train_land_df)
    X_va_l, y_va_l, _     = make_Xy(val_land_df)
    model_land = train_catboost(X_tr_l, y_tr_l, X_va_l, y_va_l, cat_l, land_params)

    # ---------- Evaluate ----------
    metrics = {"MAIN": {}, "LAND": {}}

    # MAIN
    X, y, _ = make_Xy(val_main_df)
    p = model_main.predict(X)
    metrics["MAIN"]["val"] = eval_metrics(y, p)

    X, y, _ = make_Xy(test_main_df)
    p = model_main.predict(X)
    metrics["MAIN"]["test"] = eval_metrics(y, p)

    # LAND
    X, y, _ = make_Xy(val_land_df)
    p = model_land.predict(X)
    metrics["LAND"]["val"] = eval_metrics(y, p)

    X, y, _ = make_Xy(test_land_df)
    p = model_land.predict(X)
    metrics["LAND"]["test"] = eval_metrics(y, p)

    # ---------- Save reports ----------
    if reports_dir:
        os.makedirs(reports_dir, exist_ok=True)
        with open(os.path.join(reports_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # feature importance for MAIN
        try:
            fi = model_main.get_feature_importance()
            fi_df = pd.DataFrame({"feature": FEATURE_COLS, "importance": fi}).sort_values("importance", ascending=False)
            fi_df.to_csv(os.path.join(reports_dir, "feature_importance.csv"), index=False)
        except Exception:
            pass

    bundle = {
        "segment_col": SEGMENT_COL,
        "land_value": LAND_VALUE,
        "feature_cols": FEATURE_COLS,
        "api_base_features": API_BASE_FEATURES,
        "agg_maps": agg_maps,     # ✅ used by API inference
        "model_main": model_main,
        "model_land": model_land,
        "metrics": metrics,
    }
    return bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["train", "evaluate"])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="models/trained_model.pkl")
    ap.add_argument("--reports", default="reports")
    args = ap.parse_args()

    df = load_csv(args.csv)
    df = basic_clean(df)

    if args.command == "train":
        bundle = train_segmented(df, reports_dir=args.reports)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        joblib.dump(bundle, args.out)
        print(f"✅ Saved model bundle -> {args.out}")
        print(f"✅ Metrics -> {args.reports}/metrics.json")
        print(f"✅ Feature importance -> {args.reports}/feature_importance.csv")

    elif args.command == "evaluate":
        bundle = joblib.load(args.out)
        print(json.dumps(bundle.get("metrics", {}), indent=2))


if __name__ == "__main__":
    main()
