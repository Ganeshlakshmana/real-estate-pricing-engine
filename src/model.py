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
    TARGET_COL, DATE_COL, SEGMENT_COL, LAND_VALUE, DROP_COLS,
    AGG_SPECS, MAIN_PARAMS, LAND_PARAMS
)
from .preprocessing import load_csv, basic_clean, time_split_last_days, add_features


# ---------- metrics (same style as notebook) ----------
def mape(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), 1e-6)
    return float(np.mean(np.abs((y_true - y_pred)/denom))*100)

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

def eval_metrics_log(y_true_log, y_pred_log):
    # log-space
    out = {
        "R2_log": float(r2_score(y_true_log, y_pred_log)),
        "RMSE_log": float(rmse(y_true_log, y_pred_log)),
    }
    # original space
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    out.update({
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MedAE": float(median_absolute_error(y_true, y_pred)),
        "MAPE_%": float(mape(y_true, y_pred)),
    })
    return out


# ---------- priors (EXACT notebook naming + logic) ----------
def add_agg_features(train_df, apply_df, group_cols, target_col=TARGET_COL):
    tr = train_df.copy()
    tr["_ylog"] = np.log1p(tr[target_col].astype(float))

    stats = tr.groupby(group_cols)["_ylog"].agg(["median", "mean", "count"]).reset_index()
    prefix = "AGG_" + "_".join(group_cols)

    stats = stats.rename(columns={
        "median": f"{prefix}_ylog_median",
        "mean":   f"{prefix}_ylog_mean",
        "count":  f"{prefix}_count"
    })

    out = apply_df.merge(stats, on=group_cols, how="left")

    global_med = tr["_ylog"].median()
    global_mean = tr["_ylog"].mean()

    out[f"{prefix}_ylog_median"] = out[f"{prefix}_ylog_median"].fillna(global_med)
    out[f"{prefix}_ylog_mean"]   = out[f"{prefix}_ylog_mean"].fillna(global_mean)
    out[f"{prefix}_count"]       = out[f"{prefix}_count"].fillna(0).astype(float)
    out[f"{prefix}_logcount"]    = np.log1p(out[f"{prefix}_count"])
    return out


# ---------- build XY (EXACT notebook) ----------
def build_xy(df_):
    X = df_.drop(columns=[c for c in DROP_COLS if c in df_.columns], errors="ignore").copy()
    y_raw = df_[TARGET_COL].astype(float).values
    y_log = np.log1p(y_raw)
    return X, y_raw, y_log


def clean_for_tree(X_train, X_val, X_test):
    """
    EXACT notebook clean_for_tree():
    - cat cols = object dtype
    - fill cat missing with '__MISSING__' and cast to str
    - numeric to_numeric(errors='coerce')
    - median from train numeric only
    """
    X_train = X_train.copy()
    X_val   = X_val.copy()
    X_test  = X_test.copy()

    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    for c in cat_cols:
        X_train[c] = X_train[c].fillna("__MISSING__").astype(str)
        X_val[c]   = X_val[c].fillna("__MISSING__").astype(str)
        X_test[c]  = X_test[c].fillna("__MISSING__").astype(str)

    for c in num_cols:
        X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
        X_val[c]   = pd.to_numeric(X_val[c], errors="coerce")
        X_test[c]  = pd.to_numeric(X_test[c], errors="coerce")

    med = X_train[num_cols].median()
    X_train[num_cols] = X_train[num_cols].fillna(med)
    X_val[num_cols]   = X_val[num_cols].fillna(med)
    X_test[num_cols]  = X_test[num_cols].fillna(med)

    return X_train, X_val, X_test, cat_cols, num_cols, med


def train_segment_models(df: pd.DataFrame, reports_dir: str):
    # ---- exact notebook split ----
    train_df, val_df, test_df = time_split_last_days(df, test_days=7, val_days=7)

    # ---- exact notebook features ----
    train_df = add_features(train_df)
    val_df   = add_features(val_df)
    test_df  = add_features(test_df)

    # ---- segment ----
    train_land = train_df[train_df[SEGMENT_COL] == LAND_VALUE].copy()
    val_land   = val_df[val_df[SEGMENT_COL] == LAND_VALUE].copy()
    test_land  = test_df[test_df[SEGMENT_COL] == LAND_VALUE].copy()

    train_main = train_df[train_df[SEGMENT_COL] != LAND_VALUE].copy()
    val_main   = val_df[val_df[SEGMENT_COL] != LAND_VALUE].copy()
    test_main  = test_df[test_df[SEGMENT_COL] != LAND_VALUE].copy()

    # ---- MAIN priors (exact order) ----
    for cols, _name in AGG_SPECS:
        train_main = add_agg_features(train_main, train_main, cols)
        val_main   = add_agg_features(train_main, val_main, cols)
        test_main  = add_agg_features(train_main, test_main, cols)

    # ---- build/clean MAIN ----
    X_train_m, y_train_m_raw, y_train_m = build_xy(train_main)
    X_val_m,   y_val_m_raw,   y_val_m   = build_xy(val_main)
    X_test_m,  y_test_m_raw,  y_test_m  = build_xy(test_main)

    X_train_m, X_val_m, X_test_m, cat_cols_m, num_cols_m, med_m = clean_for_tree(X_train_m, X_val_m, X_test_m)
    cat_idx_m = [X_train_m.columns.get_loc(c) for c in cat_cols_m]

    main_model = CatBoostRegressor(**MAIN_PARAMS)
    main_model.fit(
        X_train_m, y_train_m,
        cat_features=cat_idx_m,
        eval_set=(X_val_m, y_val_m),
        use_best_model=True
    )

    # ---- build/clean LAND ----
    X_train_l, y_train_l_raw, y_train_l = build_xy(train_land)
    X_val_l,   y_val_l_raw,   y_val_l   = build_xy(val_land)
    X_test_l,  y_test_l_raw,  y_test_l  = build_xy(test_land)

    X_train_l, X_val_l, X_test_l, cat_cols_l, num_cols_l, med_l = clean_for_tree(X_train_l, X_val_l, X_test_l)
    cat_idx_l = [X_train_l.columns.get_loc(c) for c in cat_cols_l]

    land_model = CatBoostRegressor(**LAND_PARAMS)
    land_model.fit(
        X_train_l, y_train_l,
        cat_features=cat_idx_l,
        eval_set=(X_val_l, y_val_l),
        use_best_model=True
    )

    # ---- evaluate EXACT (log metrics + original MAE etc) ----
    metrics = {"MAIN": {}, "LAND": {}}

    pred_val_m_log  = main_model.predict(X_val_m)
    pred_test_m_log = main_model.predict(X_test_m)
    metrics["MAIN"]["val"]  = eval_metrics_log(y_val_m, pred_val_m_log)
    metrics["MAIN"]["test"] = eval_metrics_log(y_test_m, pred_test_m_log)

    pred_val_l_log  = land_model.predict(X_val_l)
    pred_test_l_log = land_model.predict(X_test_l)
    metrics["LAND"]["val"]  = eval_metrics_log(y_val_l, pred_val_l_log)
    metrics["LAND"]["test"] = eval_metrics_log(y_test_l, pred_test_l_log)

    # reports
    if reports_dir:
        os.makedirs(reports_dir, exist_ok=True)
        with open(os.path.join(reports_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # feature importance (MAIN)
        try:
            fi = main_model.get_feature_importance()
            fi_df = pd.DataFrame({"feature": X_train_m.columns, "importance": fi}).sort_values("importance", ascending=False)
            fi_df.to_csv(os.path.join(reports_dir, "feature_importance.csv"), index=False)
        except Exception:
            pass

    bundle = {
        "segment_col": SEGMENT_COL,
        "land_value": LAND_VALUE,

        # schema + cleaning info needed for API inference
        "feature_cols_main": X_train_m.columns.tolist(),
        "feature_cols_land": X_train_l.columns.tolist(),
        "cat_cols_main": cat_cols_m,
        "cat_cols_land": cat_cols_l,
        "num_median_main": med_m.to_dict(),
        "num_median_land": med_l.to_dict(),

        "agg_specs": [cols for cols, _ in AGG_SPECS],  # recipe
        "models": {"main": main_model, "land": land_model},
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
        bundle = train_segment_models(df, reports_dir=args.reports)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        joblib.dump(bundle, args.out)
        print(f"✅ Saved model -> {args.out}")
        print(f"✅ Metrics -> {args.reports}/metrics.json")

    else:
        bundle = joblib.load(args.out)
        print(json.dumps(bundle.get("metrics", {}), indent=2))


if __name__ == "__main__":
    main()
