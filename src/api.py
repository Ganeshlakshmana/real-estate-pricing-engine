# src/api.py
from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "models/trained_model.pkl")

app = FastAPI(title="Real Estate Price Engine", version="1.0.0")

# ------------ Request Schema ------------
class PredictRequest(BaseModel):
    AREA_EN: str = Field(..., description="Area name")
    PROP_TYPE_EN: str = Field(..., description="Property type (e.g., Apartment/Villa/Land)")
    ACTUAL_AREA: float = Field(..., gt=0, description="Actual property area")
    ROOMS_EN: str | None = Field(None, description="Rooms label")
    IS_OFFPLAN_EN: str | None = Field(None, description="Yes/No")
    IS_FREE_HOLD_EN: str | None = Field(None, description="Yes/No")
    PROJECT_EN: str | None = Field(None, description="Project name")
    MASTER_PROJECT_EN: str | None = Field(None, description="Master project name")

class PredictResponse(BaseModel):
    price: float
    lower_bound: float
    upper_bound: float
    currency: str = "AED"
    key_factors: list[str]
    segment_used: str


BUNDLE = None
FEATURE_IMPORTANCE = None
SIGMA_LOG_DEFAULT = None


def _safe_fill_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            if df[c].notna().any():
                df[c] = df[c].fillna(df[c].median())
            else:
                df[c] = df[c].fillna(0.0)
        else:
            df[c] = df[c].fillna("UNKNOWN").astype(str)

    if "ACTUAL_AREA" in df.columns:
        df.loc[df["ACTUAL_AREA"] <= 0, "ACTUAL_AREA"] = np.nan
        df["ACTUAL_AREA"] = df["ACTUAL_AREA"].fillna(df["ACTUAL_AREA"].median())

    return df


def _load_feature_importance():
    path = "reports/feature_importance.csv"
    if os.path.exists(path):
        try:
            fi = pd.read_csv(path)
            if {"feature", "importance"}.issubset(fi.columns):
                return fi.sort_values("importance", ascending=False)
        except Exception:
            return None
    return None


def _choose_model(row: dict):
    seg_col = BUNDLE.get("segment_col", "PROP_TYPE_EN")
    land_value = str(BUNDLE.get("land_value", "Land"))
    v = str(row.get(seg_col, ""))
    if v == land_value:
        return BUNDLE["model_land"], "LAND"
    return BUNDLE["model_main"], "MAIN"


def _add_priors_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rebuild the same MAIN priors used in training from lookup maps saved in bundle.
    """
    df = df.copy()
    agg_maps = BUNDLE.get("agg_maps", {})

    fallback = {"mean_log": 0.0, "median_log": 0.0, "count": 0.0, "logcount": 0.0}

    for prefix, spec in agg_maps.items():
        cols = spec["cols"]
        m = spec["map"]

        def lookup(row):
            key = tuple(row.get(c, "UNKNOWN") for c in cols)
            return m.get(key, fallback)

        vals = df.apply(lookup, axis=1)
        df[f"{prefix}_mean_log"]   = [v["mean_log"] for v in vals]
        df[f"{prefix}_median_log"] = [v["median_log"] for v in vals]
        df[f"{prefix}_count"]      = [v["count"] for v in vals]
        df[f"{prefix}_logcount"]   = [v["logcount"] for v in vals]

    return df


@app.on_event("startup")
def startup():
    global BUNDLE, FEATURE_IMPORTANCE, SIGMA_LOG_DEFAULT

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file not found: {MODEL_PATH}. "
            f"Train first: python -m src.model train --csv <path> --out {MODEL_PATH}"
        )

    BUNDLE = joblib.load(MODEL_PATH)
    FEATURE_IMPORTANCE = _load_feature_importance()

    try:
        SIGMA_LOG_DEFAULT = float(BUNDLE["metrics"]["MAIN"]["val"]["RMSE_log"])
    except Exception:
        SIGMA_LOG_DEFAULT = 0.65


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": BUNDLE is not None, "model_path": MODEL_PATH}


@app.post("/api/v1/predict-price", response_model=PredictResponse)
def predict_price(req: PredictRequest):
    if BUNDLE is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    row = req.model_dump()
    df = pd.DataFrame([row])

    # add priors exactly like training
    df = _add_priors_inference(df)

    # align to model schema
    feature_cols = BUNDLE.get("feature_cols")
    if not feature_cols:
        raise HTTPException(status_code=500, detail="Model bundle missing feature_cols. Retrain the model.")

    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0 if c.endswith(("mean_log", "median_log", "count", "logcount")) else "UNKNOWN"

    df = df[feature_cols].copy()
    df = _safe_fill_input(df)

    model, segment = _choose_model(row)

    try:
        pred_log = float(model.predict(df)[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

    price = float(np.expm1(pred_log))

    # 95% CI using segment-specific RMSE_log as sigma
    sigma = SIGMA_LOG_DEFAULT
    try:
        sigma = float(BUNDLE["metrics"][segment]["val"]["RMSE_log"])
    except Exception:
        pass

    z = 1.96
    lower = float(np.expm1(pred_log - z * sigma))
    upper = float(np.expm1(pred_log + z * sigma))

    if FEATURE_IMPORTANCE is not None:
        key_factors = FEATURE_IMPORTANCE["feature"].head(5).astype(str).tolist()
    else:
        key_factors = feature_cols[:5]

    return PredictResponse(
        price=price,
        lower_bound=max(0.0, lower),
        upper_bound=max(0.0, upper),
        key_factors=key_factors,
        segment_used=segment
    )
