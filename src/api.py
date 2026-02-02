# src/api.py
from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
from datetime import date
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "models/trained_model.pkl")

app = FastAPI(title="Real Estate Price Engine", version="1.0.0")

# ---- Request schema as per challenge PDF ---- :contentReference[oaicite:2]{index=2}
class PredictRequest(BaseModel):
    property_type: str
    property_subtype: str | None = None
    area: str
    actual_area: float
    rooms: int | None = None
    parking: int | None = None
    is_offplan: bool | None = None
    is_freehold: bool | None = None
    usage: str | None = None
    nearest_metro: str | None = None
    nearest_mall: str | None = None
    nearest_landmark: str | None = None
    master_project: str | None = None  # notebook drops this, but keep for input compatibility
    project: str | None = None

    # not required in PDF, but needed to reproduce notebook date features
    instance_date: str | None = Field(None, description="YYYY-MM-DD; defaults to today")

class PredictResponse(BaseModel):
    predicted_price: float
    confidence_interval: dict
    price_per_sqft: float | None
    model_confidence: str
    key_factors: list[str]
    segment_used: str


BUNDLE = None
FEATURE_IMPORTANCE = None
SIGMA_LOG_DEFAULT = None


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


def _add_time_and_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["INSTANCE_DATE"] = pd.to_datetime(df["INSTANCE_DATE"], errors="coerce")
    df["MONTH"] = df["INSTANCE_DATE"].dt.month.fillna(1).astype(int)
    df["DOW"] = df["INSTANCE_DATE"].dt.dayofweek.fillna(0).astype(int)

    df["MONTH_SIN"] = np.sin(2*np.pi*df["MONTH"]/12)
    df["MONTH_COS"] = np.cos(2*np.pi*df["MONTH"]/12)
    df["DOW_SIN"] = np.sin(2*np.pi*df["DOW"]/7)
    df["DOW_COS"] = np.cos(2*np.pi*df["DOW"]/7)

    for c in ["NEAREST_METRO_EN","NEAREST_MALL_EN","NEAREST_LANDMARK_EN","PROJECT_EN"]:
        if c in df.columns:
            df[f"{c}_MISSING"] = (df[c].isna() | (df[c].astype(str) == "UNKNOWN")).astype(int)

    return df


def _choose_model(row: dict):
    seg_col = BUNDLE.get("segment_col", "PROP_TYPE_EN")
    land_value = str(BUNDLE.get("land_value", "Land"))
    v = str(row.get(seg_col, ""))
    if v == land_value:
        return BUNDLE["model_land"], "LAND"
    return BUNDLE["model_main"], "MAIN"


def _safe_fill(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            if df[c].notna().any():
                df[c] = df[c].fillna(df[c].median())
            else:
                df[c] = df[c].fillna(0.0)
        else:
            df[c] = df[c].fillna("UNKNOWN").astype(str)
    return df


@app.on_event("startup")
def startup():
    global BUNDLE, FEATURE_IMPORTANCE, SIGMA_LOG_DEFAULT
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found: {MODEL_PATH}. Train first.")
    BUNDLE = joblib.load(MODEL_PATH)
    FEATURE_IMPORTANCE = _load_feature_importance()
    try:
        SIGMA_LOG_DEFAULT = float(BUNDLE["metrics"]["MAIN"]["val"]["RMSE_log"])
    except Exception:
        SIGMA_LOG_DEFAULT = 0.65


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": BUNDLE is not None}


@app.post("/api/v1/predict-price", response_model=PredictResponse)
def predict(req: PredictRequest):
    if BUNDLE is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # map request -> dataset column names used in notebook/model
    inst_date = req.instance_date or str(date.today())

    row = {
        "PROP_TYPE_EN": req.property_type,
        "PROP_SB_TYPE_EN": req.property_subtype or "UNKNOWN",
        "AREA_EN": req.area,
        "ACTUAL_AREA": float(req.actual_area),
        "ROOMS_EN": str(req.rooms) if req.rooms is not None else "UNKNOWN",
        "PARKING": float(req.parking) if req.parking is not None else np.nan,
        "IS_OFFPLAN_EN": "Yes" if req.is_offplan else "No" if req.is_offplan is not None else "UNKNOWN",
        "IS_FREE_HOLD_EN": "Yes" if req.is_freehold else "No" if req.is_freehold is not None else "UNKNOWN",
        "USAGE_EN": req.usage or "UNKNOWN",
        "NEAREST_METRO_EN": req.nearest_metro or "UNKNOWN",
        "NEAREST_MALL_EN": req.nearest_mall or "UNKNOWN",
        "NEAREST_LANDMARK_EN": req.nearest_landmark or "UNKNOWN",
        "MASTER_PROJECT_EN": req.master_project or "UNKNOWN",
        "PROJECT_EN": req.project or "UNKNOWN",
        "INSTANCE_DATE": inst_date,
        # not in request but model may have seen these cols in CSV
        "GROUP_EN": "UNKNOWN",
        "PROCEDURE_EN": "UNKNOWN",
        "PROCEDURE_AREA": np.nan,
        "TOTAL_BUYER": np.nan,
        "TOTAL_SELLER": np.nan,
    }

    df = pd.DataFrame([row])

    # recreate notebook feature eng
    df = _add_time_and_missing_flags(df)

    # IMPORTANT: our saved model expects a specific feature schema
    feature_cols = BUNDLE.get("feature_cols")
    if not feature_cols:
        raise HTTPException(status_code=500, detail="Model bundle missing feature_cols. Retrain.")

    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan

    df = df[feature_cols].copy()
    df = _safe_fill(df)

    model, segment = _choose_model(row)

    try:
        pred_log = float(model.predict(df)[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

    pred_price = float(np.expm1(pred_log))

    # confidence interval
    sigma = SIGMA_LOG_DEFAULT
    try:
        sigma = float(BUNDLE["metrics"][segment]["val"]["RMSE_log"])
    except Exception:
        pass

    z = 1.96
    lower = float(np.expm1(pred_log - z * sigma))
    upper = float(np.expm1(pred_log + z * sigma))

    # price per sqft (if actual_area looks like sqft; you can rename to sqm if needed)
    ppsf = pred_price / float(req.actual_area) if req.actual_area else None

    # confidence label
    if sigma < 0.35:
        conf = "high"
    elif sigma < 0.60:
        conf = "medium"
    else:
        conf = "low"

    # key factors
    if FEATURE_IMPORTANCE is not None:
        top = FEATURE_IMPORTANCE["feature"].head(5).astype(str).tolist()
    else:
        top = ["AREA_EN", "PROP_TYPE_EN", "ACTUAL_AREA", "PROJECT_EN", "NEAREST_METRO_EN"]

    return PredictResponse(
        predicted_price=pred_price,
        confidence_interval={"lower": max(0.0, lower), "upper": max(0.0, upper)},
        price_per_sqft=ppsf,
        model_confidence=conf,
        key_factors=top,
        segment_used=segment
    )
