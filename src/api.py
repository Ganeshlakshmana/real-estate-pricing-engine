# src/api.py
from __future__ import annotations

import os
import re
from datetime import date as _date
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = os.getenv("MODEL_PATH", "models/trained_model.pkl")

app = FastAPI(title="Real Estate Price Engine", version="1.0.0")

# ---- Request schema (challenge-style) ----
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
    project: str | None = None
    master_project: str | None = None
    instance_date: str | None = Field(None, description="YYYY-MM-DD (defaults to today)")

class PredictResponse(BaseModel):
    predicted_price: float
    confidence_interval: dict
    price_per_sqft: float | None
    model_confidence: str
    key_factors: list[str]
    segment_used: str


BUNDLE = None
FEATURE_IMPORTANCE = None


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


def parse_rooms(x):
    if x is None:
        return np.nan
    s = str(x).strip().lower()
    if "studio" in s:
        return 0.0
    m = re.search(r"(\d+)", s)
    return float(m.group(1)) if m else np.nan

def parse_parking(x):
    if x is None:
        return np.nan
    m = re.search(r"(\d+)", str(x))
    return float(m.group(1)) if m else np.nan


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Same engineered features as notebook:
    - ROOMS_COUNT / PARKING_COUNT
    - AREA_RATIO
    - MONTH/DOW + cyclical encoding
    - missing flags for key columns (isna)
    """
    df = df.copy()

    df["ACTUAL_AREA"] = pd.to_numeric(df.get("ACTUAL_AREA"), errors="coerce")
    df["PROCEDURE_AREA"] = pd.to_numeric(df.get("PROCEDURE_AREA"), errors="coerce")

    df["ROOMS_COUNT"] = df.get("ROOMS_EN").apply(parse_rooms) if "ROOMS_EN" in df.columns else np.nan
    df["PARKING_COUNT"] = df.get("PARKING").apply(parse_parking) if "PARKING" in df.columns else np.nan

    df["AREA_RATIO"] = df["ACTUAL_AREA"] / (df["PROCEDURE_AREA"] + 1e-6)

    df["INSTANCE_DATE"] = pd.to_datetime(df["INSTANCE_DATE"], errors="coerce")
    df["MONTH"] = df["INSTANCE_DATE"].dt.month.fillna(1).astype(int)
    df["DOW"] = df["INSTANCE_DATE"].dt.dayofweek.fillna(0).astype(int)

    df["MONTH_SIN"] = np.sin(2*np.pi*df["MONTH"]/12)
    df["MONTH_COS"] = np.cos(2*np.pi*df["MONTH"]/12)
    df["DOW_SIN"] = np.sin(2*np.pi*df["DOW"]/7)
    df["DOW_COS"] = np.cos(2*np.pi*df["DOW"]/7)

    for c in ["NEAREST_METRO_EN", "NEAREST_MALL_EN", "NEAREST_LANDMARK_EN", "PROJECT_EN"]:
        if c in df.columns:
            df[f"{c}_MISSING"] = df[c].isna().astype(int)

    return df


def _choose_segment(row: dict) -> str:
    land_value = str(BUNDLE.get("land_value", "Land"))
    prop_type = str(row.get("PROP_TYPE_EN", ""))
    return "LAND" if prop_type == land_value else "MAIN"


def _apply_bundle_cleaning(X: pd.DataFrame, segment: str) -> pd.DataFrame:
    """
    Reproduce training-time cleaning:
    - cat cols filled with '__MISSING__'
    - numeric columns coerced + filled with train medians (saved in bundle)
    """
    X = X.copy()

    if segment == "MAIN":
        cat_cols = BUNDLE.get("cat_cols_main", [])
        med = BUNDLE.get("num_median_main", {})
    else:
        cat_cols = BUNDLE.get("cat_cols_land", [])
        med = BUNDLE.get("num_median_land", {})

    # Fill cats
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].fillna("__MISSING__").astype(str)
        else:
            X[c] = "__MISSING__"

    # Numeric columns are the rest
    num_cols = [c for c in X.columns if c not in cat_cols]
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        fill_val = med.get(c, np.nan)
        if pd.isna(fill_val):
            # fallback if median missing
            fill_val = float(np.nanmedian(X[c].values)) if np.isfinite(np.nanmedian(X[c].values)) else 0.0
        X[c] = X[c].fillna(fill_val)

    return X


@app.on_event("startup")
def startup():
    global BUNDLE, FEATURE_IMPORTANCE
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file not found: {MODEL_PATH}. "
            f"Train first: python -m src.model train --csv <path> --out {MODEL_PATH}"
        )
    BUNDLE = joblib.load(MODEL_PATH)
    FEATURE_IMPORTANCE = _load_feature_importance()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": BUNDLE is not None, "model_path": MODEL_PATH}


@app.post("/api/v1/predict-price", response_model=PredictResponse)
def predict(req: PredictRequest):
    if BUNDLE is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    inst_date = req.instance_date or str(_date.today())

    # Map API request -> training column names
    row = {
        "PROP_TYPE_EN": req.property_type,
        "PROP_SB_TYPE_EN": req.property_subtype or None,
        "AREA_EN": req.area,
        "ACTUAL_AREA": float(req.actual_area),
        "ROOMS_EN": str(req.rooms) if req.rooms is not None else None,
        "PARKING": str(req.parking) if req.parking is not None else None,
        "IS_OFFPLAN_EN": "Yes" if req.is_offplan is True else "No" if req.is_offplan is False else None,
        "IS_FREE_HOLD_EN": "Yes" if req.is_freehold is True else "No" if req.is_freehold is False else None,
        "USAGE_EN": req.usage or None,
        "NEAREST_METRO_EN": req.nearest_metro or None,
        "NEAREST_MALL_EN": req.nearest_mall or None,
        "NEAREST_LANDMARK_EN": req.nearest_landmark or None,
        "PROJECT_EN": req.project or None,
        "MASTER_PROJECT_EN": req.master_project or None,
        "INSTANCE_DATE": inst_date,

        # Present in dataset but not always in API
        "GROUP_EN": None,
        "PROCEDURE_EN": None,
        "PROCEDURE_AREA": None,
        "TOTAL_BUYER": None,
        "TOTAL_SELLER": None,
        "TRANSACTION_NUMBER": None,
        "DATE": None,
    }

    df = pd.DataFrame([row])

    # Add engineered features
    df = add_features(df)

    segment = _choose_segment(row)

    if segment == "MAIN":
        feature_cols = BUNDLE.get("feature_cols_main")
        model = BUNDLE["models"]["main"]
    else:
        feature_cols = BUNDLE.get("feature_cols_land")
        model = BUNDLE["models"]["land"]

    if not feature_cols:
        raise HTTPException(status_code=500, detail="Model bundle missing feature columns. Retrain.")

    # Ensure schema exists
    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan

    X = df[feature_cols].copy()
    X = _apply_bundle_cleaning(X, segment)

    # Predict (log space)
    try:
        pred_log = float(model.predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

    pred_price = float(np.expm1(pred_log))

    # Simple CI from validation RMSE_log in bundle (if present)
    sigma = 0.65
    try:
        sigma = float(BUNDLE["metrics"][segment]["val"]["RMSE_log"])
    except Exception:
        pass
    z = 1.96
    lower = float(np.expm1(pred_log - z * sigma))
    upper = float(np.expm1(pred_log + z * sigma))

    # Confidence label
    conf = "high" if sigma < 0.35 else "medium" if sigma < 0.60 else "low"

    # key factors (from feature_importance.csv if present)
    if FEATURE_IMPORTANCE is not None:
        key_factors = FEATURE_IMPORTANCE["feature"].head(5).astype(str).tolist()
    else:
        key_factors = ["AREA_EN", "PROP_TYPE_EN", "ACTUAL_AREA", "PROJECT_EN", "NEAREST_METRO_EN"]

    ppsf = pred_price / float(req.actual_area) if req.actual_area else None

    return PredictResponse(
        predicted_price=pred_price,
        confidence_interval={"lower": max(0.0, lower), "upper": max(0.0, upper)},
        price_per_sqft=ppsf,
        model_confidence=conf,
        key_factors=key_factors,
        segment_used=segment
    )
