from __future__ import annotations

import os
import re
import json
from datetime import date as _date

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field


MODEL_PATH = os.getenv("MODEL_PATH", "models/trained_model.pkl")

app = FastAPI(title="Real Estate Price Engine", version="1.0.0")
templates = Jinja2Templates(directory="src/templates")


# ----------------------------
# Request schema
# ----------------------------
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
CHOICES: dict = {}


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


def _load_choices():
    """
    Loads dropdown options from reports/choices.json.
    If file missing, return empty dict (form will fallback to text input option below if needed).
    """
    path = "reports/choices.json"
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


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
    df = df.copy()

    df["ACTUAL_AREA"] = pd.to_numeric(df.get("ACTUAL_AREA"), errors="coerce")
    df["PROCEDURE_AREA"] = pd.to_numeric(df.get("PROCEDURE_AREA"), errors="coerce")

    df["ROOMS_COUNT"] = df.get("ROOMS_EN").apply(parse_rooms) if "ROOMS_EN" in df.columns else np.nan
    df["PARKING_COUNT"] = df.get("PARKING").apply(parse_parking) if "PARKING" in df.columns else np.nan

    df["AREA_RATIO"] = df["ACTUAL_AREA"] / (df["PROCEDURE_AREA"] + 1e-6)

    df["INSTANCE_DATE"] = pd.to_datetime(df["INSTANCE_DATE"], errors="coerce")
    df["MONTH"] = df["INSTANCE_DATE"].dt.month.fillna(1).astype(int)
    df["DOW"] = df["INSTANCE_DATE"].dt.dayofweek.fillna(0).astype(int)

    df["MONTH_SIN"] = np.sin(2 * np.pi * df["MONTH"] / 12)
    df["MONTH_COS"] = np.cos(2 * np.pi * df["MONTH"] / 12)
    df["DOW_SIN"] = np.sin(2 * np.pi * df["DOW"] / 7)
    df["DOW_COS"] = np.cos(2 * np.pi * df["DOW"] / 7)

    for c in ["NEAREST_METRO_EN", "NEAREST_MALL_EN", "NEAREST_LANDMARK_EN", "PROJECT_EN"]:
        if c in df.columns:
            df[f"{c}_MISSING"] = df[c].isna().astype(int)

    return df


def _choose_segment(row: dict) -> str:
    land_value = str(BUNDLE.get("land_value", "Land"))
    prop_type = str(row.get("PROP_TYPE_EN", ""))
    return "LAND" if prop_type == land_value else "MAIN"


def _apply_bundle_cleaning(X: pd.DataFrame, segment: str) -> pd.DataFrame:
    X = X.copy()

    if segment == "MAIN":
        cat_cols = BUNDLE.get("cat_cols_main", [])
        med = BUNDLE.get("num_median_main", {})
    else:
        cat_cols = BUNDLE.get("cat_cols_land", [])
        med = BUNDLE.get("num_median_land", {})

    # Fill categorical
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
            med_val = np.nanmedian(X[c].values)
            fill_val = float(med_val) if np.isfinite(med_val) else 0.0
        X[c] = X[c].fillna(fill_val)

    return X


@app.on_event("startup")
def startup():
    global BUNDLE, FEATURE_IMPORTANCE, CHOICES

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file not found: {MODEL_PATH}. "
            f"Train first: python -m src.model train --csv <path> --out {MODEL_PATH}"
        )

    BUNDLE = joblib.load(MODEL_PATH)
    FEATURE_IMPORTANCE = _load_feature_importance()
    CHOICES = _load_choices()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": BUNDLE is not None,
        "model_path": MODEL_PATH,
        "choices_loaded": bool(CHOICES),
    }


# ----------------------------
# HTML UI routes
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": None, "choices": CHOICES},
    )


def _to_bool(v: str | None):
    if v is None or v == "":
        return None
    return v.lower() == "true"


@app.post("/predict-form", response_class=HTMLResponse)
def predict_form(
    request: Request,
    property_type: str = Form(...),
    property_subtype: str = Form(None),
    area: str = Form(...),
    actual_area: float = Form(...),
    rooms: int = Form(None),
    parking: int = Form(None),
    is_offplan: str = Form(None),
    is_freehold: str = Form(None),
    usage: str = Form(None),
    nearest_metro: str = Form(None),
    nearest_mall: str = Form(None),
    nearest_landmark: str = Form(None),
    project: str = Form(None),
    master_project: str = Form(None),
    instance_date: str = Form(None),
):
    req = PredictRequest(
        property_type=property_type,
        property_subtype=property_subtype or None,
        area=area,
        actual_area=float(actual_area),
        rooms=rooms,
        parking=parking,
        is_offplan=_to_bool(is_offplan),
        is_freehold=_to_bool(is_freehold),
        usage=usage or None,
        nearest_metro=nearest_metro or None,
        nearest_mall=nearest_mall or None,
        nearest_landmark=nearest_landmark or None,
        project=project or None,
        master_project=master_project or None,
        instance_date=instance_date or None,
    )

    result = predict(req)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": result, "choices": CHOICES},
    )


# ----------------------------
# JSON API route (original)
# ----------------------------
@app.post("/api/v1/predict-price", response_model=PredictResponse)
def predict(req: PredictRequest):
    if BUNDLE is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    inst_date = req.instance_date or str(_date.today())

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

        "GROUP_EN": None,
        "PROCEDURE_EN": None,
        "PROCEDURE_AREA": None,
        "TOTAL_BUYER": None,
        "TOTAL_SELLER": None,
        "TRANSACTION_NUMBER": None,
        "DATE": None,
    }

    df = pd.DataFrame([row])
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

    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan

    X = df[feature_cols].copy()
    X = _apply_bundle_cleaning(X, segment)

    try:
        pred_log = float(model.predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

    pred_price = float(np.expm1(pred_log))

    sigma = 0.65
    try:
        sigma = float(BUNDLE["metrics"][segment]["val"]["RMSE_log"])
    except Exception:
        pass

    z = 1.96
    lower = float(np.expm1(pred_log - z * sigma))
    upper = float(np.expm1(pred_log + z * sigma))

    conf = "high" if sigma < 0.35 else "medium" if sigma < 0.60 else "low"

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
        segment_used=segment,
    )
