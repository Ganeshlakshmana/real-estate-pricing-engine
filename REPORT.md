# REPORT — Real Estate Price Prediction Engine

## 1. Executive Summary

**Goal:** Predict `TRANS_VALUE` (property transaction price) from transaction + location attributes, with an explainable and production-ready pipeline.

**Headline metrics (test):**
- R² (log space): **TBD**
- MAE: **TBD**
- MAPE: **TBD**

**Key findings (data):**
- TBD (price differences by property type, area, time)
- TBD (off-plan vs ready, freehold vs leasehold)
- TBD (effects of proximity features)

**Recommended production approach:**
- Segmented model: MAIN vs LAND (LAND is sparse / different dynamics)
- Time-based validation split to avoid leakage
- Log-space regression for stability on heavy-tailed prices
- SHAP-based explanations for debugging and trust

---

## 2. Technical Decisions

### 2.1 Modeling approach
- Chosen model: **CatBoostRegressor** for strong performance with mixed numeric + categorical features and minimal preprocessing.
- Segmentation:
  - MAIN (non-Land): dense and stable; benefits from location/categorical signals
  - LAND: sparse and behaves differently; separate model reduces bias

### 2.2 Feature engineering
- Time features from `INSTANCE_DATE`: month/day-of-week + cyclic encodings (sin/cos)
- Parsed numeric features:
  - `ROOMS_COUNT` extracted from `ROOMS_EN` (e.g., “Studio”, “2 B/R”)
  - `PARKING_COUNT` from `PARKING`
- Ratio: `AREA_RATIO = ACTUAL_AREA / PROCEDURE_AREA`
- Missingness flags for key location fields

### 2.3 Validation strategy
- Time-based split using the last 7 unique days for test and the previous 7 days for validation.
- Metrics:
  - R² on log1p prices
  - MAE, MAPE on raw prices
  - Optional: price-band breakdown (low/medium/high)

---

## 3. Model Interpretability

- Global feature importance (CatBoost)
- SHAP explanations:
  - Top drivers per prediction (location, size, property type, etc.)
  - Example cases: one correct, one failure case

---

## 4. Production Readiness Assessment

### 4.1 Limitations / edge cases
- New/unseen areas or projects → higher uncertainty (wider CI)
- Extremely high-end outliers
- Data drift (market cycles)

### 4.2 Retraining recommendations
- Suggested cadence: monthly (or weekly if market is volatile)
- Monitor: MAE by property type/area + prediction drift

### 4.3 Handling unseen categories
- CatBoost can handle unseen categories, but calibration is weaker.
- Add aggregate priors (train-only) to stabilize high-cardinality location fields.

---

## 5. Future Improvements

- Additional data: geo coordinates, distance-to-metro, amenities, floor, view, building age, macro indicators.
- More robust uncertainty:
  - Quantile regression / conformal prediction
- Better location embeddings:
  - Target encoding with CV, or learned embeddings
- Model monitoring + drift detection + retraining pipeline
