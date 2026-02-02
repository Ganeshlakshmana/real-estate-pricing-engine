# Real Estate Price Prediction Engine

This repository contains a production-ready real estate valuation model and an optional FastAPI service.

## Repository structure

```
├── notebooks/
│   └── analysis.ipynb
├── src/
│   ├── model.py
│   ├── preprocessing.py
│   └── api.py
├── models/
│   └── trained_model.joblib
├── REPORT.md
├── requirements.txt
└── README.md
```

## 1) Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

## 2) Train the model

Place your CSV in the repo (or provide an absolute path). Then:

```bash
python -m src.model --train --csv path/to/transactions.csv --out models/trained_model.joblib
```

Training details:
- Leak-safe **time-based split** by the last 7 unique days (test) and the 7 days before that (validation).
- Trains in **log-space** (`log1p(TRANS_VALUE)`), then converts predictions back to raw currency.
- Uses **segmented CatBoost** models:
  - `MAIN`: all property types except `Land`
  - `LAND`: only `Land`

## 3) Run the FastAPI service (bonus)

```bash
uvicorn src.api:app --reload
```

Health check:
- `GET /health`

Predict:
- `POST /api/v1/predict-price`

Example JSON body:

```json
{
  "property_type": "Apartment",
  "property_subtype": "Flat",
  "area": "Marina District",
  "actual_area": 1200,
  "rooms": "2",
  "parking": "1",
  "is_offplan": false,
  "is_freehold": true,
  "usage": "Residential",
  "nearest_metro": "Central Station",
  "nearest_mall": "City Mall",
  "nearest_landmark": "Harbor",
  "master_project": "Marina Development",
  "project": "Marina Residence"
}
```

Notes:
- The API returns a simple 95% **confidence interval** computed in log-space using validation residuals.
- `key_factors` uses CatBoost SHAP contributions for the single request row (top 3 features).

## 4) How to use your notebook

original notebook is copied to `notebooks/analysis.ipynb`.
Keep it as the main analysis/EDA deliverable.

