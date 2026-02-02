# Real Estate Price Prediction 

This repository contains a complete solution for the real estate price prediction
take-home assessment.  
The project is structured exactly as required in the challenge document and includes
model training, preprocessing, documentation, and an optional FastAPI service.

---

## Repository Structure 

The project follows the structure specified in the challenge:

├── src/
│ ├── model.py # Model training and evaluation code
│ ├── preprocessing.py # Data loading, cleaning, and feature engineering
│ └── api.py # FastAPI inference service (bonus)
│
├── models/
│ └── trained_model.pkl # Saved trained model artifact
│
├── REPORT.md # Detailed technical documentation and analysis
├── requirements.txt # Python dependencies
└── README.md # Setup and usage instructions


Each component is designed to be modular, reusable, and production-oriented.

---

## Description of Components

### `src/model.py`
- Implements the full model training pipeline
- Performs time-aware train/validation/test split
- Handles segment-specific modeling (MAIN vs LAND)
- Trains CatBoost regression models
- Evaluates performance and saves metrics
- Persists trained models to `models/trained_model.pkl`

---

### `src/preprocessing.py`
- Loads raw transaction data
- Cleans and validates inputs
- Performs feature engineering:
  - Temporal features
  - Area and size-based features
  - Missing value indicators
  - Leakage-safe aggregate price priors
- Ensures consistency between training and inference

---

### `src/api.py` (Bonus)
- Implements a FastAPI service for inference
- Loads trained model artifacts
- Recreates feature engineering at inference time
- Exposes:
  - `GET /health`
  - `POST /api/v1/predict-price`
- Returns prediction, confidence interval, and key factors

---

### `models/trained_model.pkl`
- Serialized trained model(s)
- Contains:
  - MAIN and LAND segment models
  - Feature schema
  - Medians and preprocessing metadata
  - Evaluation metrics

---

### `REPORT.md`
- Full technical documentation of the solution
- Covers:
  - Model selection rationale
  - Why alternative models (including neural networks) were rejected
  - Trade-offs of the chosen model
  - Performance analysis and limitations
  - Data gaps and future improvements
  - Production considerations

---

### `requirements.txt`
Lists all Python dependencies required to reproduce the results, including:
- pandas
- numpy
- scikit-learn
- catboost
- fastapi
- uvicorn
- shap (for interpretability)

---

## Setup Instructions

### 1. Create and activate virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
2. Install dependencies
pip install -r requirements.txt
Model Training
To train the model and generate artifacts:

python -m src.model train \
  --csv dataset/raw/transactions-2025-03-21.csv \
  --out models/trained_model.pkl \
  --reports reports
This will:

Train the model

Save the trained model to models/trained_model.pkl

Generate evaluation metrics and feature importance

Running the API (Bonus)
uvicorn src.api:app --reload
Health check: http://127.0.0.1:8000/health

Swagger UI: http://127.0.0.1:8000/docs

<<<<<<< HEAD
=======
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

>>>>>>> 0295e2ff4af62cdf46a057c0c1f86379b66885b8
