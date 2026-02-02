# Real Estate Price Prediction Engine

This repository contains a **complete, end-to-end real estate price prediction system**
developed as part of a take-home assessment.

The solution covers:
- Data preprocessing and feature engineering
- Time-aware model training and evaluation
- Segment-specific modeling (MAIN vs LAND)
- Model interpretability and performance analysis
- A production-ready **FastAPI inference service**
- An **interactive HTML form** for manual testing
- Cloud-ready deployment (successfully built on Render)

---

## Repository Structure

├── src/
│ ├── model.py # Model training and evaluation
│ ├── preprocessing.py # Data loading, cleaning, feature engineering
│ ├── api.py # FastAPI inference service + HTML form
│ ├── scripts/
│ │ └── build_choices.py # Generates dropdown values for the form
│ └── templates/
│ └── index.html # Interactive prediction form
│
├── models/
│ └── trained_model.pkl # Serialized trained model bundle
│
├── reports/
│ ├── metrics.json
│ └── choices.json # Valid categorical values for dropdown inputs
│
├── REPORT.md # Detailed technical report
├── requirements.txt # Python dependencies
└── README.md # Project overview and usage instructions


Each component is modular, reproducible, and designed for production use.

---

## Component Overview

### `src/model.py`
- Implements the complete training pipeline
- Performs **time-based train/validation/test split**
- Trains **segment-specific CatBoost models**:
  - MAIN (apartments, villas, buildings)
  - LAND
- Evaluates models using MAE, MAPE, and R² (log-space)
- Saves a bundled model artifact containing:
  - Trained models
  - Feature schema
  - Preprocessing metadata
  - Evaluation metrics

---

### `src/preprocessing.py`
- Loads raw transaction data
- Performs data cleaning and validation
- Implements feature engineering:
  - Temporal features (month, day-of-week, cyclical encoding)
  - Area- and size-based features
  - Missing value indicators
  - Leakage-safe aggregate priors
- Ensures consistency between training and inference

---

### `src/api.py`
- Implements a **FastAPI-based inference service**
- Loads trained model artifacts at startup
- Reproduces feature engineering at inference time
- Exposes:
  - `GET /health` — service health check
  - `POST /api/v1/predict-price` — JSON-based prediction endpoint
  - `GET /` — interactive HTML form for testing predictions
- Returns:
  - Predicted price
  - Confidence interval
  - Model confidence level
  - Key contributing features

---

### `src/scripts/build_choices.py`
- Extracts valid categorical values from the training dataset
- Generates `reports/choices.json`
- Used to populate dropdowns in the HTML form
- Prevents invalid input values during inference

---

### `src/templates/index.html`
- Interactive web form embedded in FastAPI
- Dropdown-based inputs for categorical features
- Prevents bad or unseen categorical values
- Designed for demonstration and manual testing

---

### `models/trained_model.pkl`
- Serialized model bundle
- Contains:
  - MAIN and LAND CatBoost models
  - Feature lists
  - Median values for numeric imputation
  - Categorical feature definitions
  - Validation and test metrics

---

### `REPORT.md`
- Full technical documentation
- Covers:
  - Problem formulation
  - Feature engineering strategy
  - Model benchmarking and selection rationale
  - Performance analysis (MAIN vs LAND)
  - Production deployment details
  - Limitations and future improvements

---

## Setup Instructions

### 1. Create and activate a virtual environment (recommended)

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

Train MAIN and LAND models

Save the trained bundle to models/trained_model.pkl

Generate evaluation metrics and feature importance files

Generate Dropdown Values (Required for UI)
python src/scripts/build_choices.py
This creates:

reports/choices.json
Running the API Locally
uvicorn src.api:app --reload
Available Endpoints
Health check:
http://127.0.0.1:8000/health

Swagger UI:
http://127.0.0.1:8000/docs

Interactive form:
http://127.0.0.1:8000/

Example JSON Prediction Request
POST /api/v1/predict-price

{
  "property_type": "Apartment",
  "property_subtype": "Flat",
  "area": "Marina District",
  "actual_area": 1200,
  "rooms": 2,
  "parking": 1,
  "is_offplan": false,
  "is_freehold": true,
  "usage": "Residential",
  "nearest_metro": "Central Station",
  "nearest_mall": "City Mall",
  "nearest_landmark": "Harbor",
  "master_project": "Marina Development",
  "project": "Marina Residence"
}
Deployment
The application has been successfully built on Render

Free-tier runtime limitations prevented continuous execution

No code changes are required for deployment on:

Render (paid tier)

Railway

Fly.io

Azure App Service

Any Docker-based cloud platform

Start command used in production:

gunicorn -k uvicorn.workers.UvicornWorker src.api:app --bind 0.0.0.0:$PORT
Notes
Confidence intervals are computed in log-space using validation residuals

Dropdown-based UI prevents invalid categorical inputs

LAND predictions are inherently noisier due to market sparsity

