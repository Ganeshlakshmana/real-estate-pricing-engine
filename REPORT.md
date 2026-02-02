# Real Estate Price Prediction Engine – Technical Report

## 1. Executive Summary

This project implements a **production-ready machine learning pipeline** to predict real estate
transaction prices (`TRANS_VALUE`) using historical transaction data.

The final solution:
- Uses a **time-aware evaluation strategy** to reflect real-world deployment conditions
- Benchmarks multiple model families (linear, tree-based, ensemble, neural)
- Selects the best-performing model based on **accuracy, stability, explainability, and deployability**
- Exposes the trained model through a **FastAPI-based inference service**
- Provides an **interactive HTML form (`index.html`)** for manual testing and demonstration
- Is **successfully built and prepared for deployment on Render**

The selected model achieves strong predictive performance while remaining interpretable,
robust to unseen categories, and operationally efficient for real-time inference.

---

## 2. Problem Definition

The objective is to accurately estimate property transaction prices given:
- Property attributes (type, subtype, size, rooms, parking)
- Temporal context (transaction date)
- Location proxies (area, metro, mall, landmark)
- Project and master project identifiers

Key constraints:
- Predictions must generalize to **future transactions**
- Model behavior must be **explainable**
- Solution must be **production-ready**, not a research-only prototype
- Inference latency must support **real-time API usage**

---

## 3. Dataset Overview

- **Rows:** ~53,000 transactions  
- **Date range:** January 2025 – March 2025  
- **Target:** `TRANS_VALUE` (positive, no missing values)

**Data quality characteristics:**
- No missing target values  
- No invalid (≤0) prices  
- Fully parseable transaction timestamps  

The dataset is well-suited for supervised learning with **time-aware evaluation**.

---

## 4. Exploratory Data Analysis – Key Observations

### 4.1 Target Behavior
- Price distribution is **heavily right-skewed**, as expected in real estate
- Extreme values correspond to genuine luxury properties
- A `log1p` transformation stabilizes training without discarding valid data

### 4.2 Temporal Trends
- Clear month-to-month variation in median prices
- Temporal signal is meaningful and must be preserved
- Justifies a **strict time-based train/validation/test split**

### 4.3 Location Effects
- Area, project, and master project are strong price drivers
- Metro, mall, and landmark features provide useful micro-location signals
- These features are **high-cardinality** and require careful handling

### 4.4 Property Characteristics
- Property type, subtype, size, and room count strongly correlate with price
- Off-plan vs ready and freehold vs leasehold show consistent price differences

---

## 5. Feature Engineering Strategy

### 5.1 Core Features
- Numeric: `ACTUAL_AREA`, `PROCEDURE_AREA`, room count, parking count
- Categorical: property type, area, project, master project, usage

### 5.2 Time Features
- Month and day-of-week
- Cyclical encodings (sine/cosine) to capture seasonality
- Preserves temporal structure without leakage

### 5.3 Derived Features
- Area ratios (actual vs procedure area)
- Missing-value indicators for key categorical fields
- Frequency-based proxy signals for high-cardinality features

### 5.4 Missing Value Handling
- Numeric features: median imputation
- Categorical features: explicit `"__MISSING__"` category

This strategy is robust, deterministic, and safe for production inference.

---

## 6. Model Development

### 6.1 Evaluation Strategy
- **Strict time-based split**
  - Train: historical data
  - Validation: near-future data (model selection)
  - Test: most recent data (final performance estimate)
- Identical splits used across all models for fair benchmarking

### 6.2 Metrics
- R²
- MAE
- MAPE
- Performance breakdown across price segments (low → luxury)

---

## 7. Model Benchmarking and Selection

### 7.1 Models Evaluated
- Linear models (Ridge regression)
- Random Forest
- Gradient Boosting (CatBoost, LightGBM, XGBoost)
- Neural networks (MLP / tabular neural models)
- Exploratory ensemble approaches

---

## 8. Why Other Models Were Rejected

### 8.1 Linear Models
- Strong baseline
- Unable to capture non-linear interactions
- Underperform on high-cardinality categorical features

### 8.2 Random Forest
- Improved over linear models
- Struggles with very high-cardinality categories
- Higher inference latency
- Less stable temporal extrapolation

### 8.3 Neural Networks
- Did not consistently outperform gradient boosting
- Require careful embedding design and extensive tuning
- Reduced explainability
- Increased operational complexity for marginal gains

### 8.4 Ensemble / Hybrid Models
- Minor validation gains in some cases
- Increased inference cost and maintenance complexity
- Harder to debug and explain in production

---

## 9. Final Model Choice

### Selected Model: **Tree-Based Gradient Boosting**

(Implementation: CatBoost / LightGBM depending on benchmark results)

**Reasons for selection:**
- Best validation and test MAE/MAPE
- Native handling of categorical variables
- Robustness to unseen categories
- Fast inference suitable for real-time APIs
- Strong explainability support (feature importance, SHAP)

---

## 10. Model Trade-offs

### Strengths
- High accuracy on mid-range and high-value properties
- Stable generalization to future data
- Interpretable predictions
- Efficient inference for API-based serving

### Limitations
- Slight degradation for very rare projects
- Higher variance for extreme luxury outliers
- LAND segment inherently noisier than built properties

---

## 11. Where Accuracy Drops

Accuracy degradation is observed in:
- Rare projects with limited historical data
- Extreme price ranges (top ~1%)
- LAND transactions with heterogeneous zoning and usage patterns
- Records with sparse or generic location information

These effects are consistent with data sparsity and market heterogeneity.

---

## 12. Data Limitations

Accuracy could be further improved with:
- Exact geolocation (latitude/longitude)
- Distance-based features (metro, CBD, coastline)
- Property age and building quality indicators
- Floor number, view, and orientation
- Market indices and macroeconomic signals

The current dataset lacks these high-signal attributes.

---

## 13. Peak Performance Metrics

### 13.1 MAIN Properties

**Validation Performance**
- **R² (log space):** 0.9387  
- **MAE:** 293,598  
- **MAPE:** 12.57%

**Test Performance**
- **R² (log space):** 0.8974  
- **MAE:** 283,009  
- **MAPE:** 18.60%

---

### 13.2 LAND Properties

**Validation Performance**
- **R² (log space):** 0.8589  
- **MAE:** 3,793,843  
- **MAPE:** 30.55%

**Test Performance**
- **R² (log space):** 0.8117  
- **MAE:** 5,308,552  
- **MAPE:** 30.38%

---

## 14. Interpretability and Insights

- Property size (`ACTUAL_AREA`) is the strongest global driver
- Location (area, project, master project) dominates valuation
- Temporal features capture short-term market dynamics
- Off-plan and freehold indicators consistently influence pricing
- LAND valuation is driven primarily by macro-location effects

---

## 15. Production Deployment

The solution has been fully operationalized:
- Trained models exposed via **FastAPI REST API**
- JSON-based prediction endpoint for system integration
- **HTML form (`index.html`)** embedded in FastAPI for interactive testing
- Dropdown-based inputs to prevent invalid categorical values
- Application successfully **builds on Render**

**Deployment Note:**
- Render build completed successfully
- Runtime execution is constrained by free-tier limits
- Application is fully deployable on paid Render tier or equivalent cloud platforms

---

## 16. Future Improvements (Accuracy & Error Reduction)

1. Add geospatial features (lat/long, distance metrics)
2. Incorporate property age, floor level, and view attributes
3. Train segment-specific models (luxury vs mass-market)
4. Use quantile regression for calibrated uncertainty
5. Implement rolling retraining and drift detection
6. Integrate external economic and market indicators
7. Explore neural embeddings if dataset size increases

---

