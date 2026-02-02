# src/config.py

TARGET_COL = "TRANS_VALUE"
DATE_COL = "INSTANCE_DATE"

# Segmentation
SEGMENT_COL = "PROP_TYPE_EN"
LAND_VALUE = "Land"  # update only if your dataset uses different value

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
SEED = 42

# API input fields (must be present in request)
API_BASE_FEATURES = [
    "AREA_EN",
    "PROP_TYPE_EN",
    "ACTUAL_AREA",
    "ROOMS_EN",
    "IS_OFFPLAN_EN",
    "IS_FREE_HOLD_EN",
    "PROJECT_EN",
    "MASTER_PROJECT_EN",
]

# Engineered priors (added automatically in training + API inference)
PRIOR_FEATURES = [
    "agg_area_mean_log", "agg_area_median_log", "agg_area_count", "agg_area_logcount",
    "agg_project_mean_log", "agg_project_median_log", "agg_project_count", "agg_project_logcount",
    "agg_area_prop_mean_log", "agg_area_prop_median_log", "agg_area_prop_count", "agg_area_prop_logcount",
]

# Training features schema used by models
FEATURE_COLS = API_BASE_FEATURES + PRIOR_FEATURES

# Columns we never use as features
DROP_COLS_ALWAYS = [
    TARGET_COL,
    "TRANSACTION_NUMBER",
    DATE_COL,
    "DATE",
]
