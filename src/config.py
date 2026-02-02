# src/config.py

TARGET_COL = "TRANS_VALUE"
DATE_COL = "INSTANCE_DATE"
DAY_COL = "DATE"

SEGMENT_COL = "PROP_TYPE_EN"
LAND_VALUE = "Land"

SEED = 42

# Exact notebook drops
DROP_COLS = [
    "TRANS_VALUE",
    "TRANSACTION_NUMBER",
    "INSTANCE_DATE",
    "DATE",
    "MASTER_PROJECT_EN",
]

# Exact notebook priors
AGG_SPECS = [
    (["AREA_EN"], "AGG_AREA_EN"),
    (["PROJECT_EN"], "AGG_PROJECT_EN"),
    (["AREA_EN", "PROP_TYPE_EN"], "AGG_AREA_EN_PROP_TYPE_EN"),
]

# Exact CatBoost params from notebook
MAIN_PARAMS = dict(
    loss_function="RMSE",
    depth=8,
    learning_rate=0.05,
    iterations=4000,
    random_seed=SEED,
    verbose=200,
)

LAND_PARAMS = dict(
    loss_function="RMSE",
    depth=6,
    learning_rate=0.05,
    iterations=2500,
    random_seed=SEED,
    verbose=200,
)
