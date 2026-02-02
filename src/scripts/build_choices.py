import json
import os
import pandas as pd

CSV_PATH = "dataset/raw/transactions-2025-03-21.csv" 
OUT_PATH = "reports/choices.json"

# API field name -> dataset column name
COLS = {
    "property_type": "PROP_TYPE_EN",
    "property_subtype": "PROP_SB_TYPE_EN",
    "area": "AREA_EN",
    "usage": "USAGE_EN",
    "nearest_metro": "NEAREST_METRO_EN",
    "nearest_mall": "NEAREST_MALL_EN",
    "nearest_landmark": "NEAREST_LANDMARK_EN",
    "project": "PROJECT_EN",
    "master_project": "MASTER_PROJECT_EN",
}

def uniq(series, limit=3000):
    # Clean and take unique values
    vals = (
        series.dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    vals = sorted(vals)
    return vals[:limit]

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    choices = {}
    for api_field, col in COLS.items():
        if col in df.columns:
            choices[api_field] = uniq(df[col])
        else:
            choices[api_field] = []

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(choices, f, indent=2, ensure_ascii=False)

    print(f"Saved dropdown choices -> {OUT_PATH}")

if __name__ == "__main__":
    main()
