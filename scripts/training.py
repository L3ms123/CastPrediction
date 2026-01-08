# scripts/training.py
# ------------------------------------------------------------
# Train Gradient Boosting models using data read from DynamoDB (boto3).
# - Uses first 6 days (144 hours) as input window
# - Predicts 7th day (24 hours) as ground truth
#
# Output:
# - models_gb/gb_<target>.joblib
# - models_gb/metrics.json
# - models_gb/dataset_preview.csv
#
# Credentials are loaded automatically by boto3 from:
#   - environment variables (recommended for your case), or
#   - ~/.aws/credentials, or
#   - an IAM role (if running in AWS)
# ------------------------------------------------------------

import os
import json
import math
import joblib
import numpy as np
import pandas as pd

from decimal import Decimal
from datetime import datetime, timezone, timedelta

import boto3
from boto3.dynamodb.conditions import Key

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================
# -------- CONFIG ---------
# =========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
OUT_DIR = os.path.join(PROJECT_ROOT, "models_gb")

# DynamoDB
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
DYNAMODB_TABLE = os.getenv("DYNAMODB_TABLE", "WeatherReadings")

# DynamoDB schema 
PK_NAME = "PK"
SK_NAME = "SK"
PK_PREFIX = "grid#"
SK_PREFIX = "ts#"

# Use this JSON to know which grids exist (no table scan needed)
CITY_GRIDS_JSON = os.getenv(
    "CITY_GRIDS_JSON",
    os.path.join(PROJECT_ROOT, "data", "ciudades_eu_km2_with_grids.json"),
)


MAX_GRIDS = int(os.getenv("MAX_GRIDS", "0"))  # all

# ML columns
COL_GEOHASH = "geohash"
COL_TIMESTAMP = "timestamp"

TARGETS = ["temp", "humidity", "wind_speed", "precipitation", "precipitation_prob"]

HOURS_6_DAYS = 6 * 24   # 144
HOURS_7_DAYS = 7 * 24   # 168
# =========================


# =========================
# ----- Helper utils ------
# =========================
def decimal_to_float(x):
    if isinstance(x, Decimal):
        return float(x)
    return x


def load_geohashes_from_city_json(path: str) -> list[str]:
    """Collect all unique geohashes from ciudades_eu_km2_with_grids.json."""
    with open(path, "r", encoding="utf-8") as f:
        cities = json.load(f)

    geos = []
    for city in cities:
        for grid in city.get("grids", []):
            gh = grid.get("geohash")
            if gh:
                geos.append(str(gh))

    geos = sorted(set(geos))
    if MAX_GRIDS and MAX_GRIDS > 0:
        geos = geos[:MAX_GRIDS]
    return geos


# =========================
# ----- DynamoDB read -----
# =========================
def query_grid_last_7_days(table, geohash: str) -> list[dict]:
    """
    Query one grid for the last 7 days using PK + SK range.
    Requires table sort key formatted like: "ts#YYYY-MM-DDTHH:00:00Z"
    """
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(days=7)

    sk_start = f"{SK_PREFIX}{start.strftime('%Y-%m-%dT%H:00:00Z')}"
    sk_end = f"{SK_PREFIX}{now.strftime('%Y-%m-%dT%H:00:00Z')}"
    pk = f"{PK_PREFIX}{geohash}"

    items = []
    last_key = None

    while True:
        kwargs = {
            "KeyConditionExpression": Key(PK_NAME).eq(pk) & Key(SK_NAME).between(sk_start, sk_end),
            "ProjectionExpression": "#geohash, #ts, #temp, #humidity, #wind, #prec, #precprob",
            "ExpressionAttributeNames": {
                "#geohash": "geohash",
                "#ts": "timestamp",
                "#temp": "temp",
                "#humidity": "humidity",
                "#wind": "wind_speed",
                "#prec": "precipitation",
                "#precprob": "precipitation_prob",
            },
        }
        if last_key:
            kwargs["ExclusiveStartKey"] = last_key

        resp = table.query(**kwargs)
        items.extend(resp.get("Items", []))

        last_key = resp.get("LastEvaluatedKey")
        if not last_key:
            break

    return items


def load_raw_data_from_dynamodb() -> pd.DataFrame:
    """
    Lab-style boto3:
      dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
      table = dynamodb.Table(DYNAMODB_TABLE)
      table.query(...)
    """
    dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
    table = dynamodb.Table(DYNAMODB_TABLE)

    geohashes = load_geohashes_from_city_json(CITY_GRIDS_JSON)
    print(f"[DDB] Region={AWS_REGION} Table={DYNAMODB_TABLE}")
    print(f"[DDB] Geohashes loaded from JSON: {len(geohashes)}")

    rows = []
    for i, gh in enumerate(geohashes, 1):
        print(f"[DDB] ({i}/{len(geohashes)}) Query {PK_PREFIX}{gh}")
        items = query_grid_last_7_days(table, gh)

        for it in items:
            ts = it.get("timestamp")
            if ts is None:
                continue

            rows.append({
                "geohash": str(it.get("geohash", gh)),
                "timestamp": int(decimal_to_float(ts)),
                "temp": decimal_to_float(it.get("temp")),
                "humidity": decimal_to_float(it.get("humidity")),
                "wind_speed": decimal_to_float(it.get("wind_speed")),
                "precipitation": decimal_to_float(it.get("precipitation")),
                "precipitation_prob": decimal_to_float(it.get("precipitation_prob")),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No rows returned from DynamoDB.\n" )
    return df


# =========================
# ----- Dataset ----
# =========================
def ensure_columns(df: pd.DataFrame):
    missing = [c for c in [COL_GEOHASH, COL_TIMESTAMP] + TARGETS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nColumns found: {list(df.columns)}")


def build_dataset(df_raw: pd.DataFrame):
    df = df_raw.copy()
    ensure_columns(df)

    df[COL_GEOHASH] = df[COL_GEOHASH].astype(str)
    df[COL_TIMESTAMP] = pd.to_numeric(df[COL_TIMESTAMP], errors="coerce")

    for col in TARGETS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[COL_GEOHASH, COL_TIMESTAMP])
    df = df.sort_values([COL_GEOHASH, COL_TIMESTAMP]).reset_index(drop=True)

    X_rows, y_rows, groups = [], [], []

    for geoh, g in df.groupby(COL_GEOHASH):
        g = g.sort_values(COL_TIMESTAMP).copy()

        # relative hour index
        t0 = int(g[COL_TIMESTAMP].min())
        g["hour_idx"] = ((g[COL_TIMESTAMP] - t0) // 3600).astype(int)

        # remove duplicates per hour
        g = g.drop_duplicates(subset=["hour_idx"], keep="last")

        if g["hour_idx"].max() < HOURS_7_DAYS - 1:
            continue

        hist = g[g["hour_idx"].between(0, HOURS_6_DAYS - 1)].copy()
        day7 = g[g["hour_idx"].between(HOURS_6_DAYS, HOURS_7_DAYS - 1)].copy()

        if len(hist) < 100 or len(day7) < 20:
            continue

        hist["hour_of_day"] = hist["hour_idx"] % 24
        day7["hour_of_day"] = day7["hour_idx"] % 24

        overall = {}
        for var in TARGETS:
            v = hist[var].dropna()
            overall[f"{var}_overall_mean"] = float(v.mean()) if len(v) else np.nan
            overall[f"{var}_overall_std"] = float(v.std(ddof=0)) if len(v) else np.nan

        last = hist[hist["hour_idx"] == (HOURS_6_DAYS - 1)].tail(1)
        last_vals = {
            f"{var}_last": float(last[var].iloc[0]) if len(last) and pd.notna(last[var].iloc[0]) else np.nan
            for var in TARGETS
        }

        for hod in range(24):
            hh = hist[hist["hour_of_day"] == hod]
            d7 = day7[day7["hour_of_day"] == hod]

            if len(hh) < 3 or len(d7) != 1:
                continue

            feats = {"geohash": geoh, "hour_of_day": hod}
            feats.update(overall)
            feats.update(last_vals)

            for var in TARGETS:
                vv = hh[var].dropna()
                feats[f"{var}_hod_mean"] = float(vv.mean()) if len(vv) else np.nan
                feats[f"{var}_hod_min"]  = float(vv.min())  if len(vv) else np.nan
                feats[f"{var}_hod_max"]  = float(vv.max())  if len(vv) else np.nan
                feats[f"{var}_hod_std"]  = float(vv.std(ddof=0)) if len(vv) else np.nan

            target_row = {var: float(d7[var].iloc[0]) if pd.notna(d7[var].iloc[0]) else np.nan for var in TARGETS}
            X_rows.append(feats)
            y_rows.append(target_row)
            groups.append(geoh)

    X = pd.DataFrame(X_rows)
    y = pd.DataFrame(y_rows)
    groups = np.array(groups)

    X = X.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)

    y_all_nan = y[TARGETS].isna().all(axis=1)
    if y_all_nan.any():
        X = X.loc[~y_all_nan].reset_index(drop=True)
        y = y.loc[~y_all_nan].reset_index(drop=True)
        groups = groups[~y_all_nan.to_numpy()]

    X_num = X.drop(columns=["geohash"])
    X_num = X_num.fillna(X_num.median(numeric_only=True))
    X = pd.concat([X[["geohash"]].reset_index(drop=True), X_num.reset_index(drop=True)], axis=1)

    return X, y, groups


# =========================
# ----- Training ---
# =========================
def train_gb(X: pd.DataFrame, y: pd.DataFrame, groups: np.ndarray) -> pd.DataFrame:
    """
    Train one GradientBoostingRegressor per target variable.
    Split by geohash so test grids are unseen.
    Saves:
      - models_gb/gb_bundle.joblib  (ONE file containing all 5 models)
      - models_gb/metrics.csv
      - models_gb/dataset_preview.csv (optional)
    Returns a metrics dataframe.
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test  = X.iloc[test_idx].reset_index(drop=True)

    feature_cols = [c for c in X_train.columns if c != "geohash"]
    X_train_num = X_train[feature_cols]
    X_test_num  = X_test[feature_cols]

    bundle = {
        "feature_columns": feature_cols,
        "models": {},               # {target: sklearn_model}
        "meta": {
            "rows_total": int(len(X)),
            "rows_train": int(len(X_train)),
            "rows_test": int(len(X_test)),
            "unique_geohash_total": int(pd.Series(groups).nunique()),
            "unique_geohash_train": int(X_train["geohash"].nunique()),
            "unique_geohash_test": int(X_test["geohash"].nunique()),
        }
    }

    metrics_rows = []

    for var in TARGETS:
        y_train = y[var].iloc[train_idx].reset_index(drop=True)
        y_test  = y[var].iloc[test_idx].reset_index(drop=True)

        train_mask = y_train.notna()
        test_mask  = y_test.notna()

        model = GradientBoostingRegressor(random_state=42)
        model.fit(X_train_num.loc[train_mask], y_train.loc[train_mask])

        pred = model.predict(X_test_num.loc[test_mask])
        true = y_test.loc[test_mask].to_numpy()

        mae = mean_absolute_error(true, pred) if len(true) else math.nan
        mse = mean_squared_error(true, pred) if len(true) else math.nan
        rmse = math.sqrt(mse) if len(true) else math.nan

        metrics_rows.append({
            "target": var,
            "MAE": float(mae),
            "RMSE": float(rmse),
            "n_test": int(len(true)),
        })

        bundle["models"][var] = model

    # Save ONE model file
    joblib.dump(bundle, os.path.join(OUT_DIR, "gb_bundle.joblib"))

    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False)

    # Dataset preview
    #preview = pd.concat([X.head(50), y.head(50)], axis=1)
    #preview.to_csv(os.path.join(OUT_DIR, "dataset_preview.csv"), index=False)

    return metrics_df

def main():
    print("Loading data from DynamoDB...")
    df_raw = load_raw_data_from_dynamodb()

    print("Building dataset (6 days → predict day 7)...")
    X, y, groups = build_dataset(df_raw)

    print("Training Gradient Boosting models...")
    metrics_df = train_gb(X, y, groups)

    print("Saved model and metrics")
    print("\nMetrics:")
    print(metrics_df)


if __name__ == "__main__":
    main()

