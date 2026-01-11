import os
import json
import math
import joblib
import traceback
import boto3
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from boto3.dynamodb.conditions import Key
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ---------- Config ----------
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
DYNAMODB_TABLE = os.getenv("DYNAMODB_TABLE", "WeatherReadings")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CITY_JSON_PATH = os.path.join(PROJECT_ROOT, "data", "ciudades_eu_km2_with_grids.json")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models_gb", "gb_bundle.joblib")

TARGETS = ["temp", "humidity", "wind_speed", "precipitation", "precipitation_prob"]
MAX_FUTURE_DAYS = 4
HISTORY_DAYS = 6
MIN_HISTORY_ROWS = 24  # minimum for features
HISTORY_LIMIT = HISTORY_DAYS * 24 + 80  # buffer

# ---------- FastAPI ----------
app = FastAPI()

# CORS (debug: allow all)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Helpers ----------
def dec(x):
    return float(x) if isinstance(x, Decimal) else x

def norm(s: str) -> str:
    return (s or "").strip().lower()

def parse_time(iso: str) -> datetime:
    if not iso:
        raise ValueError("Missing time parameter")
    s = iso.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)

def load_city_data():
    if not os.path.exists(CITY_JSON_PATH):
        raise FileNotFoundError(f"City JSON not found: {CITY_JSON_PATH}")
    with open(CITY_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def resolve_city_to_grid(cities, city: str):
    city_n = norm(city)
    matches = [c for c in cities if norm(c.get("name")) == city_n]
    if not matches:
        raise ValueError("City not found in JSON (check spelling)")
    c = matches[0]
    grids = c.get("grids", [])
    if not grids:
        raise ValueError("City has no grids in JSON")
    g0 = grids[0]  # representative grid
    return float(g0["lat"]), float(g0["lon"]), str(g0["geohash"])

# ---------- DynamoDB history ----------
def fetch_history_from_dynamo(geohash: str, max_items: int = HISTORY_LIMIT):
    """
    Pull newest items for PK=grid#<geohash>, then sort ascending.
    This avoids any assumption about SK format (ISO vs epoch),
    and does not require Lambda changes.
    """
    dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
    table = dynamodb.Table(DYNAMODB_TABLE)
    pk = f"grid#{geohash}"

    items = []
    last_key = None

    while True:
        kwargs = {
            "KeyConditionExpression": Key("PK").eq(pk) & Key("SK").begins_with("ts#"),
            "ScanIndexForward": False,  # newest first
            "Limit": 200,
            "ProjectionExpression": "#ts,#t,#h,#w,#p,#pp",
            "ExpressionAttributeNames": {
                "#ts": "timestamp",
                "#t": "temp",
                "#h": "humidity",
                "#w": "wind_speed",
                "#p": "precipitation",
                "#pp": "precipitation_prob",
            },
        }
        if last_key:
            kwargs["ExclusiveStartKey"] = last_key

        r = table.query(**kwargs)
        items.extend(r.get("Items", []))
        last_key = r.get("LastEvaluatedKey")

        if len(items) >= max_items or not last_key:
            break

    # normalize and keep only rows with timestamp
    rows = []
    for it in items:
        ts = it.get("timestamp")
        if ts is None:
            continue
        rows.append({
            "timestamp": int(dec(ts)),
            "temp": dec(it.get("temp")),
            "humidity": dec(it.get("humidity")),
            "wind_speed": dec(it.get("wind_speed")),
            "precipitation": dec(it.get("precipitation")),
            "precipitation_prob": dec(it.get("precipitation_prob")),
        })

    # oldest -> newest
    rows.sort(key=lambda x: x["timestamp"])

    last_ts = rows[-1]["timestamp"] if rows else None
    return {"history": rows, "last_timestamp": last_ts}

# ---------- Features ----------
def build_features(history_rows, hour: int):
    # Build rows with hour-of-day derived from timestamp
    rows = []
    for it in history_rows:
        ts_i = it.get("timestamp")
        if ts_i is None:
            continue
        dt = datetime.fromtimestamp(int(ts_i), tz=timezone.utc)
        rows.append({
            "timestamp": int(ts_i),
            "hour": dt.hour,
            "temp": it.get("temp"),
            "humidity": it.get("humidity"),
            "wind_speed": it.get("wind_speed"),
            "precipitation": it.get("precipitation"),
            "precipitation_prob": it.get("precipitation_prob"),
        })

    if len(rows) < MIN_HISTORY_ROWS:
        raise ValueError(f"Not enough history rows to build features (got {len(rows)})")

    rows.sort(key=lambda r: r["timestamp"])
    feats = {"hour_of_day": int(hour)}

    # overall stats
    for var in TARGETS:
        vals = [r[var] for r in rows if r[var] is not None]
        if not vals:
            feats[f"{var}_overall_mean"] = 0.0
            feats[f"{var}_overall_std"] = 0.0
            feats[f"{var}_last"] = 0.0
        else:
            mu = sum(vals) / len(vals)
            var0 = sum((x - mu) ** 2 for x in vals) / len(vals)
            feats[f"{var}_overall_mean"] = float(mu)
            feats[f"{var}_overall_std"] = float(math.sqrt(var0))
            feats[f"{var}_last"] = float(rows[-1][var]) if rows[-1][var] is not None else float(mu)

    # hour-of-day stats
    for var in TARGETS:
        vv = [r[var] for r in rows if r["hour"] == hour and r[var] is not None]
        if len(vv) < 3:
            mu = feats[f"{var}_overall_mean"]
            feats[f"{var}_hod_mean"] = mu
            feats[f"{var}_hod_min"] = mu
            feats[f"{var}_hod_max"] = mu
            feats[f"{var}_hod_std"] = feats[f"{var}_overall_std"]
        else:
            mn = min(vv)
            mx = max(vv)
            mu = sum(vv) / len(vv)
            var0 = sum((x - mu) ** 2 for x in vv) / len(vv)
            feats[f"{var}_hod_mean"] = float(mu)
            feats[f"{var}_hod_min"] = float(mn)
            feats[f"{var}_hod_max"] = float(mx)
            feats[f"{var}_hod_std"] = float(math.sqrt(var0))

    return feats

# ---------- Routes ----------
@app.get("/health")
def health():
    return {
        "ok": True,
        "model_exists": os.path.exists(MODEL_PATH),
        "city_json_exists": os.path.exists(CITY_JSON_PATH),
        "model_path": MODEL_PATH,
        "city_json_path": CITY_JSON_PATH,
        "aws_region": AWS_REGION,
        "dynamodb_table": DYNAMODB_TABLE,
    }

@app.get("/predict")
def predict(city: str, time: str):
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

        target_dt = parse_time(time)
        hour = target_dt.hour

        cities = load_city_data()
        lat, lon, geohash = resolve_city_to_grid(cities, city)

        # Get history directly from DynamoDB (no Lambda)
        hist = fetch_history_from_dynamo(geohash)
        history_rows = hist["history"]
        last_ts = hist["last_timestamp"]

        if not last_ts:
            raise ValueError("No DynamoDB data found for this grid (history empty)")

        last_dt = datetime.fromtimestamp(int(last_ts), tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
        max_dt = last_dt + timedelta(days=MAX_FUTURE_DAYS)

        if target_dt > max_dt:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Requested time too far in the future",
                    "last_available_utc": last_dt.strftime("%Y-%m-%dT%H:00:00Z"),
                    "max_allowed_utc": max_dt.strftime("%Y-%m-%dT%H:00:00Z"),
                    "requested_utc": target_dt.strftime("%Y-%m-%dT%H:00:00Z"),
                },
            )

        feats = build_features(history_rows, hour)

        bundle = joblib.load(MODEL_PATH)
        feature_cols = bundle["feature_columns"]
        X = [float(feats.get(c, 0.0)) for c in feature_cols]

        preds = {var: float(bundle["models"][var].predict([X])[0]) for var in TARGETS}

        return {
            "city": city,
            "time": target_dt.strftime("%Y-%m-%dT%H:00:00Z"),
            "last_available_utc": last_dt.strftime("%Y-%m-%dT%H:00:00Z"),
            "max_allowed_utc": max_dt.strftime("%Y-%m-%dT%H:00:00Z"),
            "lat": lat,
            "lon": lon,
            "geohash": geohash,
            "predicted": preds,
        }

    except HTTPException:
        raise
    except Exception as e:
        print("PREDICT ERROR:", repr(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": str(e)})
