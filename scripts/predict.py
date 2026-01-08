import os
import json
import math
import joblib
import boto3
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from boto3.dynamodb.conditions import Key

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
DYNAMODB_TABLE = os.getenv("DYNAMODB_TABLE", "WeatherReadings")

PK_NAME = "PK"
SK_NAME = "SK"
PK_PREFIX = "grid#"
SK_PREFIX = "ts#"

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "gb_bundle.joblib")
CITY_JSON_PATH = os.path.join(BASE_DIR, "ciudades_eu_km2_with_grids.json")

TARGETS = ["temp", "humidity", "wind_speed", "precipitation", "precipitation_prob"]

MAX_FUTURE_DAYS = 4  


# ---------------- Helpers ----------------
def dec(x):
    return float(x) if isinstance(x, Decimal) else x

def resp(code, body):
    return {
        "statusCode": code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body),
    }

def parse_time(iso):
    if not iso:
        raise ValueError("Missing time parameter")
    s = iso.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)

def norm(s):
    return (s or "").strip().lower()

def load_city_data():
    with open(CITY_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def resolve_city_to_grid(cities, city, country=None):
    city_n = norm(city)
    country_n = norm(country) if country else None

    matches = []
    for c in cities:
        if norm(c.get("name")) == city_n:
            if country_n is None or norm(c.get("country")) == country_n:
                matches.append(c)

    if not matches:
        raise ValueError("City not found in JSON (check spelling)")

    c = matches[0]
    grids = c.get("grids", [])
    if not grids:
        raise ValueError("City has no grids in JSON")

    g0 = grids[0]  # representative grid
    return float(g0["lat"]), float(g0["lon"]), str(g0["geohash"]), c.get("country")


# ---------------- DynamoDB ----------------
def query_range(table, geohash, start_dt, end_dt):
    sk_start = f"{SK_PREFIX}{start_dt.strftime('%Y-%m-%dT%H:00:00Z')}"
    sk_end = f"{SK_PREFIX}{end_dt.strftime('%Y-%m-%dT%H:00:00Z')}"
    pk = f"{PK_PREFIX}{geohash}"

    items = []
    last_key = None
    while True:
        kwargs = {
            "KeyConditionExpression": Key(PK_NAME).eq(pk) & Key(SK_NAME).between(sk_start, sk_end),
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
        if not last_key:
            break
    return items

def get_latest_timestamp(table, geohash):
    """
    Efficient: query last item by SK descending and Limit=1
    """
    pk = f"{PK_PREFIX}{geohash}"
    r = table.query(
        KeyConditionExpression=Key(PK_NAME).eq(pk),
        ProjectionExpression="#ts",
        ExpressionAttributeNames={"#ts": "timestamp"},
        ScanIndexForward=False,  # descending by SK
        Limit=1,
    )
    items = r.get("Items", [])
    if not items:
        raise ValueError("No data for this city/grid in DynamoDB")
    return int(dec(items[0]["timestamp"]))


# ---------------- Features ----------------
def build_features(items, hour):
    rows = []
    for it in items:
        ts = it.get("timestamp")
        if ts is None:
            continue
        ts_i = int(dec(ts))
        dt = datetime.fromtimestamp(ts_i, tz=timezone.utc)
        rows.append({
            "timestamp": ts_i,
            "hour": dt.hour,
            "temp": dec(it.get("temp")),
            "humidity": dec(it.get("humidity")),
            "wind_speed": dec(it.get("wind_speed")),
            "precipitation": dec(it.get("precipitation")),
            "precipitation_prob": dec(it.get("precipitation_prob")),
        })

    if len(rows) < 24:
        raise ValueError("Not enough history to build features")

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
    hod_rows = [r for r in rows if r["hour"] == hour and r[var] is not None]
    for var in TARGETS:
        vv = [r[var] for r in rows if r["hour"] == hour and r[var] is not None]
        if len(vv) < 3:
            # fallback: use overall stats
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


# ---------------- Lambda handler ----------------
def lambda_handler(event, context):
    try:
        qs = event.get("queryStringParameters") or {}
        city = qs.get("city")
        country = qs.get("country")  # optional
        time_iso = qs.get("time")

        if not city or not time_iso:
            return resp(400, {"error": "Required: city and time (ISO) params"})

        target_dt = parse_time(time_iso)
        hour = target_dt.hour

        cities = load_city_data()
        lat, lon, geohash, resolved_country = resolve_city_to_grid(cities, city, country)

        dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
        table = dynamodb.Table(DYNAMODB_TABLE)

        # 1) find latest available timestamp for this grid
        last_ts = get_latest_timestamp(table, geohash)
        last_dt = datetime.fromtimestamp(last_ts, tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
        max_dt = last_dt + timedelta(days=MAX_FUTURE_DAYS)

        # 2) enforce future limit
        if target_dt > max_dt:
            return resp(400, {
                "error": "Requested time too far in the future",
                "last_available_utc": last_dt.strftime("%Y-%m-%dT%H:00:00Z"),
                "max_allowed_utc": max_dt.strftime("%Y-%m-%dT%H:00:00Z"),
                "requested_utc": target_dt.strftime("%Y-%m-%dT%H:00:00Z"),
            })

        # 3) build history window from last_dt (use last 6 days ending at last_dt)
        start_dt = last_dt - timedelta(days=6)
        items = query_range(table, geohash, start_dt, last_dt)

        feats = build_features(items, hour)

        bundle = joblib.load(MODEL_PATH)
        feature_cols = bundle["feature_columns"]
        X = [float(feats.get(c, 0.0)) for c in feature_cols]

        preds = {var: float(bundle["models"][var].predict([X])[0]) for var in TARGETS}

        return resp(200, {
            "city": city,
            "country": country or resolved_country,
            "time": target_dt.strftime("%Y-%m-%dT%H:00:00Z"),
            "last_available_utc": last_dt.strftime("%Y-%m-%dT%H:00:00Z"),
            "max_allowed_utc": max_dt.strftime("%Y-%m-%dT%H:00:00Z"),
            "lat": lat,
            "lon": lon,
            "geohash": geohash,
            "predicted": preds,
        })

    except Exception as e:
        return resp(500, {"error": str(e)})
