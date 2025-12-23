import json
import boto3
from decimal import Decimal
import math
import requests
from datetime import datetime, timezone
import time
import random
# ----------------------------------
# CONFIGURATION
# ----------------------------------

TTL_DAYS = 7
TTL_SECONDS = TTL_DAYS * 24 * 3600

DYNAMODB_TABLE = "WeatherReadings"
CITY_JSON = "data/ciudades_eu_km2_with_grids.json"
session = requests.Session()

# ----------------------------------
# UTILS
# ----------------------------------

# Obtain index of the time closest to the current time
def get_current_hour_index(times):
    now = datetime.now(timezone.utc)
    best_idx = 0
    min_diff = float("inf")

    for i, t in enumerate(times):
        t_dt = datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
        diff = abs((t_dt - now).total_seconds())
        if diff < min_diff:
            min_diff = diff
            best_idx = i

    return best_idx

# Obtain time data from Open-Meteo for a given grid
def fetch_hourly(lat, lon, retries=5):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "relativehumidity_2m", "windspeed_10m", "precipitation", "precipitation_probability"],
        "timezone": "UTC"
    }

    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=10)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            wait = 2 ** attempt + random.random()
            print(f"Open-Meteo error, retrying in {wait:.1f}s...")
            time.sleep(wait)

    raise RuntimeError("Open-Meteo failed after retries")


# Store data on DynamoDB
def store_grid_data(weather_table, city, lat, lon, geohash):
    data = fetch_hourly(lat, lon)

    times = data["hourly"]["time"]
    temps = data["hourly"]["temperature_2m"]
    hums  = data["hourly"]["relativehumidity_2m"]
    winds = data["hourly"]["windspeed_10m"]
    rains = data["hourly"]["precipitation"]
    rain_prob = data["hourly"]["precipitation_probability"]

    idx = get_current_hour_index(times)

    ts_iso = times[idx] + "Z"
    ts_unix = int(
        datetime.fromisoformat(times[idx])
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )

    # ensure no null values
    rain_value = rains[idx] if rains[idx] is not None else 0
    rain_prob_value = rain_prob[idx] if rain_prob[idx] is not None else 0

    item = {
        "PK": f"grid#{geohash}",
        "SK": f"ts#{ts_iso}",
        "lat": Decimal(str(lat)),
        "lon": Decimal(str(lon)),
        "geohash": geohash,
        "city_name": city,
        "temp": Decimal(str(temps[idx])),
        "humidity": Decimal(str(hums[idx])),
        "wind_speed": Decimal(str(winds[idx])),
        "precipitation": Decimal(str(rain_value)),
        "precipitation_prob": Decimal(str(rain_prob_value)),  
        "timestamp": ts_unix,
        "ttl": ts_unix + TTL_SECONDS
    }

    weather_table.put_item(Item=item)

    print(f"Inserted current hour for {city} - {geohash}")

# ----------------------------------
# MAIN
# ----------------------------------

def main():
    # Connect to DynamoDB
    dynamodb = boto3.resource("dynamodb")
    weather_table = dynamodb.Table(DYNAMODB_TABLE)

    # Load JSON
    with open(CITY_JSON, "r", encoding="utf-8") as f:
        cities = json.load(f)

    for city in cities:
        # obtain city information
        name = city['name']
        country = city['country']

        # obtain each grid information and load it
        for grid in city['grids']:
            geohash = grid["geohash"]
            lat = float(grid["lat"])
            lon = float(grid["lon"])

            store_grid_data(weather_table, name, lat, lon, geohash)
            time.sleep(1)
            
    print("Loaded all cities successfully")

def lambda_handler(event, context):
    main()
    return {"statusCode": 200,
            "message": "Weather job executed successfully"}

if __name__ == "__main__":
    main()