import json
import pygeohash as pgh
import math

# ----------------------------------
# CONFIGURATION
# ----------------------------------

<<<<<<< HEAD
CITY_JSON = "data/ciudades_eu_km2.json"
OUTPUT_JSON = "data/ciudades_eu_km2_with_grids.json"
=======
TTL_DAYS = 7
TTL_SECONDS = TTL_DAYS * 24 * 3600

DYNAMODB_TABLE = "WeatherReadings"
CITY_JSON = "data/ciudades_eu_km2.json"
session = requests.Session()
>>>>>>> 373c7ad2989a55ca884800bb44e2d53a76a4b787

# ----------------------------------
# UTILS
# ----------------------------------

def generate_city_grids(center_lat, center_lon, area_km2, precision=5):

    directions = ["top", "bottom", "left", "right"]
    center_hash = pgh.encode(center_lat, center_lon, precision)

    cell_area_km2 = 25  # Approximate area for precision 5 geohash
    n_cells = math.ceil(area_km2 / cell_area_km2)

    grids = set([center_hash])
    frontier = [center_hash]
    visited = set(frontier)

    while len(grids) < n_cells:
        new_frontier = []

        for gh in frontier:
            neighbors = [pgh.get_adjacent(gh, d) for d in directions]

            for ngh in neighbors:
                if ngh not in visited:
                    visited.add(ngh)
                    grids.add(ngh)
                    new_frontier.append(ngh)
                    if len(grids) >= n_cells:
                        break
            if len(grids) >= n_cells:
                break
        frontier = new_frontier

    return grids

<<<<<<< HEAD
=======
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

>>>>>>> 373c7ad2989a55ca884800bb44e2d53a76a4b787
# ----------------------------------
# MAIN
# ----------------------------------

def main():
    # Load JSON
    with open(CITY_JSON, "r", encoding="utf-8") as f:
        cities = json.load(f)

    for city in cities:
        center_lat = float(city['lat'])
        center_lon = float(city['lng'])
        area_km2 = float(city.get("area_km2", 400))

        city_geohashes = generate_city_grids(center_lat, center_lon, area_km2, precision=5)

        # Convert geohashes to the required dict format
        city['grids'] = [
            {"geohash": gh, "lat": pgh.decode(gh)[0], "lon": pgh.decode(gh)[1]}
            for gh in city_geohashes
        ]

    # Save the updated JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(cities, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(cities)} cities and saved to {OUTPUT_JSON}")

def lambda_handler(event, context):
    main()
    return {"statusCode": 200,
            "message": "Weather job executed successfully"}

if __name__ == "__main__":
    main()