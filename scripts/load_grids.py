import json
import pygeohash as pgh
import math

# ----------------------------------
# CONFIGURATION
# ----------------------------------

CITY_JSON = "data/ciudades_eu_km2.json"
OUTPUT_JSON = "data/ciudades_eu_km2_with_grids.json"

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

if __name__ == "__main__":
    main()