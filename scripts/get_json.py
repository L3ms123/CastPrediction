import requests
import json
from pathlib import Path

def save_json(data, filename):
    try:
        base_dir = Path(__file__).resolve().parent
        data_dir = base_dir.parent / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        file_path = data_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    except (OSError, TypeError) as e:
        raise RuntimeError(f"No se pudo guardar el JSON: {e}") from e
    
VALID_COUNTRIES = {
    "ES","FR","DE","IT","GB","PT","NL","BE","AT","PL","SE","FI","DK",
    "IE","CZ","HU","RO","BG","GR","CH","UA","RU","NO","IS","RS","SK",
    "HR","SI","EE","LV","LT","MD","AL","MK","ME","BA","XK","BY"
}

def is_europe(c):
    return (
        c["country"] in VALID_COUNTRIES
    )
    
def main():
    # URL del JSON original
    url = "https://raw.githubusercontent.com/lutangar/cities.json/master/cities.json"
    
    # Data from the largest cities in the European Union includes Russia (by population within city limits).
    # Data retrieved from United Nations World Urbanization Prospects: The 2018 Revision. 
    # The annual growth rate between 2020 and 2025 was used to estimate current values.
    ciudades_eu = set([
        "Moscow", 
        "Paris",
        "London",
        "Madrid",
        "Barcelona",
        "Saint Petersburg", 
        "Rome", 
        "Berlin",
        "Milan", 
        "Athens", 
        "Kyiv", # (Kiev)
        "Lisbon",
        "Manchester",
        "Birmingham", # (Birmingham)
        "Ufa",
        "Lille",
        "Brussels",
        "Minsk",
        "Oslo",
        "Vienna", 
        "Turin", 
        "Warsaw", 
        "Hamburg",
        "Bucureşci", # (Bucharest)
        "Budapest",
        "Lyon",
        "Glasgow",
        "Stockholm",
        "Novosibirsk",
        "Marseille Prefecture",
        "Munich", # (Munich)
        "Yekaterinburg",
        "Zürich", # (Zurich)
        "Kharkiv",
        "Novi Beograd", # (Belgrade)
        "Copenhagen",
        "Helsinki",
        "Porto",
        "Prague",
        "Kazan",
        "Sofia",
        "Köln",
        "Dublin",
        "Nizhniy Novgorod",
        "Chelyabinsk",
        "Omsk",
        "Amsterdam",
        "Krasnoyarsk",
        "Samara",
        "Rostov-na-Donu"
    ])

    response = requests.get(url)
    cities = response.json()

    # Obtener las ciudades del json que están en la lista de ciudades de EU
    ciudades_json = [
        c for c in cities
        if c["name"] in ciudades_eu
        and not (c["name"] == "Porto" and c["country"] != "PT")
        and not (c["name"] == "Lille" and c["country"] != "FR")
    ]

    ciudades_filtradas = [c for c in ciudades_json if is_europe(c)]

    # eliminar ciudades que no son de la UE (duplicadas en US, CA, AU)
    
    print("Ciudades eliminadas:", (len(ciudades_json)-len(ciudades_filtradas)))


    encontradas = [c["name"] for c in ciudades_filtradas]
    print(len(encontradas), "ciudades encontradas")

    for c in ciudades_eu - set(encontradas):
        print(f"Advertencia: {c} no encontrada")

    save_json(ciudades_filtradas, "ciudades_eu.json")

if __name__ == "__main__":
    main()