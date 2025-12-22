import requests
import json
from pathlib import Path

def save_json(data, filename):
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    # URL del JSON original
    url = "https://raw.githubusercontent.com/lutangar/cities.json/master/cities.json"
    
    # Data from the largest cities in the European Union (by population within city limits).
    # Data retrieved from United Nations World Urbanization Prospects: The 2018 Revision. 
    # The annual growth rate between 2020 and 2025 was used to estimate current values.
    ciudades_eu = set([
        "Moskva", # (Moscow)
        "Paris",
        "London",
        "Madrid",
        "Barcelona",
        "Saint Petersburg", 
        "Roma", # (Rome)
        "Berlin",
        "Milan", 
        "Athens", # (Athens)
        "Kyiv", # (Kiev)
        "Lisbon",
        "Manchester",
        "Birmingham", # (Birmingham)
        "Baku",
        "Napoles", 
        "Brussels",
        "Minsk",
        "Almaty",
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
        "Astana",
        "Dublin",
        "Nizhniy Novgorod",
        "Chelyabinsk",
        "Omsk",
        "Amsterdam",
        "Krasnoyarsk",
        "Samara",
        "Shymkent"
    ])

    response = requests.get(url)
    cities = response.json()

    # Obtener las ciudades del json que están en la lista de ciudades de EU
    ciudades_json = [c for c in cities if c["name"] in ciudades_eu]

    # eliminar ciudades que no son de la UE (duplicadas en US, CA, AU)
    ciudades_filtradas = [
        c for c in ciudades_json  
        if not (c["country"] in  ["US", "CA", "AU"])]
    print("Ciudades eliminadas:", (len(ciudades_json)-len(ciudades_filtradas)))


    encontradas = {c["name"] for c in ciudades_filtradas}
    print(len(encontradas), "ciudades encontradas")

    for c in ciudades_eu - encontradas:
        print(f"Advertencia: {c} no encontrada")

    save_json(ciudades_filtradas, "ciudades_eu.json")

if __name__ == "__main__":
    main()