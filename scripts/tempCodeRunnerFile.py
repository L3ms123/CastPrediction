        c for c in cities
        if c["name"] in ciudades_eu
        and not (c["name"] == "Porto" and c["country"] != "PT")
        and not (c["name"] == "Lille" and c["country"] != "FR")
    ]