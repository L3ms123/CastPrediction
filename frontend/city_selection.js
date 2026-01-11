const API_BASE = "https://w0zpkv3mui.execute-api.us-east-1.amazonaws.com";

window.searchMode = "city"; // global simple
let citiesData = [];
let selectedLatLng = null; 

// cargar JSON al arrancar
fetch('../data/ciudades_eu_km2_with_grids.json')
  .then(res => res.json())
  .then(data => {
    citiesData = data;
    fillCityDatalist();
  })
  .catch(err => console.error('Error cargando cities.json', err));


// función que devuelve lat/lng de una ciudad
function getCityCenter(cityName) {
  const city = citiesData.find(
    (c) => c.name.toLowerCase() === cityName.toLowerCase()
  );

  if (!city) return null;
  return [Number(city.lat), Number(city.lng)];
}

function fillCityDatalist() {
  const dataList = document.getElementById('cityList');
  if (!dataList) return;

  dataList.innerHTML = ''; // limpiar por si acaso

  citiesData.forEach((c) => {
    const option = document.createElement('option');
    option.value = c.name;    
    dataList.appendChild(option);
  });
}

async function loadPopularCity(card, cityName) {
  const tempEl = card.querySelector(".city-temp");
  const iconEl = card.querySelector(".city-icon");

  tempEl.textContent = "--°"; // estado inicial

  try {
    const url = `${API_BASE}/weatherPrediction?city=${encodeURIComponent(
      cityName
    )}`;
    const res = await fetch(url);
    if (!res.ok) {
      tempEl.textContent = "--°";
      iconEl.textContent = "help"; // icono de error / desconocido
      return;
    }

    const data = await res.json();

    const temp = data.temp != null ? Math.round(data.temp) : null;
    tempEl.textContent = temp != null ? `${temp}°` : "--°";

    // Decidir icono según datos de lluvia/nubes/lo que tengas
    const iconName = chooseIconFromData(data);
    iconEl.textContent = iconName;
  } catch (err) {
    console.error("Error loading popular city", cityName, err);
    tempEl.textContent = "--°";
    iconEl.textContent = "help";
  }
}

function initPopularCities() {
  const cards = document.querySelectorAll(".city-card[data-city]");
  cards.forEach((card) => {
    const cityName = card.getAttribute("data-city");
    if (!cityName) return;

    // Al cargar la página pedimos la última medición
    loadPopularCity(card, cityName);
  });
}

function chooseIconFromData(data) {
  const temp = data.temp;
  const rainProb = data.precipitation_prob;
  const rain = data.precipitation;

  // Mucha lluvia o prob de lluvia → rainy
  if ((rainProb != null && rainProb > 60) || (rain != null && rain > 1)) {
    return "rainy"; // icono Material Symbols
  }

  // Algo de lluvia/nubes
  if ((rainProb != null && rainProb > 20) || (rain != null && rain > 0)) {
    return "partly_cloudy_day";
  }

  // Muy despejado y cálido
  if (temp != null && temp >= 20) {
    return "wb_sunny";
  }

  // Frío pero sin lluvia
  if (temp != null && temp < 5) {
    return "ac_unit"; // nieve / frío
  }

  // Default: nublado
  return "wb_cloudy";
}


document.addEventListener("DOMContentLoaded", () => {
  const coordPreview = document.getElementById("coordPreview");
  const btnCityMode = document.getElementById("btnCityMode");
  const btnCoordMode = document.getElementById("btnCoordMode");
  const searchInput = document.getElementById("searchInput");
  const searchBtn = document.getElementById("searchBtn");
  const coordTooltip = document.getElementById("coordTooltip");
  const mapSection = document.getElementById("mapSection");
  const suggestionsEl = document.getElementById("suggestions");
  const coordText = document.getElementById("coordText"); 

  function setMode(newMode) {
    window.searchMode = newMode;

    if (newMode === "city") {
      btnCityMode.classList.add("active");
      btnCoordMode.classList.remove("active");

      searchBtn.textContent = "Search";
      coordTooltip.style.display = "none";
    } else {
      btnCityMode.classList.remove("active");
      btnCoordMode.classList.add("active");

      searchBtn.textContent = "Go to map"; 
      coordTooltip.style.display = "block";
    }
  }

  btnCityMode.addEventListener("click", () => setMode("city"));
  btnCoordMode.addEventListener("click", () => setMode("coords"));

  // menú de sugerencias debajo del input
  function renderSuggestions(query) {
    suggestionsEl.innerHTML = "";

    const trimmed = query.trim().toLowerCase();
    if (!trimmed) {
      suggestionsEl.style.display = "none";
      return;
    }

    const matches = citiesData.filter((c) =>
      c.name.toLowerCase().includes(trimmed)
    );

    if (matches.length === 0) {
      suggestionsEl.style.display = "none";
      return;
    }

    matches.slice(0, 20).forEach((c) => {
      const item = document.createElement("div");
      item.className = "suggestion-item";

      const citySpan = document.createElement("span");
      citySpan.className = "suggestion-city";
      citySpan.textContent = c.name;

      const coordsSpan = document.createElement("span");
      coordsSpan.className = "suggestion-coords";
      coordsSpan.textContent = `${Number(c.lat).toFixed(3)}, ${Number(c.lng).toFixed(3)}`;

      item.appendChild(citySpan);
      item.appendChild(coordsSpan);

      item.addEventListener("click", () => {
        searchInput.value = c.name;
        suggestionsEl.style.display = "none";
      });

      suggestionsEl.appendChild(item);
    });

    suggestionsEl.style.display = "block";
  }

  // al escribir: actualizar preview + menú
  searchInput.addEventListener("input", () => {
    const cityName = searchInput.value.trim();
    const center = getCityCenter(cityName);

    if (!center) {
      coordPreview.textContent = "";
    } else {
    const [lat, lng] = center;
    coordPreview.textContent = `${lat.toFixed(3)}, ${lng.toFixed(3)}`;

      if (window.searchMode === "coords") {
        selectedLatLng = { lat, lng };  // coords come from city name
        searchBtn.textContent = "Search";
      }
    }

    renderSuggestions(cityName);
  });

  // cerrar menú al hacer clic fuera
  document.addEventListener("click", (e) => {
    const searchBar = document.querySelector(".search-bar");
    if (!searchBar.contains(e.target)) {
      suggestionsEl.style.display = "none";
    }
  });

  // CLICK EN SEARCH: comportamiento distinto según el modo
  searchBtn.addEventListener("click", () => {
    const raw = searchInput.value.trim();
    if (!raw) {
      alert("Please enter a city name", "warning");
      return;
    }

    // ¿Coord mode + input parece coords?
    const isCoordMode = window.searchMode === "coords";
    const coordMatch = raw.match(/^\s*(-?\d+(\.\d+)?)\s*,\s*(-?\d+(\.\d+)?)\s*$/);

    if (isCoordMode && coordMatch) {
      // Caso 2: ya tenemos coords → ir a detail con lat/lng
      const lat = Number(coordMatch[1]);
      const lng = Number(coordMatch[3]);
      const url = `detail.html?lat=${lat.toFixed(5)}&lng=${lng.toFixed(5)}`;
      window.location.href = url;
      return;
    }

    const city = raw;
    if (!city) {
      alert("Please enter a city name", "warning");
      return;
    }

    if (!getCityCenter(city)) {
      console.log('City not found:', city);
      alert(`City "${city}" not found in database`, "error");
      return;
    }

    // MODO 1: CITY NAME → IR A DETAIL.HTML
    if (window.searchMode === "city") {
      console.log('Ciudad válida, navegando a detail.html');
      const url = `detail.html?city=${encodeURIComponent(city)}`;
      window.location.href = url;
    } 

    // MODO 2: COORDINATES → CENTRAR MAPA
    else {
      console.log('Ciudad válida, centrando mapa');
      const center = getCityCenter(city);
      if (!center) return;
      const [lat, lng] = center;

      selectedLatLng = { lat, lng };

      if (window.map) {
        window.map.setView([lat, lng], 12);

        if (mapSection) {
          mapSection.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      }
    }
  });

  // INICIALIZAR MAPA LEAFLET
  const mapDiv = document.getElementById("mapContainer");
  if (!mapDiv) return;

  const defaultCenter = [52.52, 13.405];
  let selectionMarker = null;

  // 1) Guardar el mapa en window.map (GLOBAL)
  window.map = L.map("mapContainer", {
    center: defaultCenter,
    zoom: 11,
    zoomControl: false
  });

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "© OpenStreetMap"
  }).addTo(window.map);

  // Click en el mapa: elegir coordenadas
  window.map.on("click", (e) => {
    const { lat, lng } = e.latlng;

    // Solo si estamos en modo coordenadas
    if (window.searchMode === "coords") {
      selectedLatLng = { lat, lng };
      searchBtn.textContent = "Search";
      searchInput.value = `${lat}, ${lng}`;
    }

    if (selectionMarker) {
      selectionMarker.setLatLng(e.latlng);
    } else {
      selectionMarker = L.marker(e.latlng).addTo(window.map);
    }

    if (coordTooltip) {
      coordTooltip.style.opacity = "1";
    }

    if (coordText) {
      coordText.textContent =
        `Lat: ${lat.toFixed(5)}, Lon: ${lng.toFixed(5)}`;
    }
  });

  const zoomInBtn = document.getElementById("zoomInBtn");
  const zoomOutBtn = document.getElementById("zoomOutBtn");

  if (zoomInBtn) {
    zoomInBtn.addEventListener("click", () => window.map.zoomIn());
  }

  if (zoomOutBtn) {
    zoomOutBtn.addEventListener("click", () => window.map.zoomOut());
  }

  const popularCards = document.querySelectorAll(".city-card[data-city]");
  popularCards.forEach((card) => {
    card.style.cursor = "pointer";
    card.addEventListener("click", () => {
      const cityName = card.getAttribute("data-city");
      if (!cityName) return;

      const url = `detail.html?city=${encodeURIComponent(cityName)}`;
      window.location.href = url;
    });
  });
  initPopularCities();

});



