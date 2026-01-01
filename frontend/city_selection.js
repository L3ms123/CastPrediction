window.searchMode = "city"; // global simple
let citiesData = [];
let selectedLatLng = null; 

// cargar JSON una vez al arrancar
fetch('../data/ciudades_eu_km2.json')
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
  ); // [web:12][web:6]

  if (!city) return null;
  return [city.lat, city.lng];
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


// humidity,  wind speed, time,  temp,prec, rain,???

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

      searchBtn.textContent = "Go to map"; // solo texto distinto para claridad
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
    const city = searchInput.value.trim();
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
      selectedLatLng = getCityCenter(city);

      if (window.map) {
        window.map.setView(selectedLatLng, 12);
        
        if (mapSection) {
          mapSection.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      }

      const lat = selectedLatLng.lat;
      const lng = selectedLatLng.lng;

      // aquí ya SIEMPRE tienes coords definitivas
      const url = `detail.html?lat=${lat.toFixed(5)}&lng=${lng.toFixed(5)}`;
      window.location.href = url;
    };
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
    selectedLatLng = e.latlng;
    
    searchBtn.textContent = "Search";   // ahora el botón sirve para ir a detail.html
    searchInput.value = `${lat}, ${lng}`;

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
});



