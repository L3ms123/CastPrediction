window.searchMode = "city"; // global simple

let citiesData = [];

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
    option.value = c.name;    // o c.city según tu JSON
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

  // pintar el menú de sugerencias debajo del input
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
      alert("Enter a city");
      return;
    }

    if (window.searchMode === "city") {
      // 1) Modo City Name → ir a detail
      const url = `detail.html?city=${encodeURIComponent(city)}`;
      window.location.href = url;
    } else {
      // 2) Modo Coordinates → centrar mapa en la ciudad y hacer scroll
      const center = getCityCenter(city);

      if (!center) {
        alert("City not found in database");
        return;
      }

      if (window.map) {
        window.map.setView(center, 12);
      }

      if (mapSection) {
        mapSection.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    }
  });
});