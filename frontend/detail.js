// detail.js

// 1) Leer parámetros de la URL (?city=Vienna)
function getQueryParams() {
  const p = new URLSearchParams(window.location.search);
  return {
    city: p.get("city"),
  };
}

// 2) Llamar a tu API Gateway 
const API_BASE = "https://8fcuk1jmo6.execute-api.us-east-1.amazonaws.com";
const CURRENT_ENDPOINT = `${API_BASE}/current-weather`;

async function fetchCurrentWeather(city) {
  const res = await fetch(
    `${CURRENT_ENDPOINT}?city=${encodeURIComponent(city)}`
  );
  if (!res.ok) {
    throw new Error(`API error: ${res.status}`);
  }
  return await res.json();
}

// 3) Formatear hora a algo legible
function formatTimeFromTimestamp(ts) {
  if (!ts) return "";
  const date = new Date(ts * 1000);
  return date.toLocaleTimeString("en-GB", {
    hour: "2-digit",
    minute: "2-digit",
    timeZoneName: "short",
  });
}

let detailMap = null;
let detailMarker = null;

// Crear o actualizar el mapa centrado en lat/lon
function updateMap(lat, lon) {
  const mapDiv = document.getElementById("mapContainer");
  if (!mapDiv || lat == null || lon == null) return;

  // fallback temporal si no hay coords de la API QUITAR ESTO LUEGO
  if (lat == null || lon == null) {
    lat = 48.20849;   // Vienna
    lon = 16.37208;
  }

  // primera vez: crear el mapa
  if (!detailMap) {
    detailMap = L.map("mapContainer", {
      center: [lat, lon],
      zoom: 11,
      zoomControl: true,
    });

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "© OpenStreetMap",
    }).addTo(detailMap);
  } else {
    // siguientes veces: solo recentrar
    detailMap.setView([lat, lon], 11);
  }

  // marcador
  if (detailMarker) {
    detailMarker.setLatLng([lat, lon]);
  } else {
    detailMarker = L.marker([lat, lon]).addTo(detailMap);
  }
}

// 4) Rellenar el HTML con los datos de la API
function updateDetailPage(w) {
  // Cabecera
  const locationNameEl = document.querySelector(".location-name");
  const locationUpdatedEl = document.querySelector(".location-updated");

  if (locationNameEl) {
    locationNameEl.textContent = w.city_name || "Unknown city";
  }
  if (locationUpdatedEl) {
    const t = formatTimeFromTimestamp(w.timestamp);
    locationUpdatedEl.textContent = t
      ? `Last updated: ${t}`
      : "Last updated: -";
  }

  // Bloque principal
  const tempEl = document.querySelector(".current-temp");
  const descEl = document.querySelector(".current-desc");
  const feelsEl = document.querySelector(".current-feels");

  if (tempEl) tempEl.textContent = `${Math.round(w.temp)}°`;

  if (descEl) {
    // descripción muy simple, puedes mejorarla
    if (w.precipitation_prob > 50) descEl.textContent = "Rainy";
    else if (w.temp >= 25) descEl.textContent = "Hot";
    else if (w.temp >= 15) descEl.textContent = "Mild";
    else descEl.textContent = "Cold";
  }

  if (feelsEl) {
    feelsEl.textContent = `Feels like ${Math.round(w.temp)}°`;
  }

  // Resumen (usa el orden que ya tienes en el HTML)
  const summaryItems = document.querySelectorAll(
    ".current-summary .summary-item .summary-value"
  );
  if (summaryItems[0]) summaryItems[0].textContent = `${w.precipitation_prob}%`; // Precipitation

  // Stats cards
  const statCards = document.querySelectorAll(".stats-grid .stat-card");

  // 1) Wind Speed
  if (statCards[0]) {
    const val = statCards[0].querySelector(".stat-value");
    if (val)
      val.innerHTML = `${w.wind_speed.toFixed(
        1
      )} <span class="stat-unit">km/h</span>`;
  }

  // 2) Humidity
  if (statCards[1]) {
    const val = statCards[1].querySelector(".stat-value");
    if (val)
      val.innerHTML = `${Math.round(
        w.humidity
      )} <span class="stat-unit">%</span>`;
  }

  updateMap(w.lat, w.lon);
}

function initForecastPanel() {
  const btnForecast = document.getElementById("btnForecast");
  const panel = document.getElementById("forecastPanel");
  const btnCancel = document.getElementById("forecastCancel");
  const btnApply = document.getElementById("forecastApply");
  const inputDate = document.getElementById("forecastDate");
  const inputTime = document.getElementById("forecastTime");

  if (!btnForecast || !panel) return;

  // abrir / cerrar panel
  btnForecast.addEventListener("click", () => {
    panel.style.display = panel.style.display === "none" ? "block" : "none";

    // pre-rellenar con hoy y hora actual redondeada
    const now = new Date();
    inputDate.value = now.toISOString().slice(0, 10);
    inputTime.value = `${now.getHours().toString().padStart(2, "0")}:00`;
  });

  btnCancel.addEventListener("click", () => {
    panel.style.display = "none";
  });

  btnApply.addEventListener("click", () => {
    if (!inputDate.value || !inputTime.value) {
      if (window.showToast) {
        showToast("Please select both date and time", "warning");
      } else {
        alert("Please select both date and time");
      }
      return;
    }

    const selectedISO = `${inputDate.value}T${inputTime.value}:00Z`;
    console.log("Selected forecast datetime:", selectedISO);

    // Aquí luego llamarás a tu API de predicción, por ejemplo:
    // fetch(`${API_BASE}/forecast?city=${city}&time=${encodeURIComponent(selectedISO)}`)

    panel.style.display = "none";
  });

  // cerrar al hacer clic fuera
  document.addEventListener("click", (e) => {
    if (!panel.contains(e.target) && e.target !== btnForecast) {
      panel.style.display = "none";
    }
  });
}



// 5) Orquestar todo al cargar la página
document.addEventListener("DOMContentLoaded", async () => {
  const { city } = getQueryParams();
  if (!city) {
    console.warn("No city in query string, e.g. detail.html?city=Vienna");
    return;
  }

  try {
    const data = await fetchCurrentWeather(city);
    updateDetailPage(data);   // pinta datos (incluye updateMap)
    initForecastPanel();      // inicializa el panel del 7‑day forecast
  } catch (err) {
    console.error(err);
    if (window.showToast) {
      showToast("Error loading weather data", "error");
    } else {
      alert("Error loading weather data");
    }
  }
  initForecastPanel();
});

