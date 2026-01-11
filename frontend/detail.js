// detail.js

// 1) Leer parámetros de la URL (?city=Vienna o ?lat=..&lng=..)
function getQueryParams() {
  const p = new URLSearchParams(window.location.search);
  return {
    city: p.get("city"),
    lat: p.get("lat") ? Number(p.get("lat")) : null,
    lng: p.get("lng") ? Number(p.get("lng")) : null,
  };
}

// 2) Llamar a tu API Gateway
const API_BASE = "https://5l3e4zv2p1.execute-api.us-east-1.amazonaws.com";
const CURRENT_ENDPOINT = `${API_BASE}/current-weather`;

async function fetchCurrentWeather({ city, lat, lng }) {
  let url;
  if (city) {
    url = `${CURRENT_ENDPOINT}?city=${encodeURIComponent(city)}`;
  } else if (lat != null && lng != null) {
    url = `${CURRENT_ENDPOINT}?lat=${lat}&lng=${lng}`;
  } else {
    throw new Error("Missing city or coordinates");
  }

  const res = await fetch(url);
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

function chooseIconFromData(data) {
  const temp = data.temp;
  const rainProb = data.precipitation_prob;
  const rain = data.precipitation;

  if ((rainProb != null && rainProb > 60) || (rain != null && rain > 1)) {
    return "rainy";
  }

  if ((rainProb != null && rainProb > 20) || (rain != null && rain > 0)) {
    return "partly_cloudy_day";
  }

  if (temp != null && temp >= 20) {
    return "wb_sunny";
  }

  if (temp != null && temp < 5) {
    return "ac_unit";
  }

  return "wb_cloudy";
}

// 4) Rellenar el HTML con los datos de la API
function updateDetailPage(w) {
  const locationNameEl = document.querySelector(".location-name");
  const locationUpdatedEl = document.querySelector(".location-updated");
  const coordsEl = document.getElementById("locationCoords"); 
  const tempEl = document.querySelector(".current-temp");
  const descEl = document.querySelector(".current-desc");
  const feelsEl = document.querySelector(".current-feels");
  const windEl = document.getElementById("windSpeed");
  const humidityEl = document.getElementById("Humidity");
  const rainProbEl = document.getElementById("rainProb");
  const rainAmountEl = document.getElementById("rainAmount");

  if (locationNameEl) {
    locationNameEl.textContent = w.city_name || "Unknown city";
  }
  if (locationUpdatedEl) {
    const t = formatTimeFromTimestamp(w.timestamp);
    locationUpdatedEl.textContent = t
      ? `Last updated: ${t}`
      : "Last updated: -";
  }

  if (coordsEl && w.lat != null && w.lon != null) {
    coordsEl.textContent = `Lat: ${w.lat.toFixed(
      5
    )}, Lon: ${w.lon.toFixed(5)}`;
  }

  if (tempEl && w.temp != null) {
    tempEl.textContent = `${Math.round(w.temp)}°`;
  }

  const mainIconEl = document.querySelector(".current-icon");
  if (mainIconEl && w.temp != null) {
    const iconName = chooseIconFromData(w);
    mainIconEl.textContent = iconName;
  }

  if (descEl && w.temp != null) {
    if (w.precipitation_prob != null && w.precipitation_prob > 50)
      descEl.textContent = "Rainy";
    else if (w.temp >= 25) descEl.textContent = "Hot";
    else if (w.temp >= 15) descEl.textContent = "Mild";
    else descEl.textContent = "Cold";
  }

  if (feelsEl && w.temp != null) {
    feelsEl.textContent = `Feels like ${Math.round(w.temp)}°`;
  }

  if (windEl && w.wind_speed != null) {
    windEl.textContent = `${w.wind_speed.toFixed(1)} km/h`;
  }

  if (humidityEl && w.humidity != null) {
    humidityEl.textContent = `${w.humidity.toFixed(1)}%`;
  }

  if (rainProbEl && w.precipitation_prob != null) {
    rainProbEl.textContent = `${w.precipitation_prob.toFixed(1)}%`;
  }

  if (rainAmountEl && w.precipitation != null) {
    rainAmountEl.textContent = `${w.precipitation.toFixed(1)} mm`;
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

  if (btnCancel) {
    btnCancel.addEventListener("click", () => {
      panel.style.display = "none";
    });
  }

  btnApply.addEventListener("click", async () => {
    if (!inputDate.value || !inputTime.value) {
      if (window.showToast) showToast("Please select both date and time", "warning");
      else alert("Please select both date and time");
      return;
    }

    const selectedISO = `${inputDate.value}T${inputTime.value}:00Z`;
    console.log("Selected forecast datetime:", selectedISO);

    try {
      const { city } = getQueryParams();

      if (!city) throw new Error("Missing city in URL");
      if (!window.EuroWeatherPrediction?.fetchPrediction) {
        throw new Error("prediction.js not loaded");
      }

      const pred = await window.EuroWeatherPrediction.fetchPrediction({
        city,
        time: selectedISO,
      });

      updateDetailPage(pred);

      const locationUpdatedEl = document.querySelector(".location-updated");
      if (locationUpdatedEl) {
        locationUpdatedEl.textContent = `Forecast for: ${selectedISO}`;
      }

      if (window.showToast) showToast("Forecast loaded", "success");
    } catch (err) {
      console.error(err);
      const msg = err?.message || "Prediction failed";
      if (window.showToast) showToast(msg, "error");
      else alert(msg);
    } finally {
      panel.style.display = "none";
    }
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
  const params = getQueryParams();
  // Debe haber city o lat+lng
  if (!params.city && (params.lat == null || params.lng == null)) {
    console.warn("No city or coordinates in query string");
    return;
  }

  try {
    const data = await fetchCurrentWeather(params);
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
});
