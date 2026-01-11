// prediction.js (FRONTEND) - LOCAL predictor
(function () {
  // IMPORTANT: local FastAPI (uvicorn)
  const LOCAL_BASE = "http://127.0.0.1:5000";
  const PREDICT_ENDPOINT = `${LOCAL_BASE}/predict`;

  function isoToUnixSeconds(iso) {
    if (!iso) return null;
    const ms = Date.parse(iso);
    if (Number.isNaN(ms)) return null;
    return Math.floor(ms / 1000);
  }

  function buildPredictUrl({ city, time }) {
    if (!city) throw new Error("Missing city");
    if (!time) throw new Error("Missing time (ISO)");

    const url = new URL(PREDICT_ENDPOINT);
    url.searchParams.set("city", city);
    url.searchParams.set("time", time);
    return url.toString();
  }

  function extractErrorMessage(data, status) {
    const d = data?.detail;
    if (typeof d === "string" && d.trim()) return d;
    if (d && typeof d === "object") {
      if (d.error) return d.error;
      try { return JSON.stringify(d); } catch { return `Predict error (status ${status})`; }
    }
    if (data?.error) return data.error;
    return `Predict error (status ${status})`;
  }

  function normalizePredictResponse(apiData) {
    const predicted = apiData.predicted || {};
    return {
      city_name: apiData.city || "Unknown city",
      lat: apiData.lat,
      lon: apiData.lon,
      geohash: apiData.geohash,
      timestamp: isoToUnixSeconds(apiData.time),

      temp: predicted.temp,
      humidity: predicted.humidity,
      wind_speed: predicted.wind_speed,
      precipitation: predicted.precipitation,
      precipitation_prob: predicted.precipitation_prob,

      time_iso: apiData.time,
      last_available_utc: apiData.last_available_utc,
      max_allowed_utc: apiData.max_allowed_utc,
    };
  }

  async function fetchPrediction({ city, time }) {
    const url = buildPredictUrl({ city, time });

    let res;
    try {
      res = await fetch(url, { method: "GET" });
    } catch {
      throw new Error("No puc connectar amb el predictor local. Comprova que uvicorn està actiu a http://127.0.0.1:5000 i que /health funciona.");
    }

    let data;
    try {
      data = await res.json();
    } catch {
      const txt = await res.text().catch(() => "");
      throw new Error(`Predictor local ha retornat no-JSON (status ${res.status}) ${txt}`);
    }

    if (!res.ok) {
      throw new Error(extractErrorMessage(data, res.status));
    }

    return normalizePredictResponse(data);
  }

  window.EuroWeatherPrediction = { fetchPrediction };
})();
