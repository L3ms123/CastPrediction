// prediction.js
// Calls predict.py (/predict) using city + time
// Returns data normalized for updateDetailPage(w)

(function () {
  const API_BASE = "https://5l3e4zv2p1.execute-api.us-east-1.amazonaws.com";
  const PREDICT_ENDPOINT = `${API_BASE}/predict`;

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

  function normalizePredictResponse(apiData) {
    // predict.py returns:
    // {
    //   city, time, lat, lon, geohash,
    //   predicted: { temp, humidity, wind_speed, precipitation, precipitation_prob }
    // }

    const predicted = apiData.predicted || {};

    return {
      city_name: apiData.city || "Unknown city",
      lat: apiData.lat,
      lon: apiData.lon,
      geohash: apiData.geohash,

      // detail.js expects unix seconds
      timestamp: isoToUnixSeconds(apiData.time),

      // weather values used by updateDetailPage()
      temp: predicted.temp,
      humidity: predicted.humidity,
      wind_speed: predicted.wind_speed,
      precipitation: predicted.precipitation,
      precipitation_prob: predicted.precipitation_prob,

      // optional (debug / future use)
      time_iso: apiData.time,
      last_available_utc: apiData.last_available_utc,
      max_allowed_utc: apiData.max_allowed_utc,
    };
  }

  async function fetchPrediction({ city, time }) {
    const url = buildPredictUrl({ city, time });

    const res = await fetch(url, {
      method: "GET",
      headers: { Accept: "application/json" },
    });

    let data;
    try {
      data = await res.json();
    } catch {
      throw new Error(`Predict API returned non-JSON (status ${res.status})`);
    }

    if (!res.ok) {
      const msg = data?.error || `Predict API error: ${res.status}`;
      throw new Error(msg);
    }

    return normalizePredictResponse(data);
  }

  // Global API for detail.js
  window.EuroWeatherPrediction = {
    fetchPrediction,
  };
})();
