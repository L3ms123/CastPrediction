// map_logic.js
document.addEventListener("DOMContentLoaded", () => {
  const mapDiv = document.getElementById("mapContainer");
  if (!mapDiv) return;

  const defaultCenter = [52.52, 13.405];

  // 1) Guardar el mapa en window.map (GLOBAL)
  window.map = L.map("mapContainer", {
    center: defaultCenter,
    zoom: 11,
    zoomControl: false
  });

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "© OpenStreetMap"
  }).addTo(window.map);

  let selectionMarker = null;
  let selectedLatLng = null;

  const coordTooltip = document.getElementById("coordTooltip");
  const coordText = document.getElementById("coordText");

  // Click en el mapa: elegir coordenadas
  window.map.on("click", (e) => {
    const { lat, lng } = e.latlng;
    selectedLatLng = e.latlng;

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

