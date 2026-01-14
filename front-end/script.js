function predictDisease() {
    const image = document.getElementById("cropImage").files[0];
    let formData = new FormData();
    formData.append("image", image);

    fetch("http://127.0.0.1:5000/predict-disease", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("diseaseResult").innerText =
            "🦠 Disease: " + data.prediction;
    })
    .catch(err => {
        alert("API Error");
    });
}
function previewImage(event) {
    const preview = document.getElementById("imagePreview");
    const file = event.target.files[0];

    if (!file) return;

    const reader = new FileReader();

    reader.onload = function () {
        preview.src = reader.result;
        preview.classList.remove("hidden");
    };

    reader.readAsDataURL(file);
}
function predictFertilizer() {
    if (!window.selectedLat || !window.selectedLon) {
        document.getElementById("fertilizerResult").innerText =
            "⚠️ Please select a location on the map";
        return;
    }

    fetch("http://127.0.0.1:5000/predict-fertilizer", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            nitrogen: nitrogen.value,
            phosphorus: phosphorus.value,
            potassium: potassium.value,
            lat: window.selectedLat,
            lon: window.selectedLon
        })
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("fertilizerResult").innerHTML =
            `🌱 <b>Recommended:</b> ${data.fertilizer}<br>
             🌡️ Temp: ${data.weather.temperature} °C<br>
             💧 Humidity: ${data.weather.humidity}%`;
    })
    .catch(() => {
        document.getElementById("fertilizerResult").innerText =
            "❌ Prediction failed";
    });
}


let map = L.map('map').setView([20.5937, 78.9629], 5); // India center
let marker;

// OpenStreetMap tiles
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap'
}).addTo(map);

// Click on map to select location
map.on('click', function (e) {
    const { lat, lng } = e.latlng;

    if (marker) {
        marker.setLatLng([lat, lng]);
    } else {
        marker = L.marker([lat, lng]).addTo(map);
    }

    marker.bindPopup(
        `📍 Selected Location<br>Lat: ${lat.toFixed(4)}<br>Lon: ${lng.toFixed(4)}`
    ).openPopup();

    // Save coordinates globally
    window.selectedLat = lat;
    window.selectedLon = lng;
});
