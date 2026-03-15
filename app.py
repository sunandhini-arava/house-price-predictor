"""
app.py  –  Flask backend for Hyderabad House Price Predictor
Run: python app.py
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib, json, os
import numpy as np
import pandas as pd

app = Flask(__name__, static_folder="static")
CORS(app)

# ── Load artefacts ────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

model    = joblib.load(os.path.join(BASE, "house_price_model.pkl"))
scaler   = joblib.load(os.path.join(BASE, "scaler.pkl"))
encoders = joblib.load(os.path.join(BASE, "encoders.pkl"))

with open(os.path.join(BASE, "features.json"))         as f: FEATURES         = json.load(f)
with open(os.path.join(BASE, "encoder_classes.json"))  as f: ENCODER_CLASSES  = json.load(f)
with open(os.path.join(BASE, "model_metrics.json"))    as f: MODEL_METRICS    = json.load(f)

FEAT_IMP_PATH = os.path.join(BASE, "feature_importance.json")
FEATURE_IMPORTANCE = {}
if os.path.exists(FEAT_IMP_PATH):
    with open(FEAT_IMP_PATH) as f:
        FEATURE_IMPORTANCE = json.load(f)

LOCALITY_BASE_PRICES = {
    "Banjara Hills": 12000, "Jubilee Hills": 11500, "Gachibowli": 9500,
    "HITEC City": 9000, "Kondapur": 8000, "Madhapur": 8500,
    "Kukatpally": 6500, "Miyapur": 5500, "Manikonda": 6000,
    "Nanakramguda": 8200, "Financial District": 8800, "Kokapet": 7500,
    "Narsingi": 6800, "Attapur": 5200, "Mehdipatnam": 5800,
    "Ameerpet": 6200, "Begumpet": 7000, "Secunderabad": 6500,
    "LB Nagar": 4800, "Uppal": 4500, "Dilsukhnagar": 5000,
    "AS Rao Nagar": 4200, "Kompally": 4800, "Bachupally": 5000,
    "Nizampet": 5200, "Chandanagar": 5500, "Tellapur": 6000,
    "Gopanpally": 7000, "Puppalaguda": 6500, "Shamshabad": 4000,
}

# ── Helper ────────────────────────────────────────────────────────────────────
def encode_input(data: dict) -> np.ndarray:
    cat_cols = ["locality", "property_type", "furnishing", "facing"]
    for col in cat_cols:
        if col in data:
            le = encoders[col]
            val = str(data[col])
            if val not in le.classes_:
                val = le.classes_[0]
            data[col] = int(le.transform([val])[0])

    area      = float(data.get("area_sqft", 1000))
    bedrooms  = int(data.get("bedrooms", 2))
    bathrooms = int(data.get("bathrooms", 2))
    floor     = int(data.get("floor", 0))
    tot_floor = int(data.get("total_floors", 1))

    data["amenity_score"] = sum([
        int(data.get("gym", 0)), int(data.get("swimming_pool", 0)),
        int(data.get("security", 0)), int(data.get("power_backup", 0)),
        int(data.get("lift", 0))
    ])
    data["room_ratio"]  = bathrooms / (bedrooms + 1)
    data["floor_ratio"] = floor / (tot_floor + 1)

    row = [data.get(f, 0) for f in FEATURES]
    return np.array(row).reshape(1, -1)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/meta")
def meta():
    return jsonify({
        "localities":      ENCODER_CLASSES.get("locality", []),
        "property_types":  ENCODER_CLASSES.get("property_type", []),
        "furnishing":      ENCODER_CLASSES.get("furnishing", []),
        "facing":          ENCODER_CLASSES.get("facing", []),
        "model_metrics":   MODEL_METRICS,
        "feature_importance": FEATURE_IMPORTANCE,
        "locality_prices": LOCALITY_BASE_PRICES,
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        X    = encode_input(data.copy())

        # Use scaler only for Ridge; XGB/RF work without it but apply anyway for safety
        price = float(model.predict(X)[0])
        price = max(price, 300000)

        # Price range ±8 %
        low  = round(price * 0.92 / 100000, 2)
        high = round(price * 1.08 / 100000, 2)
        mid  = round(price / 100000, 2)

        locality    = data.get("locality", "")
        base_psf    = LOCALITY_BASE_PRICES.get(locality, 6000)
        area        = float(data.get("area_sqft", 1000))
        market_avg  = round(base_psf * area / 100000, 2)

        return jsonify({
            "predicted_price_lakhs": mid,
            "range_low_lakhs":       low,
            "range_high_lakhs":      high,
            "market_avg_lakhs":      market_avg,
            "price_per_sqft":        round(price / area),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/compare", methods=["POST"])
def compare():
    """Compare prices across localities for same spec."""
    try:
        base    = request.get_json()
        results = []
        for loc, base_psf in LOCALITY_BASE_PRICES.items():
            d    = base.copy()
            d["locality"] = loc
            X    = encode_input(d)
            p    = float(model.predict(X)[0])
            results.append({"locality": loc, "price_lakhs": round(p / 100000, 2),
                             "price_per_sqft": base_psf})
        results.sort(key=lambda x: x["price_lakhs"])
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    print("🏠  Hyderabad House Price Predictor running at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
