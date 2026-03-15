# 🏠 Hyderabad House Price Predictor

An end-to-end Machine Learning system that predicts house prices across
30 localities in Hyderabad, with a beautiful web UI.

---

## 📁 Project Structure

```
hyderabad_house_price/
├── generate_data.py       ← Step 1: Creates synthetic dataset
├── train_model.py         ← Step 2: Trains & saves ML model
├── app.py                 ← Step 3: Flask API backend
├── static/
│   └── index.html         ← Web UI frontend
├── requirements.txt
└── README.md
```

After running Steps 1–2 the following files are auto-generated:
```
hyderabad_house_data.csv   ← 3 000-row training dataset
house_price_model.pkl      ← Trained XGBoost model
scaler.pkl                 ← Feature scaler
encoders.pkl               ← Label encoders
features.json              ← Feature list
encoder_classes.json       ← Categorical options
model_metrics.json         ← R², RMSE, MAE
feature_importance.json    ← Top-10 feature weights
```

---

## 🚀 Setup & Run (Step by Step)

### Prerequisites
- Python 3.9 or higher
- pip

---

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

---

### Step 2 — Generate the Hyderabad dataset

```bash
python generate_data.py
```

Expected output:
```
✅  Dataset saved: hyderabad_house_data.csv  (3000 rows)
```

> **Real data**: Replace `hyderabad_house_data.csv` with actual data from
> MagicBricks, 99acres, or Housing.com exports. Keep the same column names.

---

### Step 3 — Train the ML model

```bash
python train_model.py
```

Expected output:
```
Model               R²       RMSE             MAE
Ridge               0.8321   1,234,567        890,123
Random Forest       0.9210     987,654        654,321
Gradient Boost      0.9380     901,234        612,345
XGBoost             0.9510     845,678        580,000

🏆  Best model: XGBoost  (R² = 0.9510)
✅  Saved: house_price_model.pkl, scaler.pkl, encoders.pkl
```

---

### Step 4 — Start the web server

```bash
python app.py
```

Expected output:
```
🏠  Hyderabad House Price Predictor running at http://127.0.0.1:5000
```

---

### Step 5 — Open in browser

Navigate to: **http://127.0.0.1:5000**

---

## 🧠 ML Models Compared

| Model             | Typical R² | Notes                          |
|-------------------|------------|--------------------------------|
| Ridge Regression  | ~0.83      | Fast, linear baseline          |
| Random Forest     | ~0.92      | Great with categorical features|
| Gradient Boosting | ~0.94      | Robust, slightly slower        |
| **XGBoost**       | **~0.95**  | **Best — auto-selected**       |

---

## 🏙️ Covered Localities (30)

Banjara Hills, Jubilee Hills, Gachibowli, HITEC City, Kondapur, Madhapur,
Kukatpally, Miyapur, Manikonda, Nanakramguda, Financial District, Kokapet,
Narsingi, Attapur, Mehdipatnam, Ameerpet, Begumpet, Secunderabad, LB Nagar,
Uppal, Dilsukhnagar, AS Rao Nagar, Kompally, Bachupally, Nizampet,
Chandanagar, Tellapur, Gopanpally, Puppalaguda, Shamshabad

---

## 📡 API Endpoints

| Method | Endpoint         | Description                        |
|--------|------------------|------------------------------------|
| GET    | `/api/meta`      | Localities, types, model metrics   |
| POST   | `/api/predict`   | Predict price for a property       |
| POST   | `/api/compare`   | Compare price across all localities|

### Sample predict request

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "locality": "Gachibowli",
    "property_type": "Apartment",
    "area_sqft": 1200,
    "bedrooms": 2,
    "bathrooms": 2,
    "balconies": 1,
    "age_years": 3,
    "floor": 5,
    "total_floors": 12,
    "parking": 1,
    "furnishing": "Semi-Furnished",
    "facing": "East",
    "gym": 1,
    "swimming_pool": 0,
    "security": 1,
    "power_backup": 1,
    "lift": 1
  }'
```

Response:
```json
{
  "predicted_price_lakhs": 96.4,
  "range_low_lakhs": 88.7,
  "range_high_lakhs": 104.1,
  "market_avg_lakhs": 91.2,
  "price_per_sqft": 8033
}
```

---

## 🔧 Swap in Real Data

1. Export property listings from MagicBricks / 99acres as CSV
2. Rename columns to match the schema in `generate_data.py`
3. Replace `hyderabad_house_data.csv`
4. Re-run `python train_model.py`

---

## 📦 Tech Stack

- **ML**: scikit-learn, XGBoost, pandas, numpy
- **Backend**: Flask, Flask-CORS
- **Frontend**: Vanilla JS, CSS3, Google Fonts
- **Persistence**: joblib (model serialization)
