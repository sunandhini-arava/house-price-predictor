"""
train_model.py
Trains and saves the best ML model for Hyderabad house price prediction.
Run: python train_model.py
"""

import pandas as pd
import numpy as np
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

# ── 1. Load Data ──────────────────────────────────────────────────────────────
df = pd.read_csv("hyderabad_house_data.csv")
print(f"Dataset: {df.shape[0]} rows × {df.shape[1]} cols")

# ── 2. Feature Engineering ────────────────────────────────────────────────────
df["price_per_sqft"] = df["price"] / df["area_sqft"]   # for EDA only
df["amenity_score"]  = df["gym"] + df["swimming_pool"] + df["security"] + df["power_backup"] + df["lift"]
df["room_ratio"]     = df["bathrooms"] / (df["bedrooms"] + 1)
df["floor_ratio"]    = df["floor"] / (df["total_floors"] + 1)

# ── 3. Encode Categoricals ────────────────────────────────────────────────────
FEATURES = [
    "locality", "property_type", "area_sqft", "bedrooms", "bathrooms",
    "balconies", "age_years", "floor", "total_floors", "parking",
    "furnishing", "facing", "gym", "swimming_pool", "security",
    "power_backup", "lift", "amenity_score", "room_ratio", "floor_ratio"
]
TARGET = "price"

cat_cols = ["locality", "property_type", "furnishing", "facing"]
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Save encoder classes for the web app
encoder_classes = {col: list(enc.classes_) for col, enc in encoders.items()}
with open("encoder_classes.json", "w") as f:
    json.dump(encoder_classes, f, indent=2)

X = df[FEATURES]
y = df[TARGET]

# ── 4. Split ──────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── 5. Train Multiple Models ──────────────────────────────────────────────────
models = {
    "Ridge":            Ridge(alpha=1.0),
    "Random Forest":    RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    "Gradient Boost":   GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
    "XGBoost":          xgb.XGBRegressor(n_estimators=300, learning_rate=0.08, max_depth=6,
                                          subsample=0.85, colsample_bytree=0.85,
                                          random_state=42, verbosity=0),
}

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

results = []
print("\n{'Model':<20} {'R²':>8} {'RMSE':>14} {'MAE':>14}")
print("-" * 60)

best_r2    = -999
best_name  = ""
best_model = None

for name, model in models.items():
    if name == "Ridge":
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    r2   = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    results.append({"model": name, "r2": r2, "rmse": rmse, "mae": mae})
    print(f"{name:<20} {r2:>8.4f} {rmse:>14,.0f} {mae:>14,.0f}")

    if r2 > best_r2:
        best_r2    = r2
        best_name  = name
        best_model = model

print(f"\n🏆  Best model: {best_name}  (R² = {best_r2:.4f})")

# ── 6. Feature Importance ─────────────────────────────────────────────────────
if hasattr(best_model, "feature_importances_"):
    imp = pd.Series(best_model.feature_importances_, index=FEATURES)
    imp_dict = imp.sort_values(ascending=False).head(10).to_dict()
    with open("feature_importance.json", "w") as f:
        json.dump({k: float(v) for k, v in imp_dict.items()}, f, indent=2)

# ── 7. Save Artefacts ─────────────────────────────────────────────────────────
joblib.dump(best_model, "house_price_model.pkl")
joblib.dump(scaler,     "scaler.pkl")
joblib.dump(encoders,   "encoders.pkl")

with open("features.json", "w") as f:
    json.dump(FEATURES, f)

metrics = {"model": best_name, "r2": round(best_r2, 4),
           "rmse": round(results[-1]["rmse"]), "mae": round(results[-1]["mae"])}
with open("model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n✅  Saved: house_price_model.pkl, scaler.pkl, encoders.pkl")
print("✅  Saved: features.json, encoder_classes.json, model_metrics.json")
