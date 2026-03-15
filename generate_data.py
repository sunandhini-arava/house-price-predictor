"""
generate_data.py
Generates a realistic synthetic dataset for Hyderabad house prices.
Run this once before training: python generate_data.py
"""

import pandas as pd
import numpy as np

np.random.seed(42)
N = 3000

LOCALITIES = {
    "Banjara Hills":     {"base": 12000, "std": 2500},
    "Jubilee Hills":     {"base": 11500, "std": 2200},
    "Gachibowli":        {"base": 9500,  "std": 1800},
    "HITEC City":        {"base": 9000,  "std": 1700},
    "Kondapur":          {"base": 8000,  "std": 1500},
    "Madhapur":          {"base": 8500,  "std": 1600},
    "Kukatpally":        {"base": 6500,  "std": 1200},
    "Miyapur":           {"base": 5500,  "std": 1000},
    "Manikonda":         {"base": 6000,  "std": 1100},
    "Nanakramguda":      {"base": 8200,  "std": 1500},
    "Financial District":{"base": 8800,  "std": 1600},
    "Kokapet":           {"base": 7500,  "std": 1400},
    "Narsingi":          {"base": 6800,  "std": 1200},
    "Attapur":           {"base": 5200,  "std": 900},
    "Mehdipatnam":       {"base": 5800,  "std": 1000},
    "Ameerpet":          {"base": 6200,  "std": 1100},
    "Begumpet":          {"base": 7000,  "std": 1300},
    "Secunderabad":      {"base": 6500,  "std": 1200},
    "LB Nagar":          {"base": 4800,  "std": 900},
    "Uppal":             {"base": 4500,  "std": 850},
    "Dilsukhnagar":      {"base": 5000,  "std": 950},
    "AS Rao Nagar":      {"base": 4200,  "std": 800},
    "Kompally":          {"base": 4800,  "std": 900},
    "Bachupally":        {"base": 5000,  "std": 950},
    "Nizampet":          {"base": 5200,  "std": 950},
    "Chandanagar":       {"base": 5500,  "std": 1000},
    "Tellapur":          {"base": 6000,  "std": 1100},
    "Gopanpally":        {"base": 7000,  "std": 1300},
    "Puppalaguda":       {"base": 6500,  "std": 1200},
    "Shamshabad":        {"base": 4000,  "std": 750},
}

PROPERTY_TYPES = ["Apartment", "Villa", "Independent House", "Plot"]
FURNISHING     = ["Unfurnished", "Semi-Furnished", "Fully Furnished"]
FACING         = ["East", "West", "North", "South", "North-East", "North-West"]

rows = []
locality_list = list(LOCALITIES.keys())
locality_weights = [LOCALITIES[l]["base"] for l in locality_list]
locality_weights = np.array(locality_weights) / sum(locality_weights)

for _ in range(N):
    locality = np.random.choice(locality_list, p=locality_weights)
    info     = LOCALITIES[locality]

    prop_type  = np.random.choice(PROPERTY_TYPES, p=[0.55, 0.15, 0.20, 0.10])
    bedrooms   = np.random.choice([1, 2, 3, 4, 5], p=[0.10, 0.35, 0.35, 0.15, 0.05])
    bathrooms  = min(bedrooms + np.random.choice([0, 1]), 5)
    balconies  = np.random.choice([0, 1, 2, 3], p=[0.15, 0.45, 0.30, 0.10])

    if prop_type == "Apartment":
        area = bedrooms * np.random.uniform(450, 700)
    elif prop_type == "Villa":
        area = bedrooms * np.random.uniform(700, 1200)
    elif prop_type == "Independent House":
        area = bedrooms * np.random.uniform(500, 900)
    else:  # Plot
        area = np.random.uniform(200, 2000)
        bedrooms = bathrooms = balconies = 0

    age         = np.random.randint(0, 25)
    floor       = np.random.randint(0, 25) if prop_type == "Apartment" else 0
    total_floor = floor + np.random.randint(0, 10) if prop_type == "Apartment" else 0
    parking     = np.random.choice([0, 1, 2], p=[0.20, 0.60, 0.20])
    furnishing  = np.random.choice(FURNISHING)
    facing      = np.random.choice(FACING)
    gym         = np.random.choice([0, 1], p=[0.55, 0.45])
    swimming    = np.random.choice([0, 1], p=[0.65, 0.35])
    security    = np.random.choice([0, 1], p=[0.30, 0.70])
    power_back  = np.random.choice([0, 1], p=[0.35, 0.65])
    lift        = 1 if (prop_type == "Apartment" and total_floor > 3) else np.random.choice([0, 1], p=[0.60, 0.40])

    base_price = info["base"] + np.random.normal(0, info["std"] * 0.3)
    price = base_price * area

    # Adjustments
    price *= (1 + 0.03 * bedrooms)
    price *= (1 + 0.01 * bathrooms)
    price *= max(0.85, 1 - age * 0.01)
    price *= (1.05 if furnishing == "Fully Furnished" else 0.98 if furnishing == "Unfurnished" else 1.0)
    price *= (1.02 if facing in ["East", "North-East"] else 1.0)
    price *= (1 + 0.01 * gym + 0.015 * swimming + 0.005 * security + 0.005 * power_back + 0.005 * lift)
    price += np.random.normal(0, price * 0.05)
    price = max(price, 500000)

    rows.append({
        "locality":       locality,
        "property_type":  prop_type,
        "area_sqft":      round(area, 1),
        "bedrooms":       bedrooms,
        "bathrooms":      bathrooms,
        "balconies":      balconies,
        "age_years":      age,
        "floor":          floor,
        "total_floors":   total_floor,
        "parking":        parking,
        "furnishing":     furnishing,
        "facing":         facing,
        "gym":            gym,
        "swimming_pool":  swimming,
        "security":       security,
        "power_backup":   power_back,
        "lift":           lift,
        "price":          round(price),
    })

df = pd.DataFrame(rows)
df.to_csv("hyderabad_house_data.csv", index=False)
print(f"✅  Dataset saved: hyderabad_house_data.csv  ({len(df)} rows)")
print(df.head())
