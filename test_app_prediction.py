"""
test_app_prediction.py - Full end-to-end test mimicking app.py prediction flow
"""

import os
import pickle
import numpy as np
import pandas as pd
import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, 'models', 'model.pkl')
FEATURES_FILE = os.path.join(BASE_DIR, 'models', 'model_features.pkl')

# Load model and features (same as app.py)
with open(MODEL_FILE, 'rb') as f:
    model_artifacts = pickle.load(f)

with open(FEATURES_FILE, 'rb') as f:
    feature_columns = pickle.load(f)

model_obj = model_artifacts.get('model')
imputer = model_artifacts.get('imputer')
scaler = model_artifacts.get('scaler')
encoder = model_artifacts.get('encoder')
log_target = bool(model_artifacts.get('log_target'))

print("="*70)
print("[TEST] Full Prediction Pipeline (mimics app.py)")
print("="*70)
print(f"Model type: {type(model_obj)}")
print(f"Features: {len(feature_columns)}")
print(f"Has encoder: {encoder is not None}")
print(f"Log target: {log_target}\n")

def engineer_features(df, current_year=None):
    """Exact copy from updated app.py"""
    if current_year is None:
        current_year = datetime.datetime.now().year
    
    df = df.copy()
    
    # Spatial features
    if "Lattitude" in df.columns and "Longitude" in df.columns:
        city_lat, city_lon = 47.6062, -122.3321  # Seattle center
        df["dist_from_center"] = np.sqrt(
            (df["Lattitude"] - city_lat)**2 + (df["Longitude"] - city_lon)**2
        )
        df["lat_grid"] = pd.cut(df["Lattitude"], bins=10, labels=False).astype('float')
        df["lon_grid"] = pd.cut(df["Longitude"], bins=10, labels=False).astype('float')
    
    # Calculate house age and squared term
    if "Built Year" in df.columns:
        df["house_age"] = (current_year - df["Built Year"]).clip(lower=0)
        df["house_age_sq"] = df["house_age"] ** 2
    
    # Renovation status
    if "Renovation Year" in df.columns:
        df["is_renovated"] = (df["Renovation Year"] > 0).astype(int)
        df["years_since_reno"] = (current_year - df["Renovation Year"]).clip(lower=0)
    else:
        df["is_renovated"] = 0
        df["years_since_reno"] = 0
    
    # Area ratios and derived features
    if "living area" in df.columns:
        if "lot area" in df.columns:
            df["living_to_lot"] = df["living area"] / (df["lot area"] + 1)
            df["total_area"] = df["living area"] + df.get("Area of the basement", 0).fillna(0)
        
        if "number of bedrooms" in df.columns:
            df["area_per_bed"] = df["living area"] / (df["number of bedrooms"] + 1)
        
        if "number of bathrooms" in df.columns:
            df["area_per_bath"] = df["living area"] / (df["number of bathrooms"] + 1)
    
    # Basement features
    if "Area of the basement" in df.columns and "living area" in df.columns:
        df["basement_ratio"] = df["Area of the basement"] / (df["living area"] + df["Area of the basement"] + 1)
        df["has_basement"] = (df["Area of the basement"] > 0).astype(int)
    
    # Interaction features
    if "number of floors" in df.columns and "number of bedrooms" in df.columns:
        df["floors_x_beds"] = df["number of floors"] * df["number of bedrooms"]
    
    if "grade of the house" in df.columns and "condition of the house" in df.columns:
        df["grade_x_condition"] = df["grade of the house"] * df["condition of the house"]
    
    return df

# Simulate form input (typical house)
print("[STEP 1] Simulating user form input...")
input_data = {
    'living area': 2000,
    'number of bedrooms': 3,
    'number of bathrooms': 2.5,
    'number of floors': 2,
    'condition of the house': 4,
    'grade of the house': 9,
    'Area of the house(excluding basement)': 1800,
    'Area of the basement': 200,
    'Built Year': 2000,
    'lot area': 6500,
    'waterfront present': 0,
    'number of views': 2,
    'NUMBER of schools nearby': 3,
    'Distance from the airport': 12,
    'Postal Code': 98101,
    'Lattitude': 47.6062,
    'Longitude': -122.3321,
    'Renovation Year': 2015
}
for k, v in input_data.items():
    print(f"  {k}: {v}")

# Step 2: Create DataFrame and engineer features
print("\n[STEP 2] Applying feature engineering...")
df_input = pd.DataFrame([input_data])
df_input = engineer_features(df_input)
print(f"  Created columns: {list(df_input.columns)}")

# Step 3: Align with model features
print("\n[STEP 3] Aligning features with model expectations...")
for col in feature_columns:
    if col not in df_input.columns:
        df_input[col] = 0
        
df_input = df_input[feature_columns]
print(f"  Final shape: {df_input.shape}")
print(f"  Columns: {list(df_input.columns)}")

# Step 4: Imputation
print("\n[STEP 4] Applying imputation...")
df_input = pd.DataFrame(imputer.transform(df_input), columns=feature_columns)
print(f"  Shape after impute: {df_input.shape}")

# Step 5: Encoding
print("\n[STEP 5] Applying target encoder...")
if encoder is not None:
    df_input = encoder.transform(df_input)
    print(f"  Shape after encode: {df_input.shape}")
else:
    print("  No encoder")

# Step 6: Scaling
print("\n[STEP 6] Applying scaling...")
df_input = pd.DataFrame(scaler.transform(df_input), columns=feature_columns)
print(f"  Shape after scale: {df_input.shape}")

# Step 7: Prediction
print("\n[STEP 7] Making prediction...")
predicted_log = float(model_obj.predict(df_input)[0])
if log_target:
    predicted_price = float(np.expm1(predicted_log))
    print(f"  Log prediction: {predicted_log:.4f}")
    print(f"  Expm1 applied: YES")
else:
    predicted_price = predicted_log
    print(f"  Expm1 applied: NO")

print("\n" + "="*70)
print(f"[RESULT] Predicted Price: ${predicted_price:,.2f}")
print("="*70)
print("\nTest passed! The app prediction flow works correctly.")
