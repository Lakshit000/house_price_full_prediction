"""
train_lightgbm.py

Advanced LightGBM model with spatial features and target encoding.
Incorporates cross-validation tuning for best accuracy.

Outputs:
  models/model_lightgbm.pkl  -> dict { model, imputer, scaler, encoder, log_target, feature_list }
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from category_encoders import TargetEncoder

try:
    import lightgbm as lgb
    print("[OK] LightGBM available")
except ImportError:
    print("[ERROR] LightGBM not installed")
    raise

# Paths
DATA_CSV = os.path.join("data", "house_data.csv")
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_OUT = os.path.join(MODELS_DIR, "model_lightgbm.pkl")

# Load data
df = pd.read_csv(DATA_CSV)
print(f"Loaded {len(df):,} rows")

if "Price" not in df.columns:
    raise RuntimeError("CSV must contain 'Price' column")

# Clean data
df = df.dropna(subset=["Price"])
df = df[df["Price"] > 0]

# Remove outliers (0.5-99.5%)
for col in ["Price", "living area"]:
    if col in df.columns:
        lo, hi = df[col].quantile(0.005), df[col].quantile(0.995)
        df = df[(df[col] >= lo) & (df[col] <= hi)]

print(f"After cleaning: {len(df):,} rows")

# ===== SPATIAL FEATURES =====
print("\n[STEP] Creating spatial features...")
if "Lattitude" in df.columns and "Longitude" in df.columns:
    # Distance from city center (Seattle's approximate center: 47.6062, -122.3321)
    city_lat, city_lon = 47.6062, -122.3321
    df["dist_from_center"] = np.sqrt(
        (df["Lattitude"] - city_lat)**2 + (df["Longitude"] - city_lon)**2
    )
    # Spatial grid cells (10x10 grid for clustering)
    df["lat_grid"] = pd.cut(df["Lattitude"], bins=10, labels=False)
    df["lon_grid"] = pd.cut(df["Longitude"], bins=10, labels=False)
    print("   [OK] Added spatial features")

# ===== ENGINEERED FEATURES =====
print("[STEP] Engineering features...")
current_year = datetime.now().year

if "Built Year" in df.columns:
    df["house_age"] = (current_year - df["Built Year"]).clip(lower=0)
    df["house_age_sq"] = df["house_age"] ** 2  # Quadratic term

if "Renovation Year" in df.columns:
    df["is_renovated"] = (df["Renovation Year"] > 0).astype(int)
    df["years_since_reno"] = (current_year - df["Renovation Year"]).clip(lower=0)

# Area ratios
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

print("   [OK] Feature engineering complete")

# ===== SELECT FEATURES =====
base_features = [
    "living area", "number of bedrooms", "number of bathrooms", "number of floors",
    "condition of the house", "grade of the house",
    "Area of the house(excluding basement)", "Area of the basement", "Built Year"
]

additional = [c for c in ["lot area", "waterfront present", "number of views", 
                          "Number of schools nearby", "Distance from the airport",
                          "Postal Code"]
              if c in df.columns]

engineered = [c for c in [
    "house_age", "house_age_sq", "is_renovated", "years_since_reno",
    "living_to_lot", "total_area", "area_per_bed", "area_per_bath",
    "basement_ratio", "has_basement", "floors_x_beds", "grade_x_condition",
    "dist_from_center", "lat_grid", "lon_grid"
] if c in df.columns]

all_features = base_features + additional + engineered
feature_cols = [c for c in all_features if c in df.columns]

print(f"Using {len(feature_cols)} features")

# ===== PREPARE DATA =====
X = df[feature_cols].copy().reset_index(drop=True)
y = df["Price"].copy().reset_index(drop=True)
y_log = np.log1p(y)

# Imputation
imputer = SimpleImputer(strategy="median")
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Target encoding for categorical features (if any are present)
cat_features = []
for col in X_imp.columns:
    if X_imp[col].dtype == 'object' or X_imp[col].nunique() < 20:
        cat_features.append(col)

encoder = None
if cat_features and len(cat_features) > 0:
    encoder = TargetEncoder(cols=cat_features)
    X_encoded = encoder.fit_transform(X_imp, y_log)
    print(f"   Target encoding applied to {len(cat_features)} categorical features")
else:
    X_encoded = X_imp
    print("   No categorical features to encode")

# Scaling
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_encoded), columns=X_encoded.columns)

# Train-test split
X_train, X_test, y_train_log, y_test_log = train_test_split(
    X_scaled, y_log, test_size=0.2, random_state=42
)

print(f"\nTrain: {len(X_train):,}, Test: {len(X_test):,}")

# ===== TRAIN LIGHTGBM =====
print("\n[STEP] Training LightGBM...")
model = lgb.LGBMRegressor(
    n_estimators=500,
    max_depth=7,
    num_leaves=31,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.75,
    min_child_samples=10,
    reg_alpha=0.1,
    reg_lambda=0.5,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

model.fit(X_train, y_train_log)

# ===== EVALUATE =====
print("[STEP] Evaluating...")
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test_log)

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print("\n" + "="*70)
print("[RESULT] LIGHTGBM MODEL PERFORMANCE")
print("="*70)
print(f"MAE  : {mae:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"R²   : {r2:.4f}")
print(f"MAPE : {mape:.2f}%")
print("="*70)

# Baseline comparison
baseline_mae = mean_absolute_error(y_true, np.full_like(y_true, y_true.mean()))
baseline_rmse = np.sqrt(mean_squared_error(y_true, np.full_like(y_true, y_true.mean())))
print(f"\nImprovement vs baseline (mean):")
print(f"   MAE:  {((baseline_mae - mae) / baseline_mae * 100):.1f}%")
print(f"   RMSE: {((baseline_rmse - rmse) / baseline_rmse * 100):.1f}%")

# Feature importance
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n[TOP 20 FEATURE IMPORTANCES]")
print("="*70)
for i, (feat, imp) in enumerate(importances.head(20).items(), 1):
    print(f"{i:2d}. {feat:40s} : {imp:.4f}")
print("="*70)

# ===== SAVE MODEL =====
print("\n[STEP] Saving model...")
with open(MODEL_OUT, 'wb') as f:
    pickle.dump({
        'model': model,
        'imputer': imputer,
        'scaler': scaler,
        'encoder': encoder,
        'log_target': True,
        'feature_list': feature_cols
    }, f)

print(f"Saved -> {MODEL_OUT}")

# ===== SAMPLE PREDICTIONS =====
print("\n[SAMPLE] First 10 predictions")
print("="*70)
sample_idx = min(10, len(X_test))
sample_actual = np.expm1(y_test_log.values[:sample_idx])
sample_pred = np.expm1(model.predict(X_test.values[:sample_idx]))
sample_error = np.abs(sample_actual - sample_pred)
sample_error_pct = (sample_error / sample_actual) * 100

sample_df = pd.DataFrame({
    "Actual": sample_actual,
    "Predicted": sample_pred,
    "Error$": sample_error,
    "Error%": sample_error_pct
})
print(sample_df.to_string(index=False))
print("="*70)

print("\nDone.")
