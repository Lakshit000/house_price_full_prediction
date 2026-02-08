"""
tune_and_train.py

Performs a randomized hyperparameter search on XGBoost (or GradientBoosting) and
saves the best model and preprocessing pipeline to `models/model_tuned.pkl`.

Usage: run this script from the repo root.
"""
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")

# Try XGBoost first
try:
    import xgboost as xgb
    from xgboost import XGBRegressor
    USE_XGBOOST = True
    print("[OK] XGBoost available")
except Exception:
    USE_XGBOOST = False
    from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor
    print("[WARN] XGBoost not available; falling back to GradientBoostingRegressor")

# Paths
DATA_CSV = os.path.join("data", "house_data.csv")
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
OUT_PATH = os.path.join(MODELS_DIR, "model_tuned.pkl")
FEATURES_OUT = os.path.join(MODELS_DIR, "model_features_tuned.pkl")

df = pd.read_csv(DATA_CSV)
if "Price" not in df.columns:
    raise RuntimeError("CSV must contain a 'Price' column")

# Basic cleaning
df = df.dropna(subset=["Price"])
df = df[df["Price"] > 0]

# Remove extreme outliers by 0.5/99.5 quantiles for Price and living area
for col in ["Price", "living area"]:
    if col in df.columns:
        lo = df[col].quantile(0.005)
        hi = df[col].quantile(0.995)
        df = df[(df[col] >= lo) & (df[col] <= hi)]

# Feature engineering (keep it simple and deterministic)
current_year = datetime.now().year
if "Built Year" in df.columns:
    df["house_age"] = (current_year - df["Built Year"]).clip(lower=0)
else:
    df["house_age"] = 0

if "Renovation Year" in df.columns:
    df["is_renovated"] = (df["Renovation Year"] > 0).astype(int)
    df["years_since_renovation"] = (current_year - df["Renovation Year"]).clip(lower=0)
else:
    df["is_renovated"] = 0
    df["years_since_renovation"] = 0

if "living area" in df.columns and "lot area" in df.columns:
    df["living_to_lot_ratio"] = df["living area"] / (df["lot area"] + 1)
    df["total_area"] = df["living area"] + df.get("Area of the basement", 0).fillna(0)

base_features = [
    "living area",
    "number of bedrooms",
    "number of bathrooms",
    "number of floors",
    "condition of the house",
    "grade of the house",
    "Area of the house(excluding basement)",
    "Area of the basement",
    "Built Year"
]

additional = []
for c in ["lot area", "waterfront present", "number of views", "Number of schools nearby", "Distance from the airport"]:
    if c in df.columns:
        additional.append(c)

engineered = [c for c in ["house_age", "is_renovated", "years_since_renovation", "living_to_lot_ratio", "total_area"] if c in df.columns]

feature_cols = base_features + additional + engineered
feature_cols = [c for c in feature_cols if c in df.columns]

X = df[feature_cols].copy()
y = df["Price"].copy()
y_log = np.log1p(y)

imputer = SimpleImputer(strategy="median")
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=X_imp.columns)

X_train, X_test, y_train_log, y_test_log = train_test_split(X_scaled, y_log, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

if USE_XGBOOST:
    estimator = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
    param_dist = {
        'n_estimators': [100, 200, 400, 800],
        'max_depth': [3, 4, 6, 8],
        'learning_rate': [0.01, 0.02, 0.05, 0.1],
        'subsample': [0.6, 0.7, 0.8, 1.0],
        'colsample_bytree': [0.5, 0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0.5, 1.0, 2.0]
    }
else:
    from sklearn.ensemble import GradientBoostingRegressor
    estimator = GradientBoostingRegressor(random_state=42)
    param_dist = {
        'n_estimators': [200, 400, 800],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.02, 0.05],
        'subsample': [0.6, 0.7, 0.8],
        'max_features': ['sqrt', 0.6, 0.8]
    }

print("Starting randomized search (this may take a while)...")
search = RandomizedSearchCV(
    estimator=estimator,
    param_distributions=param_dist,
    n_iter=12,
    scoring='neg_mean_absolute_error',
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

search.fit(X_train, y_train_log)

best = search.best_estimator_
print("Best params:", search.best_params_)

# Evaluate on test
y_pred_log = best.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test_log)

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print("\nTUNED MODEL PERFORMANCE")
print(f"MAE: {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R2: {r2:.4f}")
print(f"MAPE: {mape:.2f}%")

# Save trained pipeline
with open(OUT_PATH, 'wb') as f:
    pickle.dump({
        'model': best,
        'imputer': imputer,
        'scaler': scaler,
        'log_target': True
    }, f)

with open(FEATURES_OUT, 'wb') as f:
    pickle.dump(feature_cols, f)

print(f"Saved tuned model -> {OUT_PATH}")

print("Done.")
