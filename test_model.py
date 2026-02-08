"""
test_model.py - Verify the LightGBM model loads and works correctly with the app
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, 'models', 'model.pkl')
FEATURES_FILE = os.path.join(BASE_DIR, 'models', 'model_features.pkl')
DATA_CSV = os.path.join(BASE_DIR, 'data', 'house_data.csv')

print("="*70)
print("[TEST] Loading model and features...")
print("="*70)

# Load model
with open(MODEL_FILE, 'rb') as f:
    model_dict = pickle.load(f)
    
print(f"Model dict keys: {model_dict.keys()}")
print(f"  - model type: {type(model_dict['model'])}")
print(f"  - has imputer: {model_dict.get('imputer') is not None}")
print(f"  - has scaler: {model_dict.get('scaler') is not None}")
print(f"  - has encoder: {model_dict.get('encoder') is not None}")
print(f"  - log_target: {model_dict.get('log_target')}")

# Load features
with open(FEATURES_FILE, 'rb') as f:
    feature_list = pickle.load(f)

print(f"Feature count: {len(feature_list)}")
print(f"First 10 features: {feature_list[:10]}")

# Load a sample from data
df = pd.read_csv(DATA_CSV)
print(f"\nData shape: {df.shape}")

# Test sample prediction
print("\n" + "="*70)
print("[TEST] Test prediction with sample")
print("="*70)

sample = df.iloc[0:3].copy()

# Create minimal input
test_input = {
    'living area': 2000,
    'number of bedrooms': 3,
    'number of bathrooms': 2,
    'number of floors': 2,
    'condition of the house': 3,
    'grade of the house': 8,
    'Area of the house(excluding basement)': 1500,
    'Area of the basement': 500,
    'Built Year': 1990,
    'lot area': 5000,
    'waterfront present': 0,
    'number of views': 0,
    'Number of schools nearby': 2,
    'Distance from the airport': 15,
    'Postal Code': 98101,
    'Lattitude': 47.6,
    'Longitude': -122.3
}

test_df = pd.DataFrame([test_input])

# Check feature alignment
print(f"\nInput features: {list(test_df.columns)}")
missing = [f for f in feature_list if f not in test_df.columns]
print(f"Missing features ({len(missing)}): {missing[:20]}...")  # Show first 20

# Add missing features with 0
for col in missing:
    test_df[col] = 0

# Now subselect to match expected features
test_df = test_df[feature_list]
print(f"After subselecting: shape = {test_df.shape}")

# Apply same preprocessing as model
imputer = model_dict['imputer']
scaler = model_dict['scaler']
encoder = model_dict.get('encoder')

test_imp = pd.DataFrame(imputer.transform(test_df), columns=feature_list)
print(f"After imputation: {test_imp.shape}")

if encoder:
    test_enc = encoder.transform(test_imp)
    print(f"After encoding: {test_enc.shape}")
else:
    test_enc = test_imp
    print("No encoder applied")

test_scaled = pd.DataFrame(scaler.transform(test_enc), columns=feature_list)
print(f"After scaling: {test_scaled.shape}")

# Make prediction
pred_log = model_dict['model'].predict(test_scaled)
if model_dict.get('log_target'):
    pred_price = np.expm1(pred_log[0])
else:
    pred_price = pred_log[0]

print(f"\nPredicted price: ${pred_price:,.2f}")

print("\n" + "="*70)
print("[SUCCESS] Model loaded and prediction works!")
print("="*70)
