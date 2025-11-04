import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os

# Adjust path if your CSV is in data/ folder
CSV_PATH = os.path.join('data', 'house_data.csv')

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Place your dataset at {CSV_PATH} and run this script.")

df = pd.read_csv(CSV_PATH)

# Rename commonly used columns (adjust if your column names differ)
df = df.rename(columns={
    'number of bedrooms': 'Bedrooms',
    'number of bathrooms': 'Bathrooms',
    'living area': 'LivingArea',
    'number of floors': 'Floors',
    'condition of the house': 'Condition',
    'grade of the house': 'Grade',
    'Area of the house(excluding basement)': 'AreaNoBasement',
    'Area of the basement': 'BasementArea',
    'Built Year': 'BuiltYear',
    'Price': 'Price'
})

# Drop rows with missing values in chosen features
features = ['LivingArea','Bedrooms','Bathrooms','Floors','Condition','Grade','AreaNoBasement','BasementArea','BuiltYear']
df = df.dropna(subset=features + ['Price'])

X = df[features]
y = df['Price']

model = LinearRegression()
model.fit(X, y)

# Save model and feature list
pickle.dump(model, open('linear_regression_model.pkl', 'wb'))
pickle.dump(features, open('model_features.pkl', 'wb'))

print('Trained linear regression model saved: linear_regression_model.pkl')
print('Features saved: model_features.pkl')
