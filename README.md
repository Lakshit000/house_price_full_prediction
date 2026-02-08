# House Price Prediction - Full Stack ML Application

A Flask-based web application for predicting house prices using LightGBM with advanced feature engineering (spatial features, target encoding, interaction terms).

## Features

✨ **Advanced ML Model**
- LightGBM regressor with 30 engineered features
- Spatial features (distance from city center, location grids)
- Target encoding for categorical variables
- 88.7% R² accuracy on test data

🎯 **Core Capabilities**
- Real-time price predictions
- User authentication & role-based access
- Prediction history tracking
- Admin dashboard for model performance monitoring
- Activity logging

📊 **Performance**
- MAE: $61,519 (down from $113,140)
- RMSE: $99,967 (down from $169,304)
- MAPE: 11.6% (down from 22.5%)

## Local Development

### Prerequisites
- Python 3.8+
- Virtual environment (venv)

### Setup

```bash
# Clone repository
git clone https://github.com/Lakshit000/house_price_full_prediction.git
cd house_price_full_prediction

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the model (optional - pre-trained model included)
python train_lightgbm.py

# Run the application
python app.py
```

Open browser to `http://localhost:5000`

### Default Credentials
- Username: `admin`
- Password: `admin`

## Model Training

### Quick Train (LightGBM with Spatial Features)
```bash
python train_lightgbm.py
```
Outputs: `models/model_lightgbm.pkl` with 30 features

### Hyperparameter Tuning
```bash
python tune_and_train.py
```
Performs RandomizedSearchCV across 12 parameter combinations

### Test Predictions
```bash
python test_app_prediction.py
```

## Deployment on Render.com

### Step 1: Push to GitHub
```bash
git add .
git commit -m "Deploy to Render"
git push origin main
```

### Step 2: Create Render Account
1. Go to [https://render.com](https://render.com)
2. Sign up with GitHub account
3. Grant access to your repositories

### Step 3: Create New Web Service
1. Click **New +** → **Web Service**
2. Select your `house_price_full_prediction` repository
3. Configure:
   - **Name**: `house-price-predictor`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Plan**: `Free` (or Starter for production)

### Step 4: Set Environment Variables
In Render dashboard, go to **Environment** and add:
```
FLASK_ENV=production
FLASK_SECRET=your-secret-key-here
```

### Step 5: Deploy
- Click **Create Web Service**
- Render automatically builds and deploys
- Your app will be live at `https://house-price-predictor.onrender.com`

## Auto-Deploy from GitHub

For automatic deployments on every push:
1. In Render dashboard, go to your service settings
2. Enable **Auto-Deploy** on push
3. Any new commits to `main` branch trigger automatic re-deployment

## Project Structure

```
├── app.py                    # Flask application
├── train_lightgbm.py         # Model training with spatial features
├── tune_and_train.py         # Hyperparameter optimization
├── requirements.txt          # Python dependencies
├── Procfile                  # Render deployment config
├── render.yaml              # Render service definition
├── models/
│   ├── model.pkl            # Trained LightGBM model
│   └── model_features.pkl   # Feature column names
├── data/
│   └── house_data.csv       # Training dataset
├── templates/               # HTML templates
│   ├── base.html
│   ├── predict.html
│   ├── dashboard.html
│   └── admin_*.html
└── static/
    ├── style.css
    └── theme.js
```

## API Routes

### Public
- `GET /` - Home page
- `POST /signup` - User registration
- `POST /login` - User login

### Authenticated
- `GET /predict` - Prediction form
- `POST /predict` - Make prediction
- `GET /history` - View prediction history
- `GET /dashboard` - User dashboard
- `GET /visuals` - Performance visualizations
- `GET /model_performance` - Model metrics

### Admin
- `GET /admin` - Admin dashboard
- `GET /admin/users` - Manage users
- `GET /admin/activity` - View activity log
- `GET /admin/export-predictions` - Export data

## Database

The app uses SQLite (`data/app.db`) with tables:
- `users` - User accounts and roles
- `predictions` - Prediction history
- `activity` - Audit log

## Technologies

- **Backend**: Flask, Python 3.8+
- **ML**: LightGBM, scikit-learn, pandas, numpy
- **Database**: SQLite
- **Server**: Gunicorn
- **Frontend**: HTML5, CSS3, JavaScript
- **Deployment**: Render.com

## Model Features (30 Total)

**Base Features** (9):
- living area, bedrooms, bathrooms, floors
- condition, grade, basement area, house age

**Spatial** (5):
- distance from city center, lat/lon grids
- postal code (target encoded)

**Engineered** (16):
- house_age, house_age_sq, is_renovated, years_since_reno
- living_to_lot, total_area, area_per_bed, area_per_bath
- basement_ratio, has_basement, floors_x_beds, grade_x_condition
- Plus airport distance, schools nearby, waterfront, views

## Performance Metrics

```
Training Set: 11,468 samples
Test Set: 2,868 samples

Results:
- R² Score: 0.8872 (explains 88.7% of variance)
- MAE: $61,519
- RMSE: $99,967
- MAPE: 11.60%
- vs Baseline: 70.9% MAE improvement
```

## Troubleshooting

### Model Loading Error
- Ensure `models/model.pkl` and `models/model_features.pkl` exist
- Run `python train_lightgbm.py` to retrain

### Port Already in Use
```bash
python app.py --port 5001
```

### Database Issues
- Remove `data/app.db` to reset
- App will recreate on next run

## License

MIT License - feel free to use this project for learning and production

## Support

For issues, create a GitHub issue or contact via email.

---

**Last Updated**: February 8, 2026
**Model Version**: LightGBM v1.0 (30 features, 88.7% R²)