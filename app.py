from flask import Flask, render_template, request, redirect, session, flash, jsonify, url_for
import sqlite3, pickle, pandas as pd, os, json
from functools import wraps

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = 'replace_this_with_a_random_secret'

MODEL_FILE = 'linear_regression_model.pkl'
FEATURES_FILE = 'model_features.pkl'
DATA_CSV = os.path.join('data', 'house_data.csv')
DB_FILE = 'users.db'

# Load model if available
if os.path.exists(MODEL_FILE) and os.path.exists(FEATURES_FILE):
    model = pickle.load(open(MODEL_FILE, 'rb'))
    features = pickle.load(open(FEATURES_FILE, 'rb'))
else:
    model = None
    features = None

def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

# ensure users and predictions tables exist
with get_db() as conn:
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        input_json TEXT,
        predicted_price REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()

@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username','').strip()
        password = request.form.get('password','').strip()
        if not username or not password:
            flash('All fields required','danger'); return render_template('signup.html')
        try:
            conn = get_db()
            conn.execute('INSERT INTO users (username,password) VALUES (?,?)', (username,password))
            conn.commit()
            flash('Account created, please login','success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists','danger')
    return render_template('signup.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username','').strip()
        password = request.form.get('password','').strip()
        conn = get_db(); cur = conn.cursor()
        cur.execute('SELECT * FROM users WHERE username=? AND password=?', (username,password))
        user = cur.fetchone(); conn.close()
        if user:
            session['username'] = username
            flash('Logged in','success')
            return redirect(url_for('dashboard'))
        flash('Invalid credentials','danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear(); flash('Logged out','info'); return redirect(url_for('login'))

def login_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrapped

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', username=session.get('username'))

@app.route('/predict', methods=['GET','POST'])
@login_required
def predict():
    if model is None or features is None:
        flash('Model not found. Run train_model.py to create model files.','danger')
        return render_template('predict.html', features=[], predicted_price=None)
    if request.method == 'POST':
        try:
            vals = []
            input_json = {}
            for f in features:
                v = request.form.get(f, '0')
                try:
                    val = float(v) if v!='' else 0.0
                except:
                    val = 0.0
                vals.append(val); input_json[f]=val
            df = pd.DataFrame([vals], columns=features)
            pred = float(model.predict(df)[0])
            # save prediction
            conn = get_db(); cur = conn.cursor()
            cur.execute('INSERT INTO predictions (username,input_json,predicted_price) VALUES (?,?,?)',
                        (session.get('username'), json.dumps(input_json), pred))
            conn.commit(); conn.close()
            return render_template('predict.html', features=features, predicted_price=round(pred,2))
        except Exception as e:
            return render_template('predict.html', features=features, error=str(e), predicted_price=None)
    return render_template('predict.html', features=features, predicted_price=None)

@app.route('/history')
@login_required
def history():
    conn = get_db(); cur = conn.cursor()
    cur.execute('SELECT * FROM predictions WHERE username=? ORDER BY timestamp DESC', (session.get('username'),))
    rows = cur.fetchall(); conn.close()
    return render_template('history.html', rows=rows)

@app.route('/visuals')
@login_required
def visuals():
    if not os.path.exists(DATA_CSV):
        flash('CSV data file not found at data/house_data.csv', 'danger')
        return render_template('visuals.html', price_list=[], bedrooms=[], avg_price_by_bed=[], area_pairs=[])
    df = pd.read_csv(DATA_CSV)
    # ensure expected columns exist (use user's original names)
    # prepare arrays
    price_list = df['Price'].tolist() if 'Price' in df.columns else []
    bedrooms = sorted(df['number of bedrooms'].unique().tolist()) if 'number of bedrooms' in df.columns else []
    avg_price_by_bed = df.groupby('number of bedrooms')['Price'].mean().tolist() if 'number of bedrooms' in df.columns and 'Price' in df.columns else []
    area_pairs = []
    if 'living area' in df.columns and 'Price' in df.columns:
        area_pairs = [{'x':float(a),'y':float(p)} for a,p in zip(df['living area'].tolist(), df['Price'].tolist())]
    return render_template('visuals.html', price_list=price_list, bedrooms=bedrooms, avg_price_by_bed=avg_price_by_bed, area_pairs=area_pairs)

@app.route('/api/price-trend')
@login_required
def api_price_trend():
    if not os.path.exists(DATA_CSV): return jsonify({})
    df = pd.read_csv(DATA_CSV)
    if 'number of bedrooms' in df.columns and 'Price' in df.columns:
        avg = df.groupby('number of bedrooms')['Price'].mean()
        return {'labels': list(map(str, list(avg.index))), 'values': list(avg.values)}
    return {'labels':[], 'values':[]}

if __name__ == '__main__':
    app.run(debug=True)
