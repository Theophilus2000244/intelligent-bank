from flask import Flask, jsonify, request
import os
import csv
from statistics import mean
import joblib
import pandas as pd
import numpy as np
import flask

# Import custom pipeline classes
from Feature_Engineering import FraudFeatureEngineer
from Feature_selection import ColumnSelector

app = Flask(__name__)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
CSV_FILENAME = "df_target_null.csv"                # transaction data for browsing
FEATURE_ENG_PATH = "feature_engineering.pkl"        # feature engineering pipeline
FEATURE_SEL_PATH = "feature_selection.pkl"          # feature selection pipeline
MODEL_PATH = "xgb_model_with_threshold.pkl"     # model + threshold dict
MODEL_FEATURES_PATH = "model_feature_names.pkl"     # list of feature names expected by model
# ----------------------------------------------------------------------

@app.before_request
def log_request():
    try:
        data = request.get_data(as_text=True)
    except Exception:
        data = '<unreadable body>'
    print(f"[REQUEST] {request.method} {request.path} BODY={data}")
    return None


def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, CSV_FILENAME)
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            item = {}
            for k, v in r.items():
                if v is None:
                    item[k] = None
                    continue
                v = v.strip()
                if v == '':
                    item[k] = None
                    continue
                try:
                    if '.' in v:
                        item[k] = float(v)
                    else:
                        item[k] = int(v)
                except Exception:
                    item[k] = v
            rows.append(item)
    return rows


DB = load_data()

# ----------------------------------------------------------------------
# Load ML artefacts
# ----------------------------------------------------------------------
feature_engineering = None
feature_selection = None
model_package = None
model_features = None

try:
    if os.path.exists(FEATURE_ENG_PATH):
        feature_engineering = joblib.load(FEATURE_ENG_PATH)
    else:
        print(f"WARNING: {FEATURE_ENG_PATH} not found")
except Exception as e:
    print(f"Error loading feature engineering pipeline: {e}")

try:
    if os.path.exists(FEATURE_SEL_PATH):
        feature_selection = joblib.load(FEATURE_SEL_PATH)
    else:
        print(f"WARNING: {FEATURE_SEL_PATH} not found")
except Exception as e:
    print(f"Error loading feature selection pipeline: {e}")

try:
    if os.path.exists(MODEL_PATH):
        model_package = joblib.load(MODEL_PATH)
    else:
        print(f"WARNING: {MODEL_PATH} not found")
except Exception as e:
    print(f"Error loading model package: {e}")

try:
    if os.path.exists(MODEL_FEATURES_PATH):
        model_features = joblib.load(MODEL_FEATURES_PATH)
        print(f"Model feature names loaded, count: {len(model_features)}")
    else:
        print(f"WARNING: {MODEL_FEATURES_PATH} not found")
except Exception as e:
    print(f"Error loading model features: {e}")
# ----------------------------------------------------------------------


@app.route('/')
def index():
    return jsonify({
        "app": "Fraud Detection API",
        "endpoints": ["/transactions", "/transactions/<id>", "/stats", "/predict (POST)"]
    })


@app.route('/routes')
def list_routes():
    routes = [str(r) for r in app.url_map.iter_rules()]
    return jsonify({"routes": routes})


@app.route('/transactions')
def list_transactions():
    limit = request.args.get('limit', type=int)
    results = DB
    if limit is not None:
        results = results[:limit]
    return jsonify(results)


@app.route('/transactions/<int:idx>')
def get_transaction(idx):
    if idx < 0 or idx >= len(DB):
        return jsonify({"error": "not found"}), 404
    return jsonify(DB[idx])


@app.route('/stats')
def stats():
    if not DB:
        return jsonify({})
    numeric_means = {}
    for k in DB[0].keys():
        colvals = [r.get(k) for r in DB if isinstance(r.get(k), (int, float))]
        if colvals:
            numeric_means[k] = mean(colvals)
    return jsonify({"count": len(DB), "numeric_means": numeric_means})


@app.route('/predict', methods=['POST'])
def predict():
    """Accept raw transaction data, apply feature engineering, align columns, select, predict."""
    # Check required artefacts
    if feature_engineering is None:
        return jsonify({"error": "Feature engineering pipeline not loaded"}), 503
    if feature_selection is None:
        return jsonify({"error": "Feature selection pipeline not loaded"}), 503
    if model_package is None:
        return jsonify({"error": "Model not loaded"}), 503
    if model_features is None:
        return jsonify({"error": "Model feature names not loaded"}), 503

    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Invalid JSON or empty body"}), 400

    data = payload.get("features") or payload.get("data")
    if data is None:
        return jsonify({"error": "Missing 'features' or 'data' key"}), 400

    # Convert to DataFrame
    try:
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            return jsonify({"error": "Features must be a dict or list of dicts"}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to convert input to DataFrame: {str(e)}"}), 400

    # 1. Apply feature engineering
    try:
        X_eng = feature_engineering.transform(df)
    except Exception as e:
        return jsonify({"error": f"Feature engineering failed: {str(e)}"}), 500

    # 2. Ensure all columns expected by the model are present (add missing with 0)
    for col in model_features:
        if col not in X_eng.columns:
            X_eng[col] = 0

    # 3. Reorder columns to match the model's expected order
    X_eng = X_eng[model_features]

    # 4. Apply feature selection (now all columns are present, so it will keep them)
    try:
        X_sel = feature_selection.transform(X_eng)
    except Exception as e:
        return jsonify({"error": f"Feature selection failed: {str(e)}"}), 500

    # 5. Predict with XGBoost
    try:
        model = model_package['model']
        threshold = model_package.get('threshold', 0.5)
        proba = model.predict_proba(X_sel)
        fraud_proba = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        predictions = (fraud_proba >= threshold).astype(int)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # 6. Build response
    if len(df) == 1:
        return jsonify({
            "prediction": int(predictions[0]),
            "fraud_probability": float(fraud_proba[0]),
            "threshold_used": float(threshold)
        })
    else:
        return jsonify([
            {
                "prediction": int(p),
                "fraud_probability": float(fp),
                "threshold_used": float(threshold)
            }
            for p, fp in zip(predictions, fraud_proba)
        ])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)