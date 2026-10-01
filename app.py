"""
app.py — Flask REST API for Electricity Bill Prediction
--------------------------------------------------------
Endpoints:
    GET  /health          → service health check
    POST /predict         → predict bill from JSON body
    GET  /model-info      → model metadata & feature list
    GET  /sample-input    → example input payload

Run:
    python app.py
"""

import os
import joblib
import numpy as np
from flask import Flask, request, jsonify

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
MODEL_PATH  = os.path.join(BASE_DIR, "models", "electricity_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "features.pkl")

app = Flask(__name__)

# ── Load artifacts once at startup ────────────────────────────────────────────
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not found. Run `python train_model.py` first."
        )
    model    = joblib.load(MODEL_PATH)
    scaler   = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, scaler, features

try:
    MODEL, SCALER, FEATURES = load_artifacts()
    MODEL_LOADED = True
except FileNotFoundError as e:
    print(f"[WARNING] {e}")
    MODEL_LOADED = False


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL_LOADED
    })


@app.route("/model-info", methods=["GET"])
def model_info():
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded"}), 503
    return jsonify({
        "model_type": type(MODEL).__name__,
        "features": FEATURES,
        "n_features": len(FEATURES),
    })


@app.route("/sample-input", methods=["GET"])
def sample_input():
    return jsonify({
        "household_size": 4,
        "num_rooms": 3,
        "ac_units": 1,
        "fans": 4,
        "refrigerators": 1,
        "washing_machines": 1,
        "tv_units": 2,
        "computers": 1,
        "monthly_ac_hours": 150,
        "monthly_other_appliance_hours": 300,
        "units_consumed_kwh": 280.0,
    })


@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    # Required base features
    base_features = [
        "household_size", "num_rooms", "ac_units", "fans",
        "refrigerators", "washing_machines", "tv_units", "computers",
        "monthly_ac_hours", "monthly_other_appliance_hours", "units_consumed_kwh",
    ]

    missing = [f for f in base_features if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        # Validate numeric types
        values = {k: float(data[k]) for k in base_features}
    except (ValueError, TypeError) as exc:
        return jsonify({"error": f"Invalid value: {exc}"}), 400

    # Feature engineering (must mirror train_model.py)
    total_appliances     = (values["ac_units"] + values["fans"] +
                            values["refrigerators"] + values["washing_machines"] +
                            values["tv_units"] + values["computers"])
    appliances_per_room  = total_appliances / (values["num_rooms"] + 1)
    ac_intensity         = values["ac_units"] * values["monthly_ac_hours"]

    row = [values[f] for f in base_features] + [
        total_appliances, appliances_per_room, ac_intensity
    ]

    X_scaled = SCALER.transform([row])
    prediction = float(MODEL.predict(X_scaled)[0])

    # Confidence interval (±10% heuristic)
    return jsonify({
        "predicted_bill_inr": round(prediction, 2),
        "lower_bound_inr":    round(prediction * 0.90, 2),
        "upper_bound_inr":    round(prediction * 1.10, 2),
        "units_consumed_kwh": values["units_consumed_kwh"],
        "input_summary": {
            "household_size": int(values["household_size"]),
            "total_appliances": int(total_appliances),
            "ac_intensity_hours": round(ac_intensity, 1),
        }
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
