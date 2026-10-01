"""
train_model.py
--------------
Trains a Random Forest Regressor on the electricity bill dataset,
evaluates it, and saves the model + feature list to disk.

Usage:
    python train_model.py
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), "electricity_bill_prediction.csv")
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH  = os.path.join(MODEL_DIR, "electricity_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "features.pkl")

FEATURES = [
    "household_size",
    "num_rooms",
    "ac_units",
    "fans",
    "refrigerators",
    "washing_machines",
    "tv_units",
    "computers",
    "monthly_ac_hours",
    "monthly_other_appliance_hours",
    "units_consumed_kwh",
]
TARGET = "electricity_bill_inr"

os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[INFO] Missing values:\n{df.isnull().sum()}")
    return df


def feature_engineering(df):
    """Add derived features that improve prediction."""
    df = df.copy()
    df["total_appliances"] = (
        df["ac_units"] + df["fans"] + df["refrigerators"] +
        df["washing_machines"] + df["tv_units"] + df["computers"]
    )
    df["appliances_per_room"] = df["total_appliances"] / (df["num_rooms"] + 1)
    df["ac_intensity"] = df["ac_units"] * df["monthly_ac_hours"]
    return df


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    r2    = r2_score(y_test, preds)
    cv    = cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()
    print(f"  [{name}]  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}  CV-R²={cv:.4f}")
    return {"model": model, "mae": mae, "rmse": rmse, "r2": r2, "preds": preds}


def plot_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        scores = model.feature_importances_
        title  = f"Feature Importances — {type(model).__name__}"
    elif hasattr(model, "coef_"):
        scores = np.abs(model.coef_)
        title  = f"Feature Coefficients (abs) — {type(model).__name__}"
    else:
        print("[INFO] Model has no feature importance attribute; skipping plot.")
        return

    importances = pd.Series(scores, index=feature_names)
    importances = importances.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances.values, y=importances.index,
                hue=importances.index, palette="viridis", legend=False)
    plt.title(title)
    plt.xlabel("Importance Score")
    plt.tight_layout()
    save_path = os.path.join(MODEL_DIR, "feature_importance.png")
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[INFO] Feature importance plot saved -> {save_path}")


def plot_actual_vs_predicted(y_test, preds):
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, preds, alpha=0.5, edgecolors="steelblue", linewidths=0.5)
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    plt.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
    plt.xlabel("Actual Bill (INR)")
    plt.ylabel("Predicted Bill (INR)")
    plt.title("Actual vs Predicted — Electricity Bill")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(MODEL_DIR, "actual_vs_predicted.png")
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[INFO] Actual vs Predicted plot saved -> {save_path}")


def main():
    df = load_data()
    df = feature_engineering(df)

    # Extended feature set after engineering
    all_features = FEATURES + ["total_appliances", "appliances_per_room", "ac_intensity"]

    X = df[all_features]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print("\n[INFO] Comparing models …")
    candidates = {
        "Linear Regression":    LinearRegression(),
        "Gradient Boosting":    GradientBoostingRegressor(n_estimators=200, random_state=42),
        "Random Forest":        RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    }

    best_name, best_result = None, {"r2": -999}
    for name, mdl in candidates.items():
        result = evaluate_model(name, mdl, X_train_sc, X_test_sc, y_train, y_test)
        if result["r2"] > best_result["r2"]:
            best_name, best_result = name, result

    print(f"\n[INFO] Best model: {best_name}  (R²={best_result['r2']:.4f})")

    best_model = best_result["model"]
    plot_feature_importance(best_model, all_features)
    plot_actual_vs_predicted(y_test.values, best_result["preds"])

    # Persist artifacts
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler,     SCALER_PATH)
    joblib.dump(all_features, FEATURES_PATH)
    print(f"\n[INFO] Model saved  -> {MODEL_PATH}")
    print(f"[INFO] Scaler saved -> {SCALER_PATH}")
    print("[INFO] Training complete!")


if __name__ == "__main__":
    main()
