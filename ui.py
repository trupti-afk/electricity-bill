"""
ui.py — Streamlit Frontend for Electricity Bill Prediction
----------------------------------------------------------
Connects to the Flask backend API (app.py) to generate predictions.
Can also run in OFFLINE mode (loads model directly) if Flask is not running.

Run:
    streamlit run ui.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── Config ────────────────────────────────────────────────────────────────────
API_URL      = "http://localhost:5000"
BASE_DIR     = os.path.dirname(__file__)
MODEL_PATH   = os.path.join(BASE_DIR, "models", "electricity_model.pkl")
SCALER_PATH  = os.path.join(BASE_DIR, "models", "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "features.pkl")
DATA_PATH    = os.path.join(BASE_DIR, "electricity_bill_prediction.csv")

st.set_page_config(
    page_title="⚡ Electricity Bill Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700;
        color: #1e3a5f; text-align: center;
        padding: 0.5rem 0 0.2rem;
    }
    .sub-header {
        font-size: 1rem; color: #57606a;
        text-align: center; margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f0f6ff; border-radius: 10px;
        padding: 1rem 1.5rem; text-align: center;
        border-left: 4px solid #3b82d4;
    }
    .metric-value {
        font-size: 2rem; font-weight: 700; color: #1e3a5f;
    }
    .metric-label { font-size: 0.85rem; color: #57606a; }
    .info-box {
        background: #fffbea; border-left: 4px solid #f5a623;
        padding: 0.8rem 1rem; border-radius: 6px;
        font-size: 0.9rem; color: #444;
    }
</style>
""", unsafe_allow_html=True)


# ── Offline helper ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_offline():
    if os.path.exists(MODEL_PATH):
        model    = joblib.load(MODEL_PATH)
        scaler   = joblib.load(SCALER_PATH)
        features = joblib.load(FEATURES_PATH)
        return model, scaler, features
    return None, None, None


def predict_offline(payload):
    model, scaler, features = load_model_offline()
    if model is None:
        return None, "Model not found. Please run train_model.py first."

    base_features = [
        "household_size", "num_rooms", "ac_units", "fans",
        "refrigerators", "washing_machines", "tv_units", "computers",
        "monthly_ac_hours", "monthly_other_appliance_hours", "units_consumed_kwh",
    ]
    values = {k: float(payload[k]) for k in base_features}
    total_appliances    = (values["ac_units"] + values["fans"] +
                           values["refrigerators"] + values["washing_machines"] +
                           values["tv_units"] + values["computers"])
    appliances_per_room = total_appliances / (values["num_rooms"] + 1)
    ac_intensity        = values["ac_units"] * values["monthly_ac_hours"]

    row = [values[f] for f in base_features] + [
        total_appliances, appliances_per_room, ac_intensity
    ]
    X_sc = scaler.transform([row])
    pred = float(model.predict(X_sc)[0])
    return {
        "predicted_bill_inr": round(pred, 2),
        "lower_bound_inr":    round(pred * 0.90, 2),
        "upper_bound_inr":    round(pred * 1.10, 2),
    }, None


def call_api(payload):
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return predict_offline(payload)
    except Exception as exc:
        return None, str(exc)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg",
             width=80)
    st.markdown("## ⚡ Bill Predictor")
    st.markdown("Fill in your household details to estimate your monthly electricity bill.")
    st.markdown("---")

    st.markdown("### 🏠 Household Info")
    household_size = st.slider("Household Size (members)", 1, 15, 4)
    num_rooms      = st.slider("Number of Rooms", 1, 10, 3)

    st.markdown("### 🔌 Appliances")
    ac_units        = st.slider("AC Units",          0, 6, 1)
    fans            = st.slider("Fans",               0, 10, 4)
    refrigerators   = st.slider("Refrigerators",     0, 4,  1)
    washing_machines = st.slider("Washing Machines", 0, 4,  1)
    tv_units        = st.slider("TV Units",           0, 6,  2)
    computers       = st.slider("Computers/Laptops", 0, 6,  1)

    st.markdown("### ⏱️ Usage Hours / Month")
    monthly_ac_hours            = st.slider("AC Hours/Month",              0, 720, 150)
    monthly_other_appliance_hours = st.slider("Other Appliance Hours/Month", 0, 720, 300)
    units_consumed_kwh          = st.number_input("Units Consumed (kWh)", min_value=0.0,
                                                   max_value=2000.0, value=280.0, step=10.0)

    predict_btn = st.button("⚡ Predict My Bill", use_container_width=True, type="primary")

# ── Main page ──────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">⚡ Electricity Bill Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered prediction using household appliance usage data</div>', unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_predict, tab_eda, tab_about = st.tabs(["🔮 Prediction", "📊 Data Insights", "ℹ️ About"])

# ─────────────────────── TAB 1: Prediction ────────────────────────────────────
with tab_predict:
    if predict_btn:
        payload = {
            "household_size": household_size,
            "num_rooms": num_rooms,
            "ac_units": ac_units,
            "fans": fans,
            "refrigerators": refrigerators,
            "washing_machines": washing_machines,
            "tv_units": tv_units,
            "computers": computers,
            "monthly_ac_hours": monthly_ac_hours,
            "monthly_other_appliance_hours": monthly_other_appliance_hours,
            "units_consumed_kwh": units_consumed_kwh,
        }

        with st.spinner("Predicting…"):
            result, error = call_api(payload)

        if error:
            st.error(f"❌ {error}")
        else:
            bill = result["predicted_bill_inr"]
            low  = result["lower_bound_inr"]
            high = result["upper_bound_inr"]

            st.success("✅ Prediction successful!")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Estimated Bill</div>
                    <div class="metric-value">₹{bill:,.0f}</div>
                    <div class="metric-label">per month</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color:#22c55e;">
                    <div class="metric-label">Lower Estimate</div>
                    <div class="metric-value" style="color:#15803d;">₹{low:,.0f}</div>
                    <div class="metric-label">-10% range</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color:#ef4444;">
                    <div class="metric-label">Upper Estimate</div>
                    <div class="metric-value" style="color:#dc2626;">₹{high:,.0f}</div>
                    <div class="metric-label">+10% range</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=bill,
                title={"text": "Monthly Bill (INR)", "font": {"size": 18}},
                delta={"reference": 3000, "increasing": {"color": "red"},
                       "decreasing": {"color": "green"}},
                gauge={
                    "axis": {"range": [0, 10000]},
                    "bar": {"color": "#3b82d4"},
                    "steps": [
                        {"range": [0, 2000],    "color": "#d1fae5"},
                        {"range": [2000, 5000],  "color": "#fef9c3"},
                        {"range": [5000, 10000], "color": "#fee2e2"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 3},
                        "thickness": 0.75, "value": 5000,
                    },
                },
            ))
            fig_gauge.update_layout(height=280, margin=dict(t=40, b=0, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Appliance breakdown bar
            appliance_data = {
                "AC":          ac_units * monthly_ac_hours * 1.5,
                "Fans":        fans * monthly_other_appliance_hours * 0.075,
                "Refrigerator": refrigerators * 720 * 0.15,
                "Washing M.":  washing_machines * 16 * 0.5,
                "TV":          tv_units * monthly_other_appliance_hours * 0.1,
                "Computer":    computers * monthly_other_appliance_hours * 0.05,
            }
            fig_bar = px.bar(
                x=list(appliance_data.keys()),
                y=list(appliance_data.values()),
                labels={"x": "Appliance", "y": "Est. Units (kWh)"},
                title="Estimated Energy Consumption by Appliance",
                color=list(appliance_data.values()),
                color_continuous_scale="Blues",
            )
            fig_bar.update_layout(showlegend=False, height=320,
                                  margin=dict(t=50, b=30, l=30, r=20))
            st.plotly_chart(fig_bar, use_container_width=True)

    else:
        st.markdown("""
        <div class="info-box">
            👈  Adjust your household details in the <strong>sidebar</strong> and
            click <strong>⚡ Predict My Bill</strong> to see your prediction.
        </div>""", unsafe_allow_html=True)

        # Show sample dataset preview
        if os.path.exists(DATA_PATH):
            df_preview = pd.read_csv(DATA_PATH).head(8)
            st.markdown("#### 📋 Sample Data Preview")
            st.dataframe(df_preview, use_container_width=True)


# ─────────────────────── TAB 2: EDA ────────────────────────────────────────────
with tab_eda:
    if not os.path.exists(DATA_PATH):
        st.warning("Dataset not found.")
    else:
        df = pd.read_csv(DATA_PATH)
        st.markdown("### 📊 Dataset Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{len(df):,}")
        c2.metric("Features", str(df.shape[1] - 1))
        c3.metric("Avg Bill (INR)", f"₹{df['electricity_bill_inr'].mean():,.0f}")
        c4.metric("Avg kWh", f"{df['units_consumed_kwh'].mean():.1f}")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            fig_hist = px.histogram(
                df, x="electricity_bill_inr", nbins=40,
                title="Distribution of Electricity Bills",
                color_discrete_sequence=["#3b82d4"],
                labels={"electricity_bill_inr": "Bill (INR)"},
            )
            fig_hist.update_layout(height=320, margin=dict(t=50, b=30))
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            fig_scatter = px.scatter(
                df, x="units_consumed_kwh", y="electricity_bill_inr",
                color="ac_units", size="household_size",
                title="Units Consumed vs Bill Amount",
                labels={
                    "units_consumed_kwh": "Units (kWh)",
                    "electricity_bill_inr": "Bill (INR)",
                    "ac_units": "AC Units",
                },
                color_continuous_scale="Viridis",
            )
            fig_scatter.update_layout(height=320, margin=dict(t=50, b=30))
            st.plotly_chart(fig_scatter, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig_box = px.box(
                df, x="ac_units", y="electricity_bill_inr",
                title="Bill Distribution by Number of ACs",
                labels={"ac_units": "AC Units", "electricity_bill_inr": "Bill (INR)"},
                color="ac_units", color_discrete_sequence=px.colors.sequential.Blues,
            )
            fig_box.update_layout(height=320, margin=dict(t=50, b=30), showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

        with col4:
            corr_cols = [
                "household_size", "ac_units", "fans", "monthly_ac_hours",
                "units_consumed_kwh", "electricity_bill_inr"
            ]
            corr_matrix = df[corr_cols].corr()
            fig_heatmap = px.imshow(
                corr_matrix, text_auto=".2f", aspect="auto",
                title="Feature Correlation Heatmap",
                color_continuous_scale="RdBu_r",
            )
            fig_heatmap.update_layout(height=360, margin=dict(t=50, b=20))
            st.plotly_chart(fig_heatmap, use_container_width=True)

        st.markdown("#### 📈 Descriptive Statistics")
        st.dataframe(df.describe().round(2), use_container_width=True)


# ─────────────────────── TAB 3: About ──────────────────────────────────────────
with tab_about:
    st.markdown("""
    ## ⚡ Electricity Bill Prediction System

    This application predicts **monthly electricity bills** for Indian households
    based on appliance inventory and usage patterns.

    ### 🔬 Model Details
    | Item | Detail |
    |------|--------|
    | Algorithm | Random Forest / Gradient Boosting (best selected) |
    | Dataset | 1,002 household records |
    | Target | `electricity_bill_inr` (INR/month) |
    | Features | 11 raw + 3 engineered = 14 total |
    | Validation | 5-fold cross-validation |

    ### 🏗️ Architecture
    ```
    ┌─────────────────┐     HTTP/JSON     ┌──────────────────┐
    │  Streamlit UI   │ ───────────────▶  │  Flask REST API  │
    │   (ui.py)       │ ◀───────────────  │   (app.py)       │
    └─────────────────┘                   └────────┬─────────┘
                                                   │
                                         ┌─────────▼─────────┐
                                         │  Trained ML Model  │
                                         │  (RandomForest /   │
                                         │   GradBoost)       │
                                         └───────────────────┘
    ```

    ### 📁 Project Structure
    ```
    electricity_project/
    ├── electricity_bill_prediction.csv   # Dataset
    ├── train_model.py                    # Model training
    ├── app.py                            # Flask API
    ├── ui.py                             # Streamlit frontend
    ├── requirements.txt
    ├── README.md
    └── models/
        ├── electricity_model.pkl
        ├── scaler.pkl
        └── features.pkl
    ```

    ### 👤 How to Use
    1. Use the **sidebar sliders** to enter your household details
    2. Click **⚡ Predict My Bill**
    3. View the **predicted bill**, confidence range, and appliance breakdown
    4. Explore the **Data Insights** tab for EDA visualisations
    """)
