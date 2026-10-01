# ⚡ Electricity Bill Prediction System

An end-to-end Machine Learning application that predicts monthly household electricity bills (INR) using appliance inventory and usage data.

---

## 📁 Project Structure

```
electricity_project/
├── electricity_bill_prediction.csv   ← Dataset (1,002 records)
├── train_model.py                    ← Model training script
├── app.py                            ← Flask REST API (backend)
├── ui.py                             ← Streamlit UI (frontend)
├── requirements.txt                  ← Python dependencies
├── README.md
└── models/                           ← Auto-created on training
    ├── electricity_model.pkl
    ├── scaler.pkl
    ├── features.pkl
    ├── feature_importance.png
    └── actual_vs_predicted.png
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
cd electricity_project
python train_model.py
```

This will:
- Load and preprocess the dataset
- Compare Linear Regression, Gradient Boosting, and Random Forest
- Save the best model to `models/electricity_model.pkl`
- Generate evaluation plots in `models/`

### 3. Start the Flask Backend API

```bash
python app.py
```

The API will be available at `http://localhost:5000`

### 4. Launch the Streamlit Frontend

Open a **new terminal** and run:

```bash
streamlit run ui.py
```

The UI will open at `http://localhost:8501`

> **Tip:** The Streamlit UI includes an **offline fallback** — it loads the model directly if the Flask server is not running.

---

## 🔌 API Reference

### `GET /health`
Returns service status and whether the model is loaded.

```json
{ "status": "ok", "model_loaded": true }
```

### `GET /model-info`
Returns model type and feature list.

### `GET /sample-input`
Returns an example JSON body for `/predict`.

### `POST /predict`
**Request body:**

```json
{
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
  "units_consumed_kwh": 280.0
}
```

**Response:**

```json
{
  "predicted_bill_inr": 2845.50,
  "lower_bound_inr": 2561.00,
  "upper_bound_inr": 3130.00,
  "units_consumed_kwh": 280.0,
  "input_summary": {
    "household_size": 4,
    "total_appliances": 9,
    "ac_intensity_hours": 150.0
  }
}
```

---

## 🤖 Model Details

| Property       | Value                              |
|----------------|------------------------------------|
| Algorithm      | Random Forest / Gradient Boosting  |
| Training rows  | ~800 (80% split)                   |
| Test rows      | ~200 (20% split)                   |
| Validation     | 5-fold cross-validation            |
| Features       | 14 (11 raw + 3 engineered)         |
| Target         | `electricity_bill_inr` (INR)       |

### Engineered Features
| Feature                | Formula                               |
|------------------------|---------------------------------------|
| `total_appliances`     | sum of all appliance counts           |
| `appliances_per_room`  | total_appliances / (num_rooms + 1)    |
| `ac_intensity`         | ac_units × monthly_ac_hours           |

---

## 📊 Frontend Tabs

| Tab              | Content                                        |
|------------------|------------------------------------------------|
| 🔮 Prediction    | Input sliders → bill prediction + gauge chart  |
| 📊 Data Insights | EDA: histograms, scatter, box plots, heatmap   |
| ℹ️ About         | Architecture, model details, usage guide       |

---

## 🛠️ Tech Stack

| Layer     | Technology              |
|-----------|-------------------------|
| ML        | scikit-learn, joblib    |
| Backend   | Flask 3.x               |
| Frontend  | Streamlit + Plotly      |
| Data      | Pandas, NumPy           |
| Charts    | Plotly Express          |

---

## 📝 License

For educational / demonstration purposes.
