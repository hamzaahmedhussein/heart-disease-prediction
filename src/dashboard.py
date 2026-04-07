import os

import streamlit as st
import requests

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="heart",
    layout="centered",
)

st.title("Heart Disease Risk Predictor")
st.markdown("### Powered by a trained MLP neural network")
st.write(
    "This dashboard connects to a FastAPI backend serving a regularized "
    "deep neural network trained on the UCI Heart Disease dataset."
)

try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    if health.get("status") == "healthy":
        st.sidebar.success("API Connected")
    else:
        st.sidebar.warning("API returned unhealthy status")
except Exception:
    st.sidebar.error(f"Cannot reach API at {API_URL}")

st.sidebar.markdown("---")
st.sidebar.markdown("**About**")
st.sidebar.markdown(
    "This model uses He Normal initialization, Swish activation, "
    "Batch Normalization, and MC Dropout for uncertainty estimation."
)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50, help="Patient age in years")
    sex = st.selectbox(
        "Sex",
        options=[("Male", 1.0), ("Female", 0.0)],
        format_func=lambda x: x[0],
    )[1]
    cp = st.selectbox(
        "Chest Pain Type",
        options=[
            ("Typical Angina", 1.0),
            ("Atypical Angina", 2.0),
            ("Non-anginal Pain", 3.0),
            ("Asymptomatic", 4.0),
        ],
        format_func=lambda x: x[0],
    )[1]
    trestbps = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=50, max_value=250, value=120,
    )
    chol = st.number_input(
        "Serum Cholesterol (mg/dl)",
        min_value=100, max_value=600, value=200,
    )
    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        options=[("No", 0.0), ("Yes", 1.0)],
        format_func=lambda x: x[0],
    )[1]
    restecg = st.selectbox(
        "Resting ECG Result",
        options=[
            ("Normal", 0.0),
            ("ST-T Wave Abnormality", 1.0),
            ("Left Ventricular Hypertrophy", 2.0),
        ],
        format_func=lambda x: x[0],
    )[1]

with col2:
    thalach = st.number_input(
        "Max Heart Rate Achieved",
        min_value=50, max_value=250, value=150,
    )
    exang = st.selectbox(
        "Exercise Induced Angina",
        options=[("No", 0.0), ("Yes", 1.0)],
        format_func=lambda x: x[0],
    )[1]
    oldpeak = st.number_input(
        "ST Depression (Exercise vs Rest)",
        min_value=0.0, max_value=10.0, value=1.0, step=0.1,
    )
    slope = st.selectbox(
        "Peak Exercise ST Slope",
        options=[
            ("Upsloping", 1.0),
            ("Flat", 2.0),
            ("Downsloping", 3.0),
        ],
        format_func=lambda x: x[0],
    )[1]
    ca = st.number_input(
        "Major Vessels (Fluoroscopy)",
        min_value=0.0, max_value=4.0, value=0.0, step=1.0,
    )
    thal = st.selectbox(
        "Thalassemia",
        options=[
            ("Normal", 3.0),
            ("Fixed Defect", 6.0),
            ("Reversible Defect", 7.0),
        ],
        format_func=lambda x: x[0],
    )[1]

st.markdown("---")
if st.button("Calculate Heart Disease Risk", use_container_width=True, type="primary"):
    payload = {
        "age": float(age), "sex": float(sex), "cp": float(cp),
        "trestbps": float(trestbps), "chol": float(chol), "fbs": float(fbs),
        "restecg": float(restecg), "thalach": float(thalach),
        "exang": float(exang), "oldpeak": float(oldpeak),
        "slope": float(slope), "ca": float(ca), "thal": float(thal),
    }

    with st.spinner("Running prediction..."):
        try:
            res = requests.post(f"{API_URL}/predict", json=payload, timeout=30)

            if res.status_code == 200:
                data = res.json()

                risk = data["risk_percentage"]
                uncertainty = data["uncertainty_std"] * 100

                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("Risk Probability", f"{risk:.1f}%")
                col_r2.metric("Uncertainty (std)", f"{uncertainty:.1f}%")
                col_r3.metric("MC Samples", data["mc_samples"])

                st.markdown("---")

                if data["requires_review"]:
                    st.error(
                        "HIGH UNCERTAINTY - The model produced high variance across forward passes. "
                        "This patient may represent an edge case requiring clinical specialist review."
                    )
                else:
                    if risk >= 50:
                        st.warning(f"{data['prediction']} - Risk is elevated. Model confidence is high.")
                    else:
                        st.success(f"{data['prediction']} - Model confidence is high for a low-risk prediction.")

            elif res.status_code == 422:
                st.error(f"Input validation error: {res.json().get('detail', 'Invalid input')}")
            else:
                st.error(f"API Error: Backend returned status code {res.status_code}")

        except requests.exceptions.ConnectionError:
            st.error(f"Could not connect to the API at {API_URL}. Start it with: make serve")
        except requests.exceptions.Timeout:
            st.error("Request timed out. The model may be loading.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
