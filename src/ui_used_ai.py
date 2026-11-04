
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Performance Predictor")
st.write("Load an MLflow model and predict student performance.")


# -------------------------
# Model Management
# -------------------------

st.subheader("Model Management")

with st.form("model_form"):
    model_id = st.text_input("MLflow Run ID")
    model_name = st.text_input("Artifact Name", value="model")

    load_submitted = st.form_submit_button(
        "Load Model",
        use_container_width=True
    )

if load_submitted:
    try:
        response = requests.post(
            f"{API_URL}/get-model",
            params={
                "model_id": model_id,
                "model_name": model_name
            },
            timeout=120
        )

        response.raise_for_status()

        st.session_state.model_loaded = True
        st.success("Model loaded successfully!")

    except requests.RequestException as e:
        st.session_state.model_loaded = False
        st.error(f"Model loading failed: {e}")


# -------------------------
# Prediction
# -------------------------

st.divider()
st.subheader("Student Information")

with st.form("prediction_form"):

    hours_studied = st.number_input(
        "Hours Studied",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5
    )

    previous_scores = st.number_input(
        "Previous Scores",
        min_value=0.0,
        max_value=100.0,
        value=75.0,
        step=1.0
    )

    extracurricular = st.selectbox(
        "Extracurricular Activities",
        ["Yes", "No"]
    )

    sleep_hours = st.number_input(
        "Sleep Hours",
        min_value=0.0,
        max_value=24.0,
        value=7.0,
        step=0.5
    )

    sample_papers = st.number_input(
        "Sample Question Papers Practiced",
        min_value=0,
        max_value=100,
        value=4,
        step=1
    )

    submitted = st.form_submit_button(
        "Predict Performance Index",
        use_container_width=True
    )


if submitted:

    input_data = {
        "Hours_Studied": hours_studied,
        "Previous_Scores": previous_scores,
        "Extracurricular_Activities": extracurricular,
        "Sleep_Hours": sleep_hours,
        "Sample_Question_Papers_Practiced": sample_papers
    }

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=input_data,
            timeout=30
        )

        response.raise_for_status()

        prediction = response.json()["prediction"]

        st.success("Prediction completed successfully!")

        st.metric(
            label="Predicted Performance Index",
            value=f"{prediction:.2f}"
        )

    except requests.RequestException as e:
        st.error(f"Prediction failed: {e}")
