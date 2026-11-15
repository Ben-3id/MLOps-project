
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Bank Marketing Predictor",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Bank Marketing Predictor")
st.write(
    "Predict whether a bank client will subscribe to a term deposit."
)


# -------------------------
# Model Management
# -------------------------

st.sidebar.subheader("Model Management")

with st.sidebar.form("model_form"):
    model_id = st.text_input("MLflow Run ID")
    model_name = st.text_input("Artifact Name", value="model")

    load_submitted = st.form_submit_button(
        "Load Model",
        use_container_width=True
    )

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

if load_submitted:
    try:
        if not model_id.strip():
            st.error("Please enter an MLflow Run ID.")
        else:
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
            st.sidebar.success("Model loaded successfully!")

    except requests.RequestException as e:
        st.session_state.model_loaded = False
        st.sidebar.error(f"Model loading failed: {e}")


if st.session_state.model_loaded:
    st.sidebar.success("Model Status: Loaded")
else:
    st.sidebar.warning("Model Status: Not Loaded")


# -------------------------
# Prediction Form
# -------------------------

st.divider()

with st.form("prediction_form"):

    # -------------------------
    # Client Profile
    # -------------------------

    st.subheader("👤 Client Profile")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35,
            step=1
        )

    with col2:
        job = st.selectbox(
            "Job",
            [
                "admin.",
                "unknown",
                "unemployed",
                "management",
                "housemaid",
                "entrepreneur",
                "student",
                "blue-collar",
                "self-employed",
                "retired",
                "technician",
                "services"
            ]
        )

    with col3:
        marital = st.selectbox(
            "Marital Status",
            ["married", "divorced", "single"]
        )

    with col4:
        education = st.selectbox(
            "Education",
            ["unknown", "secondary", "primary", "tertiary"]
        )

    # -------------------------
    # Financial Information
    # -------------------------

    st.divider()
    st.subheader("💰 Financial Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        default = st.selectbox(
            "Credit Default",
            ["no", "yes"]
        )

    with col2:
        balance = st.number_input(
            "Average Yearly Balance (€)",
            value=1000,
            step=100
        )

    with col3:
        housing = st.selectbox(
            "Housing Loan",
            ["no", "yes"]
        )

    with col4:
        loan = st.selectbox(
            "Personal Loan",
            ["no", "yes"]
        )

    # -------------------------
    # Campaign & Contact
    # -------------------------

    st.divider()
    st.subheader("📞 Campaign & Contact Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        contact = st.selectbox(
            "Contact Type",
            ["unknown", "telephone", "cellular"]
        )

    with col2:
        day = st.number_input(
            "Last Contact Day",
            min_value=1,
            max_value=31,
            value=15
        )

    with col3:
        month = st.selectbox(
            "Last Contact Month",
            [
                "jan", "feb", "mar", "apr",
                "may", "jun", "jul", "aug",
                "sep", "oct", "nov", "dec"
            ]
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        duration = st.number_input(
            "Contact Duration (Seconds)",
            min_value=0,
            value=180,
            step=10
        )

    with col2:
        campaign = st.number_input(
            "Contacts During Current Campaign",
            min_value=1,
            value=2,
            step=1
        )

    with col3:
        pdays = st.number_input(
            "Days Since Previous Contact (-1 = Never)",
            min_value=-1,
            value=-1,
            step=1
        )

    col1, col2 = st.columns(2)

    with col1:
        previous = st.number_input(
            "Previous Campaign Contacts",
            min_value=0,
            value=0,
            step=1
        )

    with col2:
        poutcome = st.selectbox(
            "Previous Campaign Outcome",
            ["unknown", "other", "failure", "success"]
        )

    # -------------------------
    # Submit
    # -------------------------

    st.divider()

    submitted = st.form_submit_button(
        "🔍 Predict Subscription",
        use_container_width=True
    )


# -------------------------
# Prediction
# -------------------------

if submitted:

    if not st.session_state.model_loaded:
        st.error("Please load an MLflow model first.")

    else:

        input_data = {
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "balance": balance,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "day": day,
            "month": month,
            "duration": duration,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "poutcome": poutcome
        }

        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=input_data,
                timeout=30
            )

            response.raise_for_status()

            result = response.json()
            prediction = result["prediction"]

            st.divider()
            st.subheader("📊 Prediction Result")

            if prediction in [1, "1", "yes", "Yes", True]:
                st.success("The client is likely to subscribe.")
                st.metric("Prediction", "Yes")
            else:
                st.info("The client is unlikely to subscribe.")
                st.metric("Prediction", "No")

            if "probability" in result:
                st.metric(
                    "Subscription Probability",
                    f"{float(result['probability']) * 100:.2f}%"
                )

            elif "probabilities" in result:
                st.write("Class Probabilities")
                st.json(result["probabilities"])

        except requests.RequestException as e:
            st.error(f"Prediction failed: {e}")

        except (KeyError, TypeError, ValueError) as e:
            st.error(f"Invalid API response: {e}")
