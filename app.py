
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Health Insurance Premium Estimator",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Health Insurance Premium Estimator")
st.write("Enter your details to estimate your insurance premium.")
st.divider()

# Load the model bundle
MODEL_PATH = Path(__file__).parent / "insurance_bundle.pkl"

@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)

try:
    bundle = load_bundle()
    model = bundle["model"]
    scaler = bundle["scaler"]
    features = list(bundle["features"])
except Exception as e:
    st.error(f"Unable to load model bundle: {e}")
    st.stop()

st.success("Insurance model loaded successfully!")
st.subheader("Enter Your Details")

inputs = {}

with st.form("insurance_form"):
    columns = st.columns(2)

    for i, feature in enumerate(features):
        f = str(feature)
        key = f.lower().strip()

        with columns[i % 2]:
            if key == "sex":
                inputs[f] = st.selectbox(
                    "Sex", [0, 1],
                    format_func=lambda x: {
                        0: "Female (0)", 1: "Male (1)"
                    }[x]
                )

            elif key == "smoker":
                inputs[f] = st.selectbox(
                    "Smoking Status", [0, 1],
                    format_func=lambda x: {
                        0: "No (0)", 1: "Yes (1)"
                    }[x]
                )

            elif key.startswith("region_"):
                inputs[f] = st.selectbox(
                    f.replace("_", " ").title(), [0, 1]
                )

            elif key in ("claim_amount", "annual_salary","hospital_expenditure"):
                inputs[f] = st.number_input(
                    f.replace("_", " ").title(),
                    min_value=0.0,
                    value=0.0,
                    step=1000.0,
                    format="%.2f"
                )

            else:
                inputs[f] = st.number_input(
                    f.replace("_", " ").title(),
                    min_value=0.0,
                    value=0.0,
                    step=1.0
                )

    submitted = st.form_submit_button(
        "Estimate Premium",
        type="primary",
        use_container_width=True
    )

if submitted:
    try:
        input_df = pd.DataFrame([inputs], columns=features)
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]

        st.divider()
        st.subheader("Prediction Result")
        st.metric(
            "Estimated Insurance Premium",
            f"{prediction:,.2f}"
        )

        with st.expander("View Your Input Details"):
            st.dataframe(input_df, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()
st.caption("Health Insurance Premium Estimator | Machine Learning")

with st.expander("View Model Features"):
    st.write(features)
