
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------------
# 1. Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Health Insurance Premium Estimator",
    page_icon="💰",
    layout="wide"
)

# -----------------------------------
# 2. Application Header
# -----------------------------------
st.title("💰 Health Insurance Premium Estimator")

st.write(
    "Enter your details below to estimate your "
    "health insurance premium using a machine learning model."
)

st.divider()

# -----------------------------------
# 3. Load Saved Model Bundle
# -----------------------------------
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
    st.error(f"Unable to load the model: {e}")
    st.stop()

# -----------------------------------
# 4. Input Form
# -----------------------------------
st.subheader("Enter Your Details")

st.caption(
    "Complete all fields and click Estimate Premium."
)

inputs = {}

with st.form("insurance_form"):

    columns = st.columns(2)

    for i, feature in enumerate(features):

        feature_name = str(feature)
        key = feature_name.lower().strip()

        with columns[i % 2]:

            # Sex
            if key == "sex":

                inputs[feature_name] = st.selectbox(
                    "Sex",
                    options=[0, 1],
                    format_func=lambda x: {
                        0: "Female",
                        1: "Male"
                    }[x]
                )

            # Smoking status
            elif key == "smoker":

                inputs[feature_name] = st.selectbox(
                    "Smoking Status",
                    options=[0, 1],
                    format_func=lambda x: {
                        0: "Non-smoker",
                        1: "Smoker"
                    }[x]
                )

            # One-hot encoded region columns
            elif key.startswith("region_"):

                inputs[feature_name] = st.selectbox(
                    key.replace("_", " ").title(),
                    options=[0, 1]
                )

            # Large numeric values
            elif key in (
                "claim_amount",
                "annual_salary",
                "hospital_expenditure"
            ):

                inputs[feature_name] = st.number_input(
                    key.replace("_", " ").title(),
                    min_value=0.0,
                    value=0.0,
                    step=0.01,
                    format="%.5f",
                    key=f"input_{key}"
                )

            # Other numeric fields
            else:

                inputs[feature_name] = st.number_input(
                    key.replace("_", " ").title(),
                    min_value=0.0,
                    value=0.0,
                    step=1.0,
                    key=f"input_{key}"
                )

    submitted = st.form_submit_button(
        "💰 Estimate Premium",
        type="primary",
        use_container_width=True
    )

# -----------------------------------
# 5. Predict Insurance Premium
# -----------------------------------
if submitted:

    try:
        # Arrange inputs in the saved model's feature order
        input_df = pd.DataFrame(
            [inputs],
            columns=features
        )

        # Convert input values to numeric format
        input_df = input_df.astype(float)

        # Apply the existing fitted scaler
        scaled_input = scaler.transform(input_df)

        # Predict using the saved model
        prediction = float(
            model.predict(scaled_input)[0]
        )

        # Display prediction
        st.divider()
        st.subheader("Your Estimated Premium")

        st.metric(
            label="Estimated Insurance Premium",
            value=f"₹{prediction:,.2f}"
        )

        st.caption(
            "This is a machine learning estimate, "
            "not an official insurance quotation."
        )

        # Show entered values
        with st.expander("View Your Input Details"):

            st.dataframe(
                input_df,
                use_container_width=True
            )

    except Exception as e:

        st.error(
            f"Unable to calculate the premium: {e}"
        )

# -----------------------------------
# 6. Footer
# -----------------------------------
st.divider()

st.caption(
    "Health Insurance Premium Estimator | "
    "Machine Learning Project"
)

# -----------------------------------
# 7. Model Feature Information
# -----------------------------------
with st.expander("View Model Features"):

    st.write(features)
