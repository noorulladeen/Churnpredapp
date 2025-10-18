import streamlit as st
import joblib
import numpy as np

# Load pre-trained scaler and model
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

# Optional: Add a banner or logo
# st.image("churn_banner.png", use_column_width=True)

# Sidebar info
st.sidebar.title("📘 About")
st.sidebar.info(
    "This app predicts customer churn using a trained machine learning model. "
    "Enter the customer details on the right and click Predict to see the result."
)

# App title and instructions
st.title("🔮 Churn Prediction App")
st.write("Fill in the customer details below and click **Predict** to see if they are likely to churn.")
st.divider()

# User inputs
age = st.number_input("🧓 Age", min_value=10, max_value=100, value=30)
gender = st.selectbox("⚧️ Gender", ['Male', 'Female'])
tenure = st.number_input("📅 Tenure (in months)", min_value=0, max_value=130, value=10)
monthlycharge = st.number_input("💰 Monthly Charges", min_value=30, max_value=150, value=75)

st.divider()

# Predict button
if st.button("🚀 Predict"):
    try:
        # Encode gender
        gender_encoded = 1 if gender.lower() == "female" else 0

        # Prepare input
        X = np.array([age, gender_encoded, tenure, monthlycharge]).reshape(1, -1)

        # Scale input
        X_scaled = scaler.transform(X)

        # Predict
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]

        # Display result
        result = "Churn" if prediction == 1 else "Not a Churn"
        st.success(f"🧾 Prediction: **{result}**")
        st.info(f"📊 Model Confidence: **{probability:.2%}**")

    except Exception as e:
        st.error(f"⚠️ Something went wrong: {e}")
else:
    st.write("⬅️ Enter values and click **Predict** to begin.")
