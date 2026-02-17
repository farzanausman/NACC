import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="NAAC CGPA Prediction", layout="wide")

st.title("🏫 NAAC CGPA Prediction System")
st.write("Enter your institution details in the sidebar and click Predict")

# -----------------------------
# Sample default values for your features
# -----------------------------
# Replace these keys with your actual feature names
sample_defaults = {
    "student_strength": 2000.0,
    "faculty_count": 150.0,
    "research_publications": 50.0,
    "placement_percentage": 80.0,
    "infrastructure_score": 75.0,
    "student_faculty_ratio": 13.3
    # Add more features from your dataset here
}

# -----------------------------
# Load model, scaler, and feature names
# -----------------------------
try:
    model = joblib.load("naac_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Institution Parameters")

inputs = []
for feature in feature_names:
    default_value = sample_defaults.get(feature, 10.0)

    # Handle mixed numeric types
    if isinstance(default_value, int):
        value = st.sidebar.number_input(f"{feature}", min_value=0, value=int(default_value))
    else:
        value = st.sidebar.number_input(f"{feature}", min_value=0.0, value=float(default_value))

    inputs.append(value)

# -----------------------------
# Optional Button: Fill Sample Data
# -----------------------------
if st.sidebar.button("Use Sample Data"):
    for i, feature in enumerate(feature_names):
        default_value = sample_defaults.get(feature, 10.0)
        st.session_state[feature] = default_value

# -----------------------------
# Predict Button
# -----------------------------
if st.sidebar.button("Predict NAAC CGPA"):
    try:
        # Convert inputs to array and scale
        input_array = np.array(inputs).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        # Predict CGPA
        prediction = model.predict(input_scaled)[0]

        # NAAC grade function
        def naac_grade(cgpa):
            if cgpa >= 3.51:
                return "A++"
            elif cgpa >= 3.26:
                return "A+"
            elif cgpa >= 3.01:
                return "A"
            elif cgpa >= 2.76:
                return "B++"
            elif cgpa >= 2.51:
                return "B+"
            elif cgpa >= 2.01:
                return "B"
            else:
                return "C"

        grade = naac_grade(prediction)

        # -----------------------------
        # Show Results in Colored Boxes
        # -----------------------------
        st.subheader("🏆 Prediction Result")
        cgpa_color = "green" if prediction >= 3.0 else "orange" if prediction >= 2.5 else "red"
        st.markdown(f"<h2 style='color:{cgpa_color}'>Predicted NAAC CGPA: {round(prediction, 2)}</h2>", unsafe_allow_html=True)
        st.markdown(f"<h3>Predicted NAAC Grade: {grade}</h3>", unsafe_allow_html=True)

        # -----------------------------
        # Feature Importance Chart
        # -----------------------------
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=True)

        st.subheader("📊 Feature Importance")
        fig, ax = plt.subplots(figsize=(10,6))
        ax.barh(importance_df["Feature"], importance_df["Importance"], color='skyblue')
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Features")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error during prediction: {e}")





