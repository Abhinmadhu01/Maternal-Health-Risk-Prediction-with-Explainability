# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib
import json

# Load the trained multi-output model
multi_output_model = joblib.load('multi_output_model.pkl')

# Load the reasons list and convert it to a dictionary
with open('reasons.json', 'r') as f:
    reasons_list = json.load(f)
reasons_dict = {i: reason for i, reason in enumerate(reasons_list)}  # Convert list to dictionary

# Define the risk map
risk_map = {0: 'Low Risk', 1: 'Mid Risk', 2: 'High Risk'}

# Collect user inputs
st.title("Maternal Health Risk Prediction Chatbot")
st.write("Please provide the following details:")

# Define feature statistics
feature_stats = {
    "Age": {"mean": 29.87, "std": 13.47, "min": 10, "max": 70},
    "SystolicBP": {"mean": 113.19, "std": 18.40, "min": 70, "max": 160},
    "DiastolicBP": {"mean": 76.46, "std": 13.86, "min": 49, "max": 100},
    "BS": {"mean": 8.72, "std": 3.29, "min": 6, "max": 19},
    "BodyTemp": {"mean": 98.66, "std": 1.37, "min": 98, "max": 103},
    "HeartRate": {"mean": 74.30, "std": 8.08, "min": 60, "max": 90},
}

# Collect user inputs with validation
age = st.number_input(
    "Age",
    min_value=feature_stats["Age"]["min"],
    max_value=feature_stats["Age"]["max"],
    value=int(feature_stats["Age"]["mean"]),
    help=f"Typical range: {feature_stats['Age']['mean'] - 3 * feature_stats['Age']['std']:.2f} to {feature_stats['Age']['mean'] + 3 * feature_stats['Age']['std']:.2f}",
)
systolic_bp = st.number_input(
    "Systolic Blood Pressure",
    min_value=feature_stats["SystolicBP"]["min"],
    max_value=feature_stats["SystolicBP"]["max"],
    value=int(feature_stats["SystolicBP"]["mean"]),
    help=f"Typical range: {feature_stats['SystolicBP']['mean'] - 3 * feature_stats['SystolicBP']['std']:.2f} to {feature_stats['SystolicBP']['mean'] + 3 * feature_stats['SystolicBP']['std']:.2f}",
)
diastolic_bp = st.number_input(
    "Diastolic Blood Pressure",
    min_value=feature_stats["DiastolicBP"]["min"],
    max_value=feature_stats["DiastolicBP"]["max"],
    value=int(feature_stats["DiastolicBP"]["mean"]),
    help=f"Typical range: {feature_stats['DiastolicBP']['mean'] - 3 * feature_stats['DiastolicBP']['std']:.2f} to {feature_stats['DiastolicBP']['mean'] + 3 * feature_stats['DiastolicBP']['std']:.2f}",
)
bs = st.number_input(
    "Blood Sugar Level (e.g., 7.5)",
    min_value=feature_stats["BS"]["min"],
    max_value=feature_stats["BS"]["max"],
    value=int(feature_stats["BS"]["mean"]),
    help=f"Typical range: {feature_stats['BS']['mean'] - 3 * feature_stats['BS']['std']:.2f} to {feature_stats['BS']['mean'] + 3 * feature_stats['BS']['std']:.2f}",
)
body_temp = st.number_input(
    "Body Temperature (°F)",
    min_value=feature_stats["BodyTemp"]["min"],
    max_value=feature_stats["BodyTemp"]["max"],
    value=int(feature_stats["BodyTemp"]["mean"]),
    help=f"Typical range: {feature_stats['BodyTemp']['mean'] - 3 * feature_stats['BodyTemp']['std']:.2f} to {feature_stats['BodyTemp']['mean'] + 3 * feature_stats['BodyTemp']['std']:.2f}",
)
heart_rate = st.number_input(
    "Heart Rate",
    min_value=feature_stats["HeartRate"]["min"],
    max_value=feature_stats["HeartRate"]["max"],
    value=int(feature_stats["HeartRate"]["mean"]),
    help=f"Typical range: {feature_stats['HeartRate']['mean'] - 3 * feature_stats['HeartRate']['std']:.2f} to {feature_stats['HeartRate']['mean'] + 3 * feature_stats['HeartRate']['std']:.2f}",
)

if st.button("Predict Risk"):
    # Prepare input for the model
    manual_input = {
        'Age': age,
        'SystolicBP': systolic_bp,
        'DiastolicBP': diastolic_bp,
        'BS': bs,
        'BodyTemp': body_temp,
        'HeartRate': heart_rate
    }
    manual_input_df = pd.DataFrame([manual_input])

    # Ensure preprocessing matches training
    with open('columns.json', 'r') as f:
        columns = json.load(f)
    manual_input_df = pd.get_dummies(manual_input_df)
    for col in columns:
        if col not in manual_input_df.columns:
            manual_input_df[col] = 0
    manual_input_df = manual_input_df[columns]

    # Handle predictions
    prediction = multi_output_model.predict(manual_input_df)  # Generate prediction
    predicted_risk = risk_map[prediction[0][0]]  # RiskLevel
    try:
        # Decode the predicted reason using the dictionary
        predicted_reason = reasons_dict.get(prediction[0][1], "No specific reason available for this prediction.")
    except (KeyError, ValueError):
        # Fallback if the reason is not found or conversion fails
        predicted_reason = "No specific reason available for this prediction."

    # Fallback mechanism for mismatched reasons
    if predicted_risk == "High Risk" and "low risk" in predicted_reason.lower():
        predicted_reason = "High Risk - Elevated indicators detected. Please consult a healthcare provider."
    elif predicted_risk == "Mid Risk" and "low risk" in predicted_reason.lower():
        predicted_reason = "Mid Risk - Some indicators require attention. Maintain regular checkups."

    # Display the response
    st.write(f"**Predicted Risk Level:** {predicted_risk}")
    st.write(f"**Reason for Risk Level:** {predicted_reason}")
    