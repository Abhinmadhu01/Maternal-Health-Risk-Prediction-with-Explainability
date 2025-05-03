# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib
import json

# Load the trained multi-output model
multi_output_model = joblib.load('multi_output_model.pkl')

# Define the risk map
risk_map = {0: 'Low Risk', 1: 'Mid Risk', 2: 'High Risk'}

# Define thresholds based on the statistical analysis
thresholds = {
    'Age': {
        'high': 70,  # Max age in stats
        'low': 10    # Min age in stats
    },
    'SystolicBP': {
        'high': 140,  # Above mean + 1SD (113 + 18 = 131), using 140 as clinical threshold
        'very_high': 160,  # Max in stats
        'low': 90,    # Below mean - 1SD (113 - 18 = 95), using 90 as clinical threshold
        'very_low': 70   # Min in stats
    },
    'DiastolicBP': {
        'high': 90,   # Above mean + 1SD (76 + 14 = 90)
        'very_high': 100,  # Max in stats
        'low': 60,    # Below mean - 1SD (76 - 14 = 62), using 60 as clinical threshold
        'very_low': 49   # Min in stats
    },
    'BS': {
        'high': 12.0,  # Mean (8.72) + 1SD (3.29) = 12.01
        'very_high': 15.0,  # Approaching max (19)
        'low': 6.0,    # Min in stats
        'very_low': 4.0  # Below normal clinical range
    },
    'BodyTemp': {
        'high': 100.0,  # Mean (98.66) + 1SD (1.37) = 100.03
        'very_high': 101.0,
        'low': 97.0,    # Below normal clinical range
        'very_low': 96.0
    },
    'HeartRate': {
        'high': 82,   # Mean (74.30) + 1SD (8.08) = 82.38
        'very_high': 90,  # Max in stats
        'low': 66,    # Mean - 1SD (74.30 - 8.08 = 66.22)
        'very_low': 60   # Min in stats
    }
}

def generate_recommendations(input_data):
    recommendations = []
    
    # Age analysis
    if pd.notna(input_data['Age']):
        if input_data['Age'] > thresholds['Age']['high']:
            recommendations.append("Consider geriatric pregnancy consultation")
        elif input_data['Age'] < 18:  # Teen pregnancy threshold
            recommendations.append("Adolescent pregnancy requires specialized care")
    
    # Blood Pressure analysis
    if pd.notna(input_data['SystolicBP']):
        if input_data['SystolicBP'] >= thresholds['SystolicBP']['very_high']:
            recommendations.append("Urgent medical attention needed for systolic BP")
        elif input_data['SystolicBP'] >= thresholds['SystolicBP']['high']:
            recommendations.append("Monitor systolic BP and reduce salt intake")
        elif input_data['SystolicBP'] <= thresholds['SystolicBP']['very_low']:
            recommendations.append("Medical evaluation needed for low systolic BP")
        elif input_data['SystolicBP'] <= thresholds['SystolicBP']['low']:
            recommendations.append("Increase fluid intake for low systolic BP")
    
    if pd.notna(input_data['DiastolicBP']):
        if input_data['DiastolicBP'] >= thresholds['DiastolicBP']['very_high']:
            recommendations.append("Urgent care needed for diastolic BP")
        elif input_data['DiastolicBP'] >= thresholds['DiastolicBP']['high']:
            recommendations.append("Monitor diastolic BP and practice relaxation techniques")
        elif input_data['DiastolicBP'] <= thresholds['DiastolicBP']['very_low']:
            recommendations.append("Medical evaluation needed for low diastolic BP")
        elif input_data['DiastolicBP'] <= thresholds['DiastolicBP']['low']:
            recommendations.append("Consider compression stockings for low diastolic BP")
    
    # Blood Sugar analysis
    if pd.notna(input_data['BS']):
        if input_data['BS'] >= thresholds['BS']['very_high']:
            recommendations.append("Immediate diabetes screening recommended")
        elif input_data['BS'] >= thresholds['BS']['high']:
            recommendations.append("Monitor carbohydrate intake and blood sugar levels")
        elif input_data['BS'] <= thresholds['BS']['very_low']:
            recommendations.append("Check for hypoglycemia symptoms")
    
    # Body Temperature analysis
    if pd.notna(input_data['BodyTemp']):
        if input_data['BodyTemp'] >= thresholds['BodyTemp']['very_high']:
            recommendations.append("Seek immediate treatment for high fever")
        elif input_data['BodyTemp'] >= thresholds['BodyTemp']['high']:
            recommendations.append("Monitor temperature and stay hydrated")
        elif input_data['BodyTemp'] <= thresholds['BodyTemp']['very_low']:
            recommendations.append("Medical evaluation for hypothermia")
    
    # Heart Rate analysis
    if pd.notna(input_data['HeartRate']):
        if input_data['HeartRate'] >= thresholds['HeartRate']['very_high']:
            recommendations.append("Cardiac evaluation recommended for high heart rate")
        elif input_data['HeartRate'] >= thresholds['HeartRate']['high']:
            recommendations.append("Reduce caffeine and monitor heart rate")
        elif input_data['HeartRate'] <= thresholds['HeartRate']['very_low']:
            recommendations.append("Medical evaluation for low heart rate")
    
    # Pregnancy-specific recommendations
    if any(keyword in ";".join(recommendations) for keyword in ["BP", "blood pressure", "sugar", "diabetes"]):
        recommendations.append("Prenatal monitoring recommended")
    
    # If no specific recommendations
    if not recommendations:
        recommendations.append("All parameters normal - maintain routine prenatal care")
    
    return "; ".join(recommendations)

# Streamlit app
st.title("Maternal Health Risk Prediction Chatbot")
st.write("Please provide the following details:")

# Define feature statistics
feature_stats = {
    "Age": {"mean": 29.87, "std": 13.47, "min": 10, "max": 70},
    "SystolicBP": {"mean": 113.19, "std": 18.40, "min": 70, "max": 160},
    "DiastolicBP": {"mean": 76.46, "std": 13.86, "min": 49, "max": 100},
    "BS": {"mean": 8.72, "std": 3.29, "min": 6.0, "max": 19.0},  # Changed to float
    "BodyTemp": {"mean": 98.66, "std": 1.37, "min": 98.0, "max": 103.0},  # Changed to float
    "HeartRate": {"mean": 74.30, "std": 8.08, "min": 60, "max": 90},
}

# Collect user inputs with consistent numeric types
age = st.number_input(
    "Age",
    min_value=int(feature_stats["Age"]["min"]),
    max_value=int(feature_stats["Age"]["max"]),
    value=int(feature_stats["Age"]["mean"]),
    step=1,
    help=f"Typical range: {feature_stats['Age']['min']} to {feature_stats['Age']['max']}",
)

systolic_bp = st.number_input(
    "Systolic Blood Pressure",
    min_value=int(feature_stats["SystolicBP"]["min"]),
    max_value=int(feature_stats["SystolicBP"]["max"]),
    value=int(feature_stats["SystolicBP"]["mean"]),
    step=1,
    help=f"Typical range: {feature_stats['SystolicBP']['min']} to {feature_stats['SystolicBP']['max']}",
)

diastolic_bp = st.number_input(
    "Diastolic Blood Pressure",
    min_value=int(feature_stats["DiastolicBP"]["min"]),
    max_value=int(feature_stats["DiastolicBP"]["max"]),
    value=int(feature_stats["DiastolicBP"]["mean"]),
    step=1,
    help=f"Typical range: {feature_stats['DiastolicBP']['min']} to {feature_stats['DiastolicBP']['max']}",
)

bs = st.number_input(
    "Blood Sugar Level",
    min_value=float(feature_stats["BS"]["min"]),
    max_value=float(feature_stats["BS"]["max"]),
    value=float(feature_stats["BS"]["mean"]),
    step=0.1,
    format="%.1f",
    help=f"Typical range: {feature_stats['BS']['min']} to {feature_stats['BS']['max']}",
)

body_temp = st.number_input(
    "Body Temperature (°F)",
    min_value=float(feature_stats["BodyTemp"]["min"]),
    max_value=float(feature_stats["BodyTemp"]["max"]),
    value=float(feature_stats["BodyTemp"]["mean"]),
    step=0.1,
    format="%.1f",
    help=f"Typical range: {feature_stats['BodyTemp']['min']} to {feature_stats['BodyTemp']['max']}",
)

heart_rate = st.number_input(
    "Heart Rate",
    min_value=int(feature_stats["HeartRate"]["min"]),
    max_value=int(feature_stats["HeartRate"]["max"]),
    value=int(feature_stats["HeartRate"]["mean"]),
    step=1,
    help=f"Typical range: {feature_stats['HeartRate']['min']} to {feature_stats['HeartRate']['max']}",
)

if st.button("Predict Risk"):
    # Prepare input dictionary
    user_input = {
        'Age': age,
        'SystolicBP': systolic_bp,
        'DiastolicBP': diastolic_bp,
        'BS': bs,
        'BodyTemp': body_temp,
        'HeartRate': heart_rate
    }
    
    # Generate model prediction
    manual_input_df = pd.DataFrame([user_input])
    
    # Ensure preprocessing matches training
    with open('columns.json', 'r') as f:
        columns = json.load(f)
    manual_input_df = pd.get_dummies(manual_input_df)
    for col in columns:
        if col not in manual_input_df.columns:
            manual_input_df[col] = 0
    manual_input_df = manual_input_df[columns]
    
    # Get model prediction
    prediction = multi_output_model.predict(manual_input_df)
    predicted_risk = risk_map[prediction[0][0]]  # RiskLevel
    
    # Generate recommendations based on thresholds
    recommendations = generate_recommendations(user_input)
    
    # Display results
    st.write(f"**Predicted Risk Level:** {predicted_risk}")
    st.write(f"**Recommendations:** {recommendations}")
    
    # Show threshold analysis
    with st.expander("Detailed Parameter Analysis"):
        st.write("**Age Analysis:**")
        if age > thresholds['Age']['high']:
            st.warning(f"Age {age} is above the typical maximum ({thresholds['Age']['high']})")
        elif age < 18:
            st.warning(f"Age {age} indicates adolescent pregnancy")
        else:
            st.success(f"Age {age} is within typical range")
            
        st.write("**Blood Pressure Analysis:**")
        if systolic_bp >= thresholds['SystolicBP']['very_high']:
            st.error(f"Systolic BP {systolic_bp} is very high (≥{thresholds['SystolicBP']['very_high']})")
        elif systolic_bp >= thresholds['SystolicBP']['high']:
            st.warning(f"Systolic BP {systolic_bp} is elevated (≥{thresholds['SystolicBP']['high']})")
        elif systolic_bp <= thresholds['SystolicBP']['very_low']:
            st.error(f"Systolic BP {systolic_bp} is very low (≤{thresholds['SystolicBP']['very_low']})")
        elif systolic_bp <= thresholds['SystolicBP']['low']:
            st.warning(f"Systolic BP {systolic_bp} is low (≤{thresholds['SystolicBP']['low']})")
        else:
            st.success(f"Systolic BP {systolic_bp} is normal")
            
        if diastolic_bp >= thresholds['DiastolicBP']['very_high']:
            st.error(f"Diastolic BP {diastolic_bp} is very high (≥{thresholds['DiastolicBP']['very_high']})")
        elif diastolic_bp >= thresholds['DiastolicBP']['high']:
            st.warning(f"Diastolic BP {diastolic_bp} is elevated (≥{thresholds['DiastolicBP']['high']})")
        elif diastolic_bp <= thresholds['DiastolicBP']['very_low']:
            st.error(f"Diastolic BP {diastolic_bp} is very low (≤{thresholds['DiastolicBP']['very_low']})")
        elif diastolic_bp <= thresholds['DiastolicBP']['low']:
            st.warning(f"Diastolic BP {diastolic_bp} is low (≤{thresholds['DiastolicBP']['low']})")
        else:
            st.success(f"Diastolic BP {diastolic_bp} is normal")
            
        st.write("**Blood Sugar Analysis:**")
        if bs >= thresholds['BS']['very_high']:
            st.error(f"Blood sugar {bs} is very high (≥{thresholds['BS']['very_high']})")
        elif bs >= thresholds['BS']['high']:
            st.warning(f"Blood sugar {bs} is elevated (≥{thresholds['BS']['high']})")
        elif bs <= thresholds['BS']['very_low']:
            st.error(f"Blood sugar {bs} is very low (≤{thresholds['BS']['very_low']})")
        else:
            st.success(f"Blood sugar {bs} is normal")
            
        st.write("**Body Temperature Analysis:**")
        if body_temp >= thresholds['BodyTemp']['very_high']:
            st.error(f"Body temperature {body_temp}°F indicates fever (≥{thresholds['BodyTemp']['very_high']}°F)")
        elif body_temp >= thresholds['BodyTemp']['high']:
            st.warning(f"Body temperature {body_temp}°F is elevated (≥{thresholds['BodyTemp']['high']}°F)")
        elif body_temp <= thresholds['BodyTemp']['very_low']:
            st.error(f"Body temperature {body_temp}°F is very low (≤{thresholds['BodyTemp']['very_low']}°F)")
        else:
            st.success(f"Body temperature {body_temp}°F is normal")
            
        st.write("**Heart Rate Analysis:**")
        if heart_rate >= thresholds['HeartRate']['very_high']:
            st.error(f"Heart rate {heart_rate} is very high (≥{thresholds['HeartRate']['very_high']})")
        elif heart_rate >= thresholds['HeartRate']['high']:
            st.warning(f"Heart rate {heart_rate} is elevated (≥{thresholds['HeartRate']['high']})")
        elif heart_rate <= thresholds['HeartRate']['very_low']:
            st.error(f"Heart rate {heart_rate} is very low (≤{thresholds['HeartRate']['very_low']})")
        else:
            st.success(f"Heart rate {heart_rate} is normal")