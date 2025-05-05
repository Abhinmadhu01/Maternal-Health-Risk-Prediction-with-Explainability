# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib
import json

# Load the trained multi-output model
multi_output_model = joblib.load('risk_level_xgb_model.pkl')

# Define the risk map
risk_map = {0: 'Low Risk', 1: 'High Risk'}

# Define thresholds based on the statistical analysis
thresholds = {
    'Age': {
        'high': 70,
        'low': 10
    },
    'SystolicBP': {
        'high': 140,
        'very_high': 160,
        'low': 90,
        'very_low': 70
    },
    'DiastolicBP': {
        'high': 90,
        'very_high': 100,
        'low': 60,
        'very_low': 49
    },
    'BS': {
        'high': 12.0,
        'very_high': 15.0,
        'low': 6.0,
        'very_low': 4.0
    },
    'BodyTemp': {
        'high': 100.0,
        'very_high': 101.0,
        'low': 97.0,
        'very_low': 96.0
    },
    'HeartRate': {
        'high': 82,
        'very_high': 90,
        'low': 66,
        'very_low': 60
    },
    'BMI': {
        'high': 30,
        'very_high': 35,
        'low': 18.5,
        'very_low': 16
    },
    'Previous Complications': {
        'high': 1,
        'low': 0
    },
    'Preexisting Diabetes': {
        'high': 1,
        'low': 0
    },
    'Gestational Diabetes': {
        'high': 1,
        'low': 0
    },
    'Mental Health': {
        'high': 1,
        'low': 0
    }
}


# Updated guidelines-based recommendation generator
def generate_recommendations(input_data, thresholds):
    recommendations = []
    
    # Advanced Maternal Age (ACOG: ≥35 years) and Adolescent Pregnancy (WHO: 10-19 years)
    if pd.notna(input_data['Age']):
        if input_data['Age'] >= 35:
            recommendations.append(
                "Advanced maternal age (≥35 years): consider additional fetal aneuploidy screening and non-invasive prenatal testing per ACOG guidelines"
            )  # ([pubmed.ncbi.nlm.nih.gov](https://pubmed.ncbi.nlm.nih.gov/35852294/?utm_source=chatgpt.com))
        elif input_data['Age'] < 20:
            recommendations.append(
                "Adolescent pregnancy: ensure comprehensive adolescent-friendly care and counseling per WHO recommendations"
            )  # ([who.int](https://www.who.int/news/item/23-04-2025-who-releases-new-guideline-to-prevent-adolescent-pregnancies-and-improve-girls--health?utm_source=chatgpt.com))
    
    # Blood Pressure (ACOG classification: Normal <120/80, Stage 1 130-139/80-89, Stage 2 ≥140/90)
    if pd.notna(input_data['SystolicBP']) and pd.notna(input_data['DiastolicBP']):
        sys, dia = input_data['SystolicBP'], input_data['DiastolicBP']
        if sys >= 140 or dia >= 90:
            recommendations.append(
                "Hypertension in pregnancy (Stage 2): initiate antihypertensive therapy and consider low-dose aspirin prophylaxis (81 mg/day) after 12 weeks gestation per ACOG"
            )  # ([acog.org](https://www.acog.org/topics/hypertension-and-preeclampsia-in-pregnancy?utm_source=chatgpt.com), [journals.lww.com](https://journals.lww.com/greenjournal/fulltext/2018/07000/acog_committee_opinion_no__743_summary__low_dose.52.aspx?utm_source=chatgpt.com))
        elif 130 <= sys < 140 or 80 <= dia < 90:
            recommendations.append(
                "Stage 1 hypertension: monitor BP biweekly and encourage dietary sodium restriction per ACOG guidance"
            )  # ([acog.org](https://www.acog.org/topics/hypertension-and-preeclampsia-in-pregnancy?utm_source=chatgpt.com))
        elif sys < 90 or dia < 60:
            recommendations.append(
                "Hypotension (<90/60 mmHg): assess for volume status and advise increased fluid intake per clinical best practices"
            )
    
    # Gestational and Preexisting Diabetes (ADA Standards)
    if pd.notna(input_data['BS']):
        bs = input_data['BS']
        # Gestational diabetes screening typically at 24-28 weeks, but early screening if risk factors
        if bs >= thresholds['BS']['very_high']:
            recommendations.append(
                "Elevated blood glucose: refer for oral glucose tolerance test (OGTT) and begin medical nutrition therapy per ADA standards"
            )  # ([diabetes.org](https://diabetes.org/living-with-diabetes/pregnancy/gestational-diabetes?utm_source=chatgpt.com), [diabetesjournals.org](https://diabetesjournals.org/care/article/47/Supplement_1/S282/153948/15-Management-of-Diabetes-in-Pregnancy-Standards?utm_source=chatgpt.com))
        elif bs <= thresholds['BS']['very_low']:
            recommendations.append(
                "Hypoglycemia (<70 mg/dL): evaluate for symptoms and consider dietary modification per ADA hypoglycemia guidelines"
            )
    if pd.notna(input_data['Preexisting Diabetes']) and input_data['Preexisting Diabetes'] == 1:
        recommendations.append(
            "Preexisting diabetes: optimize glycemic control (HbA1c <6.5%) and coordinate care with endocrinology per ADA pregnancy standards"
        )  # ([diabetesjournals.org](https://diabetesjournals.org/care/article/47/Supplement_1/S282/153948/15-Management-of-Diabetes-in-Pregnancy-Standards?utm_source=chatgpt.com))
    if pd.notna(input_data['Gestational Diabetes']) and input_data['Gestational Diabetes'] == 1:
        recommendations.append(
            "Gestational diabetes: implement diet and exercise plan, self-monitoring of blood glucose, and consider insulin therapy as needed"
        )  # ([diabetes.org](https://diabetes.org/living-with-diabetes/pregnancy/gestational-diabetes?utm_source=chatgpt.com))
    
    # Fever and Hypothermia (WHO, Cleveland Clinic)
    if pd.notna(input_data['BodyTemp']):
        temp = input_data['BodyTemp']
        if temp >= 38.0:
            recommendations.append(
                "Fever (≥38°C): seek evaluation for infection and initiate antipyretic therapy"
            )
        elif temp < 35.0:
            recommendations.append(
                "Hypothermia (<35°C): urgent medical evaluation and warming measures per hypothermia management protocols"
            )  # ([ncbi.nlm.nih.gov](https://www.ncbi.nlm.nih.gov/books/NBK545239/?utm_source=chatgpt.com), [my.clevelandclinic.org](https://my.clevelandclinic.org/health/diseases/21164-hypothermia-low-body-temperature?utm_source=chatgpt.com))
    
    # Heart Rate (Tachycardia >100 bpm)
    if pd.notna(input_data['HeartRate']):
        hr = input_data['HeartRate']
        if hr > 100:
            recommendations.append(
                "Sinus tachycardia (>100 bpm): evaluate for infection, anemia, and hyperthyroidism; consider ECG per ACOG"
            )  # ([pmc.ncbi.nlm.nih.gov](https://pmc.ncbi.nlm.nih.gov/articles/PMC8439506/?utm_source=chatgpt.com))
        elif hr < 60:
            recommendations.append(
                "Bradycardia (<60 bpm): assess for symptomatic bradycardia and refer for cardiology evaluation"
            )
    
    # BMI (WHO classification)
    if pd.notna(input_data['BMI']):
        bmi = input_data['BMI']
        if bmi >= 30:
            recommendations.append(
                "Obesity (BMI ≥30): refer to nutritionist for weight management plan and monitor gestational weight gain"
            )  # ([who.int](https://www.who.int/europe/news-room/fact-sheets/item/a-healthy-lifestyle---who-recommendations?utm_source=chatgpt.com))
        elif bmi < 18.5:
            recommendations.append(
                "Underweight (BMI <18.5): advise increased caloric intake and micronutrient supplementation"
            )

    # Previous Complications and Mental Health
    if pd.notna(input_data['Previous Complications']) and input_data['Previous Complications'] == 1:
        recommendations.append(
            "History of pregnancy complications: increase frequency of prenatal visits and targeted fetal surveillance"
        )
    if pd.notna(input_data['Mental Health']) and input_data['Mental Health'] == 1:
        recommendations.append(
            "Perinatal mental health concern: perform depression/anxiety screening (e.g., EPDS) and consider referral to mental health services"
        )  # ([acog.org](https://www.acog.org/programs/perinatal-mental-health/implementing-perinatal-mental-health-screening?utm_source=chatgpt.com))
    
    # Universal Prenatal Care Reminder
    if any(x for x in recommendations if 'pregnancy' in x.lower() or 'glucose' in x.lower()):
        recommendations.append(
            "Ensure ongoing prenatal monitoring and education at each visit"
        )

    if not recommendations:
        recommendations.append(
            "All parameters within normal limits: continue standard prenatal care"
        )
    
    return '; '.join(recommendations)


# Streamlit app
st.title("Maternal Health Risk Prediction Chatbot")
st.write("Please provide the following details:")

# Define feature statistics
feature_stats = {
    "Age": {"mean": 29.87, "std": 13.47, "min": 10, "max": 70},
    "SystolicBP": {"mean": 113.19, "std": 18.40, "min": 70, "max": 160},
    "DiastolicBP": {"mean": 76.46, "std": 13.86, "min": 49, "max": 100},
    "BS": {"mean": 8.72, "std": 3.29, "min": 6.0, "max": 19.0},
    "BodyTemp": {"mean": 98.66, "std": 1.37, "min": 98.0, "max": 103.0},
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

# Collect user inputs
age = st.number_input("Age", min_value=10, max_value=70, value=30, step=1)
systolic_bp = st.number_input("Systolic Blood Pressure", min_value=70, max_value=160, value=113, step=1)
diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=49, max_value=100, value=76, step=1)
bs = st.number_input("Blood Sugar Level", min_value=6.0, max_value=19.0, value=8.7, step=0.1, format="%.1f")
body_temp = st.number_input("Body Temperature (°F)", min_value=96.0, max_value=103.0, value=98.6, step=0.1, format="%.1f")
heart_rate = st.number_input("Heart Rate", min_value=60, max_value=90, value=74, step=1)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1, format="%.1f")
previous_complications = st.selectbox("Previous Complications", options=[0, 1])
preexisting_diabetes = st.selectbox("Preexisting Diabetes", options=[0, 1])
gestational_diabetes = st.selectbox("Gestational Diabetes", options=[0, 1])
mental_health = st.selectbox("Mental Health Issues", options=[0, 1])

if st.button("Predict Risk"):
    # Prepare input dictionary
    user_input = {
        'Age': age,
        'SystolicBP': systolic_bp,
        'DiastolicBP': diastolic_bp,
        'BS': bs,
        'BodyTemp': body_temp,
        'HeartRate': heart_rate,
        'BMI': bmi,
        'Previous Complications': previous_complications,
        'Preexisting Diabetes': preexisting_diabetes,
        'Gestational Diabetes': gestational_diabetes,
        'Mental Health': mental_health
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
    predicted_risk = risk_map[prediction[0]]  # RiskLevel
    
    # Generate recommendations based on thresholds
    recommendations = generate_recommendations(user_input, thresholds)
    
    # Display results
    st.write(f"**Predicted Risk Level:** {predicted_risk}")
    st.write(f"**Recommendations:** {recommendations}")
    
   # Show threshold analysis
with st.expander("Detailed Parameter Analysis"):
    # Age Analysis
    st.write("**Age Analysis:**")
    if age > thresholds['Age']['high']:
        st.warning(
            f"Advanced maternal age (≥{thresholds['Age']['high']} years) — per ACOG, consider offering aneuploidy screening and detailed ultrasound."
        )
    elif age < 18:
        st.warning(
            f"Adolescent pregnancy (<18 years) — per WHO, ensure adolescent-friendly antenatal care and psychosocial support."
        )
    else:
        st.success(
            f"Maternal age ({age} years) within typical range — continue routine antenatal surveillance."
        )

    # Blood Pressure Analysis
    st.write("**Blood Pressure Analysis:**")
    if systolic_bp >= thresholds['SystolicBP']['very_high'] or diastolic_bp >= thresholds['DiastolicBP']['very_high']:
        st.error(
            f"Severe hypertension (≥{thresholds['SystolicBP']['very_high']}/{thresholds['DiastolicBP']['very_high']} mm Hg) — initiate or adjust antihypertensives and evaluate for preeclampsia."
        )
    elif systolic_bp >= thresholds['SystolicBP']['high'] or diastolic_bp >= thresholds['DiastolicBP']['high']:
        st.warning(
            f"Stage 1 hypertension ({systolic_bp}/{diastolic_bp} mm Hg) — monitor biweekly and advise dietary sodium restriction."
        )
    elif systolic_bp <= thresholds['SystolicBP']['very_low'] or diastolic_bp <= thresholds['DiastolicBP']['very_low']:
        st.error(
            f"Hypotension (≤{thresholds['SystolicBP']['very_low']}/{thresholds['DiastolicBP']['very_low']} mm Hg) — assess volume status and consider increased oral fluids."
        )
    elif systolic_bp <= thresholds['SystolicBP']['low'] or diastolic_bp <= thresholds['DiastolicBP']['low']:
        st.warning(
            f"Low blood pressure ({systolic_bp}/{diastolic_bp} mm Hg) — ensure adequate hydration and evaluate for orthostatic symptoms."
        )
    else:
        st.success(
            f"Blood pressure ({systolic_bp}/{diastolic_bp} mm Hg) is within normal limits."
        )

    # Blood Sugar Analysis
    st.write("**Blood Glucose Analysis:**")
    if bs >= thresholds['BS']['very_high']:
        st.error(
            f"Marked hyperglycemia (≥{thresholds['BS']['very_high']} mmol/L) — per ADA, refer for OGTT and initiate medical nutrition therapy."
        )
    elif bs >= thresholds['BS']['high']:
        st.warning(
            f"Elevated blood glucose ({bs} mmol/L) — reinforce diet/exercise plan and self-monitoring."
        )
    elif bs <= thresholds['BS']['very_low']:
        st.error(
            f"Hypoglycemia (≤{thresholds['BS']['very_low']} mmol/L) — assess for symptoms, advise carbohydrate intake."
        )
    else:
        st.success(
            f"Blood glucose ({bs} mmol/L) within target range."
        )

    # Body Temperature Analysis
    st.write("**Body Temperature Analysis:**")
    if body_temp >= thresholds['BodyTemp']['very_high']:
        st.error(
            f"Fever (≥{thresholds['BodyTemp']['very_high']}°F) — evaluate for infection, initiate antipyretics."
        )
    elif body_temp >= thresholds['BodyTemp']['high']:
        st.warning(
            f"Low-grade fever ({body_temp}°F) — monitor temperature and hydration."
        )
    elif body_temp <= thresholds['BodyTemp']['very_low']:
        st.error(
            f"Hypothermia (≤{thresholds['BodyTemp']['very_low']}°F) — urgent evaluation and warming measures."
        )
    else:
        st.success(
            f"Temperature ({body_temp}°F) is normal."
        )

    # Heart Rate Analysis
    st.write("**Cardiac Rate Analysis:**")
    if heart_rate >= thresholds['HeartRate']['very_high']:
        st.error(
            f"Tachycardia (> {thresholds['HeartRate']['very_high']} bpm) — rule out anemia, infection, thyroid dysfunction, and consider ECG."
        )
    elif heart_rate >= thresholds['HeartRate']['high']:
        st.warning(
            f"Mild tachycardia ({heart_rate} bpm) — monitor and review stimulant intake."
        )
    elif heart_rate <= thresholds['HeartRate']['very_low']:
        st.error(
            f"Bradycardia (< {thresholds['HeartRate']['very_low']} bpm) — assess for symptoms and cardiology referral."
        )
    else:
        st.success(
            f"Heart rate ({heart_rate} bpm) within normal physiologic range."
        )

    # BMI Analysis
    st.write("**BMI Analysis:**")
    if bmi >= thresholds['BMI']['very_high']:
        st.error(
            f"Severe obesity (BMI ≥ {thresholds['BMI']['very_high']}) — refer to nutritionist and monitor gestational weight gain."
        )
    elif bmi >= thresholds['BMI']['high']:
        st.warning(
            f"Obesity (BMI ≥ {thresholds['BMI']['high']}) — recommend diet/exercise plan and watch weight gain."
        )
    elif bmi <= thresholds['BMI']['very_low']:
        st.error(
            f"Severe underweight (BMI ≤ {thresholds['BMI']['very_low']}) — evaluate nutritional status and supplement calories."
        )
    elif bmi <= thresholds['BMI']['low']:
        st.warning(
            f"Underweight (BMI ≤ {thresholds['BMI']['low']}) — encourage increased caloric and protein intake."
        )
    else:
        st.success(
            f"BMI ({bmi}) is within normal range."
        )

    # Previous Complications
    st.write("**Obstetric History Analysis:**")
    if previous_complications == 1:
        st.warning(
            "History of pregnancy complications — increase antenatal visit frequency and targeted fetal surveillance."
        )
    else:
        st.success(
            "No prior complications — standard prenatal visit schedule."
        )

    # Preexisting Diabetes
    st.write("**Preexisting Diabetes Analysis:**")
    if preexisting_diabetes == 1:
        st.warning(
            "Preexisting diabetes — optimize glycemic control (HbA1c <6.5%) and coordinate endocrinology care."
        )
    else:
        st.success(
            "No preexisting diabetes — continue routine glucose monitoring."
        )

    # Gestational Diabetes
    st.write("**Gestational Diabetes Analysis:**")
    if gestational_diabetes == 1:
        st.warning(
            "Gestational diabetes — implement diet/exercise, self-monitoring of BG, consider insulin if needed."
        )
    else:
        st.success(
            "No gestational diabetes — follow standard screening schedule at 24–28 weeks."
        )

    # Mental Health
    st.write("**Perinatal Mental Health Analysis:**")
    if mental_health == 1:
        st.warning(
            "Reported mental health concerns — conduct EPDS screening and refer to perinatal mental health services."
        )
    else:
        st.success(
            "No mental health concerns — continue routine psychosocial support."
        )
