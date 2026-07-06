import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import base64

from utils import (
    load_model,
    preprocess_input,
    predict_heart_disease,
    recommendation,
    risk_level,
    probability_text
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# LOAD CSS
# =====================================================

def load_css(css_file):
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# =====================================================
# LOAD BACKGROUND IMAGE
# =====================================================

def set_background(image_file):

    with open(image_file, "rb") as image:

        encoded = base64.b64encode(image.read()).decode()

    st.markdown(
        f"""
        <style>

        .stApp{{
            background-image:url("data:image/jpg;base64,{encoded}");
            background-size:cover;
            background-position:center;
            background-attachment:fixed;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

# Uncomment after adding image

# set_background("assets/heart_bg.jpg")

# =====================================================
# LOAD MODEL
# =====================================================

model, scaler, columns = load_model()

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.image("assets/logo.png", width=120)

    st.title("❤️ Heart Disease")

    st.markdown("---")

    st.success("Machine Learning Model")

    st.write("**Algorithm**")
    st.info("Logistic Regression")

    st.write("**Scaler**")
    st.info("StandardScaler")

    st.write("**Features**")
    st.info("20")

    st.markdown("---")

    st.caption("Developed by")
    st.write("**Mohd Nasrullah Siddiqui**")

# =====================================================
# HERO SECTION
# =====================================================

left, right = st.columns([1,2])

with left:

    st.image("assets/heart.png", width=260)

with right:

    st.markdown(
        """
        <h1 class='main-title'>
        ❤️ Heart Disease Prediction
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p class='sub-title'>

        AI Powered Clinical Decision Support System

        Predict the possibility of cardiovascular disease
        using Machine Learning and Logistic Regression.

        </p>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# =====================================================
# INFORMATION CARDS
# =====================================================

c1, c2, c3 = st.columns(3)

with c1:

    st.metric(
        "Model",
        "Logistic Regression"
    )

with c2:

    st.metric(
        "Features",
        "20"
    )

with c3:

    st.metric(
        "Prediction",
        "Binary"
    )

st.markdown("---")

# =====================================================
# PATIENT DETAILS TITLE
# =====================================================

st.markdown(
"""
<div class="glass">

<h2>🩺 Patient Clinical Information</h2>

Enter the patient's medical information below.

</div>

""",
unsafe_allow_html=True
)


# =====================================================
# PATIENT INPUT FORM
# =====================================================

col1, col2 = st.columns(2)

# -----------------------------
# LEFT COLUMN
# -----------------------------

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=40
    )

    sex = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure (mmHg)",
        min_value=80,
        max_value=220,
        value=120
    )

    cholesterol = st.number_input(
        "Cholesterol (mg/dL)",
        min_value=0,
        max_value=700,
        value=200
    )

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL",
        ["No", "Yes"]
    )

# -----------------------------
# RIGHT COLUMN
# -----------------------------

with col2:

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ASY", "ATA", "NAP", "TA"]
    )

    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "LVH", "ST"]
    )

    max_hr = st.slider(
        "Maximum Heart Rate",
        60,
        220,
        150
    )

    old_peak = st.slider(
        "Old Peak",
        0.0,
        6.5,
        1.0,
        0.1
    )

    exercise_angina = st.selectbox(
        "Exercise Induced Angina",
        ["No", "Yes"]
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# LIVE PATIENT SUMMARY
# =====================================================

st.subheader("📋 Patient Summary")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Age", age)

with m2:
    st.metric("Blood Pressure", f"{resting_bp} mmHg")

with m3:
    st.metric("Cholesterol", cholesterol)

with m4:
    st.metric("Max Heart Rate", max_hr)

st.markdown("---")

# =====================================================
# PREDICT BUTTON
# =====================================================

predict_btn = st.button(
    "❤️ Predict Heart Disease",
    use_container_width=True
)

# =====================================================
# PREPROCESS INPUT
# =====================================================

if predict_btn:

    fasting = 1 if fasting_bs == "Yes" else 0

    input_df = preprocess_input(

        age=age,
        bp=resting_bp,
        cholesterol=cholesterol,
        fasting=fasting,
        maxhr=max_hr,
        oldpeak=old_peak,
        sex=sex,
        chestpain=chest_pain,
        ecg=resting_ecg,
        exercise=exercise_angina,
        slope=st_slope,
        columns=columns

    )

    prediction, probability = predict_heart_disease(

        model,
        scaler,
        input_df

    )

    risk, color = risk_level(probability)

    advice = recommendation(prediction)

# =====================================================
# PREDICTION RESULT
# =====================================================

st.markdown("---")

st.subheader("🩺 Prediction Result")

if prediction == 1:

    st.markdown(
        """
        <div class="error-card">
        ❤️ High Risk of Heart Disease Detected
        </div>
        """,
        unsafe_allow_html=True
    )

else:

    st.markdown(
        """
        <div class="success-card">
        💚 Low Risk of Heart Disease
        </div>
        """,
        unsafe_allow_html=True
    )

# =====================================================
# PROBABILITY
# =====================================================

st.markdown("### 🎯 Prediction Confidence")

st.progress(float(probability))

st.metric(
    "Probability",
    probability_text(probability)
)

st.markdown("---")

# =====================================================
# GAUGE CHART
# =====================================================

fig = go.Figure(go.Indicator(

    mode="gauge+number",

    value=probability*100,

    title={
        "text":"Heart Disease Risk (%)",
        "font":{"size":24}
    },

    gauge={

        "axis":{"range":[0,100]},

        "bar":{"color":"darkred"},

        "steps":[

            {
                "range":[0,30],
                "color":"#2ecc71"
            },

            {
                "range":[30,60],
                "color":"#f1c40f"
            },

            {
                "range":[60,100],
                "color":"#e74c3c"
            }

        ],

    }

))

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# RISK LEVEL
# =====================================================

st.markdown("### 📊 Risk Assessment")

risk_color = {
    "green":"🟢",
    "orange":"🟠",
    "red":"🔴"
}

st.info(
    f"{risk_color[color]} **{risk}**"
)

# =====================================================
# HEALTH SUMMARY
# =====================================================

st.markdown("### 📈 Health Summary")

m1,m2,m3 = st.columns(3)

with m1:

    st.metric(
        "Blood Pressure",
        resting_bp
    )

with m2:

    st.metric(
        "Cholesterol",
        cholesterol
    )

with m3:

    st.metric(
        "Maximum HR",
        max_hr
    )

# =====================================================
# PIE CHART
# =====================================================

pie = go.Figure(

    data=[

        go.Pie(

            labels=[
                "Risk",
                "Healthy"
            ],

            values=[
                probability*100,
                100-probability*100
            ],

            hole=.60

        )

    ]

)

pie.update_layout(

    title="Risk Distribution"

)

st.plotly_chart(

    pie,

    use_container_width=True

)

# =====================================================
# RECOMMENDATIONS
# =====================================================

st.markdown("### 👨‍⚕️ Doctor Recommendations")

st.markdown(
    '<div class="recommend">',
    unsafe_allow_html=True
)

for item in advice:

    st.write(f"✅ {item}")

st.markdown(
    "</div>",
    unsafe_allow_html=True
)

st.markdown("---")

# =====================================================
# DOWNLOAD PREDICTION REPORT
# =====================================================

st.markdown("### 📄 Download Prediction Report")

report = f"""
=========================================
        HEART DISEASE PREDICTION REPORT
=========================================

Patient Information
------------------------
Age                : {age}
Gender             : {sex}
Blood Pressure     : {resting_bp}
Cholesterol        : {cholesterol}
Chest Pain Type    : {chest_pain}
Resting ECG        : {resting_ecg}
Max Heart Rate     : {max_hr}
Old Peak           : {old_peak}
Exercise Angina    : {exercise_angina}
ST Slope           : {st_slope}

-----------------------------------------

Prediction

Risk Level         : {risk}

Probability        : {probability*100:.2f} %

-----------------------------------------

Recommendations

"""

for rec in advice:
    report += f"\n• {rec}"

st.download_button(
    label="📥 Download Report",
    data=report,
    file_name="Heart_Disease_Report.txt",
    mime="text/plain",
)

st.markdown("---")

# =====================================================
# ABOUT MODEL
# =====================================================

tab1, tab2 = st.tabs(["🤖 Model", "📊 Dataset"])

with tab1:

    st.markdown("""
### Logistic Regression Model

This application predicts heart disease using a Logistic Regression model.

### Machine Learning Pipeline

- Data Cleaning
- One-Hot Encoding
- Feature Scaling
- Logistic Regression
- Probability Prediction

### Libraries

- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- Plotly
""")

with tab2:

    st.markdown("""
### Dataset Information

Total Features : **20**

Target :

- 0 = Healthy
- 1 = Heart Disease

Input Parameters

- Age
- Gender
- Resting Blood Pressure
- Cholesterol
- Chest Pain Type
- ECG
- Maximum Heart Rate
- Exercise Angina
- ST Slope
- Old Peak

""")

st.markdown("---")

# =====================================================
# FOOTER
# =====================================================

st.markdown(
"""
<div class="footer">

❤️ Heart Disease Prediction System

Developed by <b>Mohd Nasrullah Siddiqui</b>

Powered by Streamlit | Scikit-Learn | Plotly

</div>
""",
unsafe_allow_html=True,
)