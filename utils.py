import joblib
import pandas as pd
import numpy as np


# ===============================================
# Load Model Files
# ===============================================

def load_model():
    model = joblib.load("LR.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("col.pkl")

    print("TYPE:", type(columns))
    print("VALUE:", columns)

    return model, scaler, columns

# ===============================================
# One Hot Encoding
# ===============================================

def preprocess_input(
    age,
    bp,
    cholesterol,
    fasting,
    maxhr,
    oldpeak,
    sex,
    chestpain,
    ecg,
    exercise,
    slope,
    columns,
):

    data = dict.fromkeys(columns, 0)

    # Numerical

    data["Age"] = age
    data["RestingBP"] = bp
    data["Cholesterol"] = cholesterol
    data["FastingBS"] = fasting
    data["MaxHR"] = maxhr
    data["Oldpeak"] = oldpeak

    # Gender

    if sex == "Male":
        data["Sex_M"] = 1
    else:
        data["Sex_F"] = 1

    # Chest Pain

    chest_map = {
        "ASY": "ChestPainType_ASY",
        "ATA": "ChestPainType_ATA",
        "NAP": "ChestPainType_NAP",
        "TA": "ChestPainType_TA",
    }

    if chestpain in chest_map:
        data[chest_map[chestpain]] = 1

    # ECG

    ecg_map = {
        "Normal": "RestingECG_Normal",
        "LVH": "RestingECG_LVH",
        "ST": "RestingECG_ST",
    }

    if ecg in ecg_map:
        data[ecg_map[ecg]] = 1

    # Exercise Angina

    if exercise == "Yes":
        data["ExerciseAngina_Y"] = 1
    else:
        data["ExerciseAngina_N"] = 1

    # ST Slope

    slope_map = {
        "Up": "ST_Slope_Up",
        "Flat": "ST_Slope_Flat",
        "Down": "ST_Slope_Down",
    }

    if slope in slope_map:
        data[slope_map[slope]] = 1

    return pd.DataFrame([data])


# ===============================================
# Prediction
# ===============================================

def predict_heart_disease(
    model,
    scaler,
    dataframe,
):

    scaled = scaler.transform(dataframe)

    prediction = model.predict(scaled)[0]

    probability = model.predict_proba(scaled)[0][1]

    return prediction, probability


# ===============================================
# Risk Level
# ===============================================

def risk_level(probability):

    if probability < 0.30:
        return "Low Risk", "green"

    elif probability < 0.60:
        return "Moderate Risk", "orange"

    else:
        return "High Risk", "red"


# ===============================================
# Recommendation
# ===============================================

def recommendation(prediction):

    if prediction == 1:

        return [
            "Consult a Cardiologist",
            "Monitor Blood Pressure",
            "Reduce Salt Intake",
            "Exercise Regularly",
            "Avoid Smoking",
            "Take Prescribed Medicines",
            "Maintain Healthy Weight",
            "Regular ECG Check-up",
        ]

    return [
        "Maintain Healthy Lifestyle",
        "Eat Balanced Diet",
        "Regular Exercise",
        "Drink Plenty of Water",
        "Sleep 7–8 Hours",
        "Annual Health Check-up",
        "Reduce Stress",
        "Stay Active",
    ]


# ===============================================
# Probability Text
# ===============================================

def probability_text(probability):

    return f"{probability * 100:.2f}%"