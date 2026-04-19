import streamlit as st
import pandas as pd
import joblib


model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")


st.title("🍔 Obesity Level Prediction App")

st.write("Fill in your details:")


gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", 10, 100, 25)
height = st.number_input("Height (meters)", 1.0, 2.5, 1.7)
weight = st.number_input("Weight (kg)", 30, 200, 70)

family_history = st.selectbox("Family History Overweight", ["yes", "no"])
favc = st.selectbox("Frequent High Calorie Food (FAVC)", ["yes", "no"])

fcvc = st.slider("Vegetable Consumption (1-3)", 1, 3, 2)
ncp = st.slider("Number of Meals (NCP)", 1, 5, 3)

caec = st.selectbox("Food Between Meals (CAEC)", ["no", "Sometimes", "Frequently", "Always"])

smoke = st.selectbox("Do you smoke?", ["yes", "no"])

ch2o = st.slider("Water Intake (CH2O)", 1, 3, 2)

scc = st.selectbox("Do you monitor calories (SCC)?", ["yes", "no"])

faf = st.slider("Physical Activity (FAF)", 0, 3, 1)
tue = st.slider("Technology Usage (TUE)", 0, 10, 3)

calc = st.selectbox("Alcohol Consumption (CALC)", ["no", "Sometimes", "Frequently"])

mtrans = st.selectbox("Transportation (MTRANS)", 
                      ["Walking", "Bike", "Motorbike", "Public_Transportation", "Automobile"])


if st.button("Predict"):

    input_dict = {
        "Gender": gender,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "family_history_with_overweight": family_history,
        "FAVC": favc,
        "FCVC": fcvc,
        "NCP": ncp,
        "CAEC": caec,
        "SMOKE": smoke,
        "CH2O": ch2o,
        "SCC": scc,
        "FAF": faf,
        "TUE": tue,
        "CALC": calc,
        "MTRANS": mtrans
    }

    input_df = pd.DataFrame([input_dict])

    
    for col in input_df.columns:
        if col in encoders:
            input_df[col] = encoders[col].transform(input_df[col])

  
    input_scaled = scaler.transform(input_df)

  
    pred = model.predict(input_scaled)

    result = target_encoder.inverse_transform(pred)[0]

   
    st.success(f"🎯 Predicted Obesity Level: {result}")

    # Extra Message
    if "Obesity" in result:
        st.error("⚠️ High Risk! Improve diet and exercise.")
    else:
        st.info("✅ You are in a healthy range!")
