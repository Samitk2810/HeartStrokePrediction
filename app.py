import streamlit as st
import pandas as pd
import joblib as jb

model = jb.load('LR_heart.pkl')
scaler = jb.load('scaler.pkl')
columns = jb.load('columns.pkl')

st.title("Heart Disease Prediction By Samit 💓")
st.markdown("Provide the following details: ")

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ['M', 'F'])
cp = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Serum Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting Electrocardiographic Results", ["Normal", "ST", "LVH"])
max_hr = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", ["Y", "N"])
oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.2, 1.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", ["Up", "Flat", "Down"])

if st.button("Predict"):
    # Create a DataFrame with the input values
    raw_input ={
        'Age': age,
        'Sex_'+sex:1,
        'ChestPainType_'+cp:1,
        'RestingBP': trestbps,
        'Cholestrol': chol,
        'FastingBS': fbs,
        'RestingECG_'+restecg:1,
        'MaxHR': max_hr,
        'ExerciseAngina_'+exang:1,
        'Oldpeak': oldpeak,
        'ST_Slope_'+slope:1
    }

    input_data = pd.DataFrame([raw_input])

    for col in columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[columns]  # Ensure the order of columns matches the training data           
    
    # Scale the input data
    numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

    # Scale only numeric features
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

    # Predict using the full DataFrame
    prediction = model.predict(input_data)

    # Display the results
    st.write("Prediction:")
    if prediction[0] == 1:
        st.write("⚠️ High risk of having a heart disease.")
    else:
        st.write("✅ Low risk of having a heart disease.")
