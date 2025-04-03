import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import os

# Load models safely
DIABETES_MODEL_PATH = '/users/rahul/Desktop/Multiple_Disease_prediction/dmodel.sav'
HEART_MODEL_PATH = '/users/rahul/Desktop/Multiple_Disease_prediction/model.sav'

if os.path.exists(DIABETES_MODEL_PATH) and os.path.exists(HEART_MODEL_PATH):
    diabetes_model = pickle.load(open(DIABETES_MODEL_PATH, 'rb'))
    heart_model = pickle.load(open(HEART_MODEL_PATH, 'rb'))
else:
    st.error("Error: Model files not found. Check file paths.")
    st.stop()

# Sidebar Menu
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction'],
        icons=['activity', 'heart'],
        default_index=0
    )

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction')
    
    col1, col2 = st.columns(2)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies', '0')
        Glucose = st.text_input('Glucose Level', '0')
        BloodPressure = st.text_input('Blood Pressure', '0')
        SkinThickness = st.text_input('Skin Thickness', '0')
        Insulin = st.text_input('Insulin Level', '0')
    with col2:
        BMI = st.text_input('BMI', '0')
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function', '0')
        Age = st.text_input('Age', '0')
    
    if st.button('Diabetes Test Result'):
        try:
            input_data = np.array([
                float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness), 
                float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)
            ]).reshape(1, -1)
            diabetes_prediction = diabetes_model.predict(input_data)
            
            if diabetes_prediction[0] == 1:
                st.success('The person is diabetic.')
            else:
                st.success('The person is not diabetic.')
        except ValueError:
            st.error("Please enter valid numerical values.")

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction')
    
    col1, col2 = st.columns(2)
    with col1:
        Age = st.text_input('Age', '0')
        Sex = st.text_input('Sex (1 = Male, 0 = Female)', '0')
        ChestPainType = st.text_input('Chest Pain Type (0-3)', '0')
        RestingBP = st.text_input('Resting Blood Pressure', '0')
        Cholesterol = st.text_input('Serum Cholesterol (mg/dL)', '0')
        FastingBS = st.text_input('Fasting Blood Sugar > 120 mg/dL (1 = Yes, 0 = No)', '0')
    with col2:
        RestingECG = st.text_input('Resting ECG Results (0-2)', '0')
        MaxHR = st.text_input('Maximum Heart Rate Achieved', '0')
        ExerciseAngina = st.text_input('Exercise-Induced Angina (1 = Yes, 0 = No)', '0')
        Oldpeak = st.text_input('ST Depression Induced by Exercise', '0')
        ST_Slope = st.text_input('ST Slope (0-2)', '0')
    
    if st.button('Heart Disease Test Result'):
        try:
            input_data = np.array([
                float(Age), float(Sex), float(ChestPainType), float(RestingBP), float(Cholesterol), 
                float(FastingBS), float(RestingECG), float(MaxHR), float(ExerciseAngina), float(Oldpeak), float(ST_Slope)
            ]).reshape(1, -1)
            heart_prediction = heart_model.predict(input_data)
            
            if heart_prediction[0] == 1:
                st.success('The person has heart disease.')
            else:
                st.success('The person does not have heart disease.')
        except ValueError:
            st.error("Please enter valid numerical values.")
