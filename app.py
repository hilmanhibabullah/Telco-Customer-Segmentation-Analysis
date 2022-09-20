import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title('Customer Churn Predictions')
st.write(''' 
        Created by Group N

        Use sidebar to select input feature
''' )

df = pd.read_csv('clean_telco.csv')

dataframe = st.checkbox('Show DataFrame')

if dataframe:
    st.write(df)

st.sidebar.header('User Input Features')

def user_input():
    gender = st.sidebar.radio('Gender', df['gender'].unique())
    SeniorCitizen = st.sidebar.number_input('Senior Citizen', min_value=0, max_value=1 )
    Partner = st.sidebar.radio('Partner', df['Partner'].unique())
    Dependents = st.sidebar.radio('Dependents', df['Dependents'].unique())
    tenure = st.sidebar.number_input('Tenure', min_value=0, max_value=72)
    PhoneService = st.sidebar.radio('Phone Service',df['PhoneService'].unique())
    MultipleLines = st.sidebar.radio('MultipleLines',df['MultipleLines'].unique())
    InternetService = st.sidebar.radio('InternetService', df['InternetService'].unique())
    OnlineSecurity = st.sidebar.radio('OnlineSecurity', df['OnlineSecurity'].unique())
    OnlineBackup = st.sidebar.radio('OnlineBackup', df['OnlineBackup'].unique())
    DeviceProtection = st.sidebar.radio('DeviceProtection', df['DeviceProtection'].unique())
    TechSupport = st.sidebar.radio('TechSupport', df['TechSupport'].unique())
    StreamingTV = st.sidebar.radio('StreamingTV', df['StreamingTV'].unique())
    StreamingMovies = st.sidebar.radio('StreamingMovies', df['StreamingMovies'].unique())
    Contract = st.sidebar.radio('Contract', df['Contract'].unique())
    PaperlessBilling = st.sidebar.radio('PaperlessBilling', df['PaperlessBilling'].unique())
    PaymentMethod = st.sidebar.radio('PaymentMethod', df['PaymentMethod'].unique())
    MonthlyCharges = st.sidebar.number_input('MonthlyCharges',  min_value=0.0, value= 0.0)
    TotalCharges = st.sidebar.number_input('TotalCharges',  min_value=0.0, value= 0.0)



    data = {
        'gender' : gender,
        'SeniorCitizen' : SeniorCitizen,
        'Partner' : Partner,
        'Dependents' : Dependents,
        'tenure' : tenure,
        'PhoneService' : PhoneService,
        'MultipleLines' : MultipleLines,
        'InternetService': InternetService ,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup' : OnlineBackup,
        'DeviceProtection' : DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV' : StreamingTV,
        'StreamingMovies' : StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling' : PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges' : MonthlyCharges,
        'TotalCharges' : TotalCharges,
    }
    features = pd.DataFrame(data, index=[0])
    return features


input = user_input()

st.write('User Input')
st.write(input)


load_model = joblib.load("model_grid.pkl")
predictions = load_model.predict(input)

submit_button = st.button("Predict")

if submit_button:
        result = load_model.predict(input)

        updated_res = result.flatten().astype(float)
        if updated_res[0] == 1:
            st.success('This Customer tend to Churn')
        else:
            st.error('This Customer tend to Not Churn')