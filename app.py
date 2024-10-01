import streamlit as st
from utils import PrepProcesor, columns 

import numpy as np
import pandas as pd
import joblib

# Load the model
model = joblib.load('xgbpipe.joblib')

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #e9ecef;
        font-family: 'Arial', sans-serif;
    }
    .title {
        text-align: center;
        color: #343a40;
        font-size: 40px;
        margin: 20px 0;
        font-weight: bold;
    }
    .input-label {
        font-size: 16px;
        color: #495057;
        margin-bottom: 5px;
    }
    .stButton {
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 16px;
        transition: background-color 0.3s ease;
    }
    .stButton:hover {
        background-color: #0056b3;
    }
    .input-container {
        margin: 20px 0;
        padding: 10px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .result {
        text-align: center;
        font-size: 18px;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True
)

# Title
st.markdown('<h1 class="title">Did They Survive? 🚢</h1>', unsafe_allow_html=True)

# Input fields in a styled container
with st.container():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)

    passengerid = st.text_input("Input Passenger ID", '123456') 
    pclass = st.selectbox("Choose Class", [1, 2, 3])
    name  = st.text_input("Input Passenger Name", 'John Smith')
    sex = st.selectbox("Choose Sex", ['male', 'female'])
    age = st.slider("Choose Age", 0, 100)
    sibsp = st.slider("Choose Siblings", 0, 10)
    parch = st.slider("Choose Parch", 0, 2)
    ticket = st.text_input("Input Ticket Number", "12345") 
    fare = st.number_input("Input Fare Price", 0, 1000)
    cabin = st.text_input("Input Cabin", "C52") 
    embarked = st.selectbox("Did They Embark?", ['S', 'C', 'Q'])

    st.markdown('</div>', unsafe_allow_html=True)

def predict(): 
    row = np.array([passengerid, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked]) 
    X = pd.DataFrame([row], columns=columns)
    prediction = model.predict(X)
    
    if prediction[0] == 1: 
        st.success('Passenger Survived! 🎉', icon="✅")
    else: 
        st.error('Passenger Did Not Survive. 😢', icon="❌")

trigger = st.button('Predict', on_click=predict, key="predict_button")
