# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 00:08:15 2025

@author: HARSHIT NARAIN
"""
import streamlit as st
import numpy as np
import joblib
import pandas as pd
import datetime

# Load trained model (which already includes the preprocessor)
model = joblib.load("airfare_model.joblib")

st.title("✈️ Airfare Price Prediction")
st.write("Enter your flight details below to predict the fare:")

# User input fields
airline = st.text_input("Enter Airline Code (e.g., AI, EK, QR)", "AI")
departure_date = st.date_input("Select Departure Date", datetime.date.today())

departure_year = departure_date.year
departure_month = departure_date.month
departure_day = departure_date.day

# Convert input to DataFrame (raw features, NOT encoded)
input_data = pd.DataFrame([[airline, departure_year, departure_month, departure_day]],
                          columns=["Airline", "Departure_Year", "Departure_Month", "Departure_Day"])

# Predict airfare using the model (which includes preprocessor)
if st.button("Predict Fare"):
    try:
        prediction = model.predict(input_data)
        st.success(f"Predicted Airfare: ${prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Error: {e}")
