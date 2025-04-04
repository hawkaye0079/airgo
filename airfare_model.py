# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 00:06:35 2025

@author: HARSHIT NARAIN
"""
import streamlit as st
import pandas as pd
import datetime
from amadeus import Client, ResponseError
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
import joblib

amadeus = Client(
    client_id="4TObrLiMnKERKFZACkNLFTwaPqPxVHkA",
    client_secret="ZqxP2IKd1n2v6ugT"
)

st.title("Flight Price Finder")
source = st.text_input("Enter Source Airport Code (e.g., JFK)").upper()
destination = st.text_input("Enter Destination Airport Code (e.g., LHR)").upper()
airline = st.text_input("Enter Airline Code (optional, e.g., AI, EK, QR)").upper()
date = st.date_input("Select departure date").strftime("%Y-%m-%d")

if st.button("Search Flights"):
    def get_flight_prices(source, destination, airline, date):
        try:
            response = amadeus.shopping.flight_offers_search.get(
                originLocationCode=source,
                destinationLocationCode=destination,
                departureDate=date,
                adults=1,
                currencyCode="USD"
            )
            return response.data
        except ResponseError as error:
            st.error(f"API Error: {error}")
            return None

    def parse_flight_data(flight_data):
        flights = []
        if not flight_data:
            return flights
        for flight in flight_data:
            for itinerary in flight.get("itineraries", []):
                flights.append({
                    "Price (USD)": flight["price"]["total"],
                    "Source": source,
                    "Destination": destination,
                    "Airline": airline,
                    "Departure": itinerary["segments"][0]["departure"]["at"]
                })
        return flights

    flight_data = get_flight_prices(source, destination, airline, date)
    parsed_flights = parse_flight_data(flight_data)

    if parsed_flights:
        df = pd.DataFrame(parsed_flights)
        st.dataframe(df)
        df.to_csv("airfare_dataset.csv", index=False)
        st.success("Flight data saved successfully!")
    else:
        st.warning("No flight data found.")

df = pd.read_csv("airfare_dataset.csv")
df["Departure"] = pd.to_datetime(df["Departure"])
df["Departure_Year"] = df["Departure"].dt.year
df["Departure_Month"] = df["Departure"].dt.month
df["Departure_Day"] = df["Departure"].dt.day
df.drop(columns=["Departure"], inplace=True)

X = df[["Source", "Destination", "Airline", "Departure_Year", "Departure_Month", "Departure_Day"]]
y = df["Price (USD)"]

model = joblib.load("airfare_model.joblib")

if st.button("Predict Fare"):
    input_data = pd.DataFrame([[source, destination, airline, date]],
                              columns=["Source", "Destination", "Airline", "Departure_Year", "Departure_Month", "Departure_Day"])
    try:
        prediction = model.predict(input_data)
        st.success(f"Predicted Airfare: ${prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Error: {e}")

categorical_features = ["Airline"]
numeric_features = ["Departure_Year", "Departure_Month", "Departure_Day"]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42))
])

model_pipeline.fit(X_train, y_train)

joblib.dump(model_pipeline, "airfare_model.joblib")
print("Model saved successfully!")

st.success("Model Training Complete! 🚀")
