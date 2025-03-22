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

# -------------------- Step 1: Fetch Flight Data from Amadeus API --------------------

amadeus = Client(
    client_id="4TObrLiMnKERKFZACkNLFTwaPqPxVHkA",
    client_secret="ZqxP2IKd1n2v6ugT"
)

st.title("Flight Price Finder")
origin = st.text_input("Enter origin airport code (e.g., JFK)").upper()
destination = st.text_input("Enter destination airport code (e.g., LHR)").upper()
date = st.date_input("Select departure date").strftime("%Y-%m-%d")

if st.button("Search Flights"):
    def get_flight_prices(origin, destination, date):
        try:
            response = amadeus.shopping.flight_offers_search.get(
                originLocationCode=origin,
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
                    "Airline": flight["validatingAirlineCodes"][0],
                    "Departure": itinerary["segments"][0]["departure"]["at"]
                })
        return flights

    flight_data = get_flight_prices(origin, destination, date)
    parsed_flights = parse_flight_data(flight_data)

    if parsed_flights:
        df = pd.DataFrame(parsed_flights)
        st.dataframe(df)
        df.to_csv("airfare_dataset.csv", index=False)
        st.success("Flight data saved successfully!")
    else:
        st.warning("No flight data found.")

# -------------------- Step 2: Load & Process Dataset --------------------

df = pd.read_csv("airfare_dataset.csv")

# Convert datetime columns
df["Departure"] = pd.to_datetime(df["Departure"])

# Extract relevant date-based features
df["Departure_Year"] = df["Departure"].dt.year
df["Departure_Month"] = df["Departure"].dt.month
df["Departure_Day"] = df["Departure"].dt.day

# Drop unused columns
df.drop(columns=["Departure"], inplace=True)

# Define Features and Target Variable
X = df[["Airline", "Departure_Year", "Departure_Month", "Departure_Day"]]  # Features
y = df["Price (USD)"]  # Target variable

# -------------------- Step 3: Preprocessing --------------------

categorical_features = ["Airline"]
numeric_features = ["Departure_Year", "Departure_Month", "Departure_Day"]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),  # Ignore unknown airlines
        ('num', StandardScaler(), numeric_features)
    ]
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------- Step 4: Train Machine Learning Models --------------------

# Define model pipeline
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42))
])

# Train model
model_pipeline.fit(X_train, y_train)

# -------------------- Step 5: Save the Trained Model --------------------

# Save the trained model and preprocessor
joblib.dump(model_pipeline, "airfare_model.joblib")
print("Model saved successfully!")

st.success("Model Training Complete! 🚀")