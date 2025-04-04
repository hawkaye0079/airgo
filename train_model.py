# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 00:15:28 2025

@author: HARSHIT NARAIN
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

data = pd.read_csv("airfare_dataset.csv")

data["Departure"] = pd.to_datetime(data["Departure"], errors='coerce')
data.dropna(subset=["Departure"], inplace=True)

data["Departure_Year"] = data["Departure"].dt.year
data["Departure_Month"] = data["Departure"].dt.month
data["Departure_Day"] = data["Departure"].dt.day

data.drop(columns=["Departure", "Arrival", "Stops"], inplace=True)

features = ["Airline", "Departure_Year", "Departure_Month", "Departure_Day"]
target = "Price (USD)"
X = data[features]
y = data[target]

categorical_features = ["Airline"]
numeric_features = ["Departure_Year", "Departure_Month", "Departure_Day"]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
])

model.fit(X_train, y_train)

joblib.dump(model, "airfare_model.joblib")
joblib.dump(preprocessor, "preprocessor.joblib")

print("Model and preprocessor saved successfully!")

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse) 
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")
