# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 00:08:15 2025

@author: HARSHIT NARAIN
"""
import streamlit as st
import pandas as pd
import datetime
import joblib
from amadeus import Client, ResponseError

amadeus = Client(
    client_id="4TObrLiMnKERKFZACkNLFTwaPqPxVHkA",
    client_secret="ZqxP2IKd1n2v6ugT"
)

st.title("Flight Price Finder")
source = st.text_input("Enter Source Airport Code (e.g., JFK)").upper()
destination = st.text_input("Enter Destination Airport Code (e.g., LHR)").upper()
date = st.date_input("Select departure date").strftime("%Y-%m-%d")

if st.button("Search Flights"):
    def get_flight_prices(source, destination, date):
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
                    "Airline": flight["validatingAirlineCodes"][0] if "validatingAirlineCodes" in flight else "Unknown",
                    "Price (USD)": flight["price"]["total"],
                    "Source": source,
                    "Destination": destination,
                    "Departure": itinerary["segments"][0]["departure"]["at"]
                })
        return flights

    flight_data = get_flight_prices(source, destination, date)
    parsed_flights = parse_flight_data(flight_data)

    if parsed_flights:
        df = pd.DataFrame(parsed_flights)
        st.dataframe(df)
        df.to_csv("airfare_dataset.csv", index=False)
        st.success("Flight data saved successfully!")
    else:
        st.warning("No flight data found.")
