import kagglehub
import pandas as pd
import os

# -- Read in the data --

def load_data():
    path = kagglehub.dataset_download("usdot/flight-delays")

    # Access each individual csv in usdot/flights-delays
    flights_path = os.path.join(path, "flights.csv")
    airports_path = os.path.join(path, "airports.csv")
    airlines_path = os.path.join(path, "airlines.csv")

    # Assign each csv a dataframe. nrows for testing.
    flights = pd.read_csv(flights_path)
    airports = pd.read_csv(airports_path)
    flights = flights.drop(columns=["CANCELLATION_REASON", "AIR_SYSTEM_DELAY", "SECURITY_DELAY","AIRLINE_DELAY","LATE_AIRCRAFT_DELAY","WEATHER_DELAY"])
    return flights, airports