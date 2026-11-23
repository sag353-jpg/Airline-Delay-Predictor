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
    flights = pd.read_csv(flights_path, dtype={"ORIGIN_AIRPORT": str, "DESTINATION_AIRPORT": str})
    airports = pd.read_csv(airports_path)
    flights = flights.drop(columns=["CANCELLATION_REASON", "AIR_SYSTEM_DELAY", "SECURITY_DELAY","AIRLINE_DELAY","LATE_AIRCRAFT_DELAY","WEATHER_DELAY"])
    flights = fix_october_airports(flights)
    return flights, airports


def fix_october_airports(flights):
    """October 2015 uses numeric BTS airport IDs (e.g. 10397) instead of IATA codes (ATL).

    Recover the codes from the data: the same airline + flight number + scheduled time
    almost always flies the same route in September and November, so each numeric ID
    gets the IATA code it lines up with most often.
    """
    key = ["AIRLINE", "FLIGHT_NUMBER", "SCHEDULED_DEPARTURE"]
    october = flights.loc[flights["MONTH"] == 10, key + ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]]
    reference = (flights.loc[flights["MONTH"].isin([9, 11]), key + ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]]
                 .drop_duplicates())
    matched = october.merge(reference, on=key, suffixes=("", "_IATA"))

    votes = pd.concat([
        matched[["ORIGIN_AIRPORT", "ORIGIN_AIRPORT_IATA"]].set_axis(["ID", "IATA"], axis=1),
        matched[["DESTINATION_AIRPORT", "DESTINATION_AIRPORT_IATA"]].set_axis(["ID", "IATA"], axis=1),
    ])
    counts = votes.value_counts().rename("n").reset_index()
    best = counts.sort_values("n", ascending=False).drop_duplicates("ID")
    agreement = best["n"] / best["ID"].map(counts.groupby("ID")["n"].sum())
    best = best[agreement >= 0.8]  # only trust clear winners (in practice ~99% agree)
    id_to_iata = dict(zip(best["ID"], best["IATA"]))

    for col in ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]:
        flights[col] = flights[col].replace(id_to_iata)
    # The handful of IDs that couldn't be mapped stay numeric and get dropped by the airport merge
    return flights
