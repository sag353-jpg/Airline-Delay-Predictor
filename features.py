from data import load_data

# -- Feature Engineering --

def prepare_features():
    flights, airports = load_data()

    # Convert AIRLINE into integer keys

    airline_ids = {
        "UA": 0,
        "AA": 1,
        "US": 2,
        "F9": 3,
        "B6": 4,
        "OO": 5,
        "AS": 6,
        "NK": 7,
        "WN": 8,
        "DL": 9,
        "EV": 10,
        "HA": 11,
        "MQ": 12,
        "VX": 13,
    }

    flights["AIRLINE"] = flights["AIRLINE"].map(airline_ids)

    # Convert location (states) into integers
    flights = flights.rename(columns={"ORIGIN_AIRPORT": "IATA_CODE"})
    flights = flights.merge(airports, on = "IATA_CODE")
    state_ids = {
        "AK": 0, "AL": 1, "AR": 2, "AZ": 3, "CA": 4, "CO": 5, "CT": 6,
        "DC": 7, "DE": 8, "FL": 9, "GA": 10, "HI": 11, "IA": 12, "ID": 13,
        "IL": 14, "IN": 15, "KS": 16, "KY": 17, "LA": 18, "MA": 19, "MD": 20,
        "ME": 21, "MI": 22, "MN": 23, "MO": 24, "MS": 25, "MT": 26, "NC": 27,
        "ND": 28, "NE": 29, "NH": 30, "NJ": 31, "NM": 32, "NV": 33, "NY": 34,
        "OH": 35, "OK": 36, "OR": 37, "PA": 38, "RI": 39, "SC": 40, "SD": 41,
        "TN": 42, "TX": 43, "UT": 44, "VA": 45, "VT": 46, "WA": 47, "WI": 48,
        "WV": 49, "WY": 50,
    }
    flights["STATE"] = flights["STATE"].map(state_ids)
    return flights
