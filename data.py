import zipfile
from pathlib import Path

import pandas as pd

# -- Read in the data --
#
# Flights: BTS "Reporting Carrier On-Time Performance" monthly files, 2023-2025
# (https://www.transtats.bts.gov). Downloaded as zips into data/bts/.
# The first load combines them into one parquet file, which later loads take seconds to read.

DATA_DIR = Path(__file__).parent / "data"
BTS_DIR = DATA_DIR / "bts"
FLIGHTS_CACHE = DATA_DIR / "flights_2023_2025.parquet"

# BTS column -> name used throughout this project
COLUMNS = {
    "FlightDate": "FLIGHT_DATE",
    "Year": "YEAR",
    "Month": "MONTH",
    "DayofMonth": "DAY",
    "DayOfWeek": "DAY_OF_WEEK",          # 1 = Monday
    "Reporting_Airline": "AIRLINE",
    "Flight_Number_Reporting_Airline": "FLIGHT_NUMBER",
    "Tail_Number": "TAIL_NUMBER",
    "Origin": "ORIGIN_AIRPORT",
    "Dest": "DESTINATION_AIRPORT",
    "CRSDepTime": "SCHEDULED_DEPARTURE",  # HHMM, local time
    "CRSArrTime": "SCHEDULED_ARRIVAL",    # HHMM, local time
    "CRSElapsedTime": "SCHEDULED_TIME",
    "Distance": "DISTANCE",
    "DepDelay": "DEPARTURE_DELAY",
    "ArrDelay": "ARRIVAL_DELAY",
    "Cancelled": "CANCELLED",
    "Diverted": "DIVERTED",
}
# Not loaded on purpose: delay-cause columns (CarrierDelay, WeatherDelay, ...) and actual
# times. They're only known after the flight, so they would leak the answer.


def load_flights():
    if not FLIGHTS_CACHE.exists():
        build_flights_cache()
    return pd.read_parquet(FLIGHTS_CACHE)


def build_flights_cache():
    months = []
    for path in sorted(BTS_DIR.glob("ontime_*.zip")):
        print(f"Reading {path.name}", end="\r", flush=True)
        with zipfile.ZipFile(path) as z:
            csv_name = next(n for n in z.namelist() if n.endswith(".csv"))
            month = pd.read_csv(z.open(csv_name), usecols=list(COLUMNS),
                                dtype={"Tail_Number": str, "Origin": str, "Dest": str})
        months.append(month.rename(columns=COLUMNS))
    print()
    flights = pd.concat(months, ignore_index=True)
    flights["FLIGHT_DATE"] = pd.to_datetime(flights["FLIGHT_DATE"])
    for col in ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "TAIL_NUMBER"]:
        flights[col] = flights[col].astype("string")
    flights.to_parquet(FLIGHTS_CACHE, index=False)
    print(f"Saved {len(flights):,} flights to {FLIGHTS_CACHE.name}")
