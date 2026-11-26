"""Predict delay risk for a single flight (used by app.py).

Lookup tables come from the BTS history: build them once with `python predict.py build`.
"""
import json
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import xgboost as xgb
from timezonefinder import TimezoneFinder

from features import days_to_holiday, hhmm_to_minutes
from weather import WEATHER_FEATURES, add_weather_features, airport_coords, fetch_forecast

MODELS_DIR = Path(__file__).parent / "models"
TABLES_DIR = MODELS_DIR / "tables"

# How many recent weeks of schedule to use for "typical" congestion and flight lookups
RECENT_WEEKS = 8


# ---------- Lookup tables (built once from history) ----------

def build_tables():
    from data import load_flights
    flights = load_flights()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    flights["DEP_HOUR"] = flights["SCHEDULED_DEPARTURE"] // 100
    flights["ARR_HOUR"] = flights["SCHEDULED_ARRIVAL"] // 100
    flights["delayed"] = flights["ARRIVAL_DELAY"] >= 15

    # Routes: typical scheduled duration and distance (for SCHEDULE_PADDING / DISTANCE),
    # plus the historical delay rate to compare against
    routes = (flights.groupby(["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"])
              .agg(MEDIAN_SCHEDULED_TIME=("SCHEDULED_TIME", "median"), DISTANCE=("DISTANCE", "median"),
                   DELAY_RATE=("delayed", "mean"), FLIGHTS=("delayed", "size")).reset_index())
    routes.to_parquet(TABLES_DIR / "routes.parquet", index=False)

    # Congestion: average scheduled traffic per airport, weekday and hour over the last weeks
    recent = flights[flights["FLIGHT_DATE"] > flights["FLIGHT_DATE"].max() - pd.Timedelta(weeks=RECENT_WEEKS)]

    def typical_count(airport_col, hour_col=None):
        keys = [airport_col, "FLIGHT_DATE", "DAY_OF_WEEK"] + ([hour_col] if hour_col else [])
        per_day = recent.groupby(keys).size().rename("n").reset_index()
        return per_day.groupby([k for k in keys if k != "FLIGHT_DATE"])["n"].mean().rename("COUNT").reset_index()

    typical_count("ORIGIN_AIRPORT", "DEP_HOUR").to_parquet(TABLES_DIR / "origin_hour_deps.parquet", index=False)
    typical_count("ORIGIN_AIRPORT").to_parquet(TABLES_DIR / "origin_day_deps.parquet", index=False)
    typical_count("DESTINATION_AIRPORT", "ARR_HOUR").to_parquet(TABLES_DIR / "dest_hour_arrs.parquet", index=False)

    # Flight numbers: the legs each airline + flight number flew recently, most common first
    legs = (recent.groupby(["AIRLINE", "FLIGHT_NUMBER", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT",
                            "SCHEDULED_DEPARTURE", "SCHEDULED_ARRIVAL"]).size().rename("TIMES").reset_index()
            .sort_values("TIMES", ascending=False))
    legs["FLIGHT_NUMBER"] = legs["FLIGHT_NUMBER"].astype(int)
    legs.to_parquet(TABLES_DIR / "flight_numbers.parquet", index=False)

    # Overall delay rate per airline (for context in the app)
    flights.groupby("AIRLINE")["delayed"].mean().rename("DELAY_RATE").reset_index().to_parquet(
        TABLES_DIR / "airlines.parquet", index=False)
    print(f"Saved lookup tables to {TABLES_DIR}")


class Predictor:
    def __init__(self, model_name="advance"):
        self.meta = json.loads((MODELS_DIR / f"{model_name}_meta.json").read_text())
        self.booster = xgb.Booster()
        self.booster.load_model(MODELS_DIR / f"{model_name}.json")
        self.routes = pd.read_parquet(TABLES_DIR / "routes.parquet").set_index(
            ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"])
        self.origin_hour = pd.read_parquet(TABLES_DIR / "origin_hour_deps.parquet").set_index(
            ["ORIGIN_AIRPORT", "DAY_OF_WEEK", "DEP_HOUR"])["COUNT"]
        self.origin_day = pd.read_parquet(TABLES_DIR / "origin_day_deps.parquet").set_index(
            ["ORIGIN_AIRPORT", "DAY_OF_WEEK"])["COUNT"]
        self.dest_hour = pd.read_parquet(TABLES_DIR / "dest_hour_arrs.parquet").set_index(
            ["DESTINATION_AIRPORT", "DAY_OF_WEEK", "ARR_HOUR"])["COUNT"]
        self.flight_numbers = pd.read_parquet(TABLES_DIR / "flight_numbers.parquet")
        self.airline_rates = pd.read_parquet(TABLES_DIR / "airlines.parquet").set_index("AIRLINE")["DELAY_RATE"]
        self.coords = airport_coords().set_index("IATA_CODE")
        self._tz = TimezoneFinder()

    # --- options for the app's dropdowns ---
    @property
    def airlines(self):
        return self.meta["categories"]["AIRLINE"]

    @property
    def airports(self):
        return self.meta["categories"]["ORIGIN"]

    def lookup_flight(self, airline, flight_number):
        """Recent legs flown under this flight number (most common first)."""
        f = self.flight_numbers
        return f[(f["AIRLINE"] == airline) & (f["FLIGHT_NUMBER"] == int(flight_number))].reset_index(drop=True)

    def route_info(self, origin, dest):
        try:
            return self.routes.loc[(origin, dest)]
        except KeyError:
            return None

    # --- the prediction ---
    def predict(self, airline, origin, dest, flight_date, dep_hhmm, arr_hhmm, weather=None):
        """Returns a dict with the delay probability, context, and the biggest factors.

        weather: optional DataFrame of hourly forecasts (IATA_CODE, TIME, ...); fetched if None.
        """
        flight_date = pd.Timestamp(flight_date)
        dow = flight_date.day_of_week + 1  # BTS: 1 = Monday
        row = {
            "AIRLINE": airline, "ORIGIN": origin, "DEST": dest,
            "ORIGIN_AIRPORT": origin, "DESTINATION_AIRPORT": dest,
            "SCHEDULED_DEPARTURE": dep_hhmm, "SCHEDULED_ARRIVAL": arr_hhmm,
            "DAY_OF_WEEK": dow, "MONTH": flight_date.month,
            "DEP_MINUTES": hhmm_to_minutes(dep_hhmm), "ARR_MINUTES": hhmm_to_minutes(arr_hhmm),
        }
        row["DAYS_TO_HOLIDAY"] = days_to_holiday(pd.Series([flight_date])).iloc[0]

        route = self.route_info(origin, dest)
        scheduled_time = self.scheduled_minutes(origin, dest, flight_date, dep_hhmm, arr_hhmm)
        row["DISTANCE"] = route["DISTANCE"] if route is not None else self.great_circle_miles(origin, dest)
        row["SCHEDULE_PADDING"] = (scheduled_time - route["MEDIAN_SCHEDULED_TIME"]
                                   if route is not None and scheduled_time is not None else np.nan)

        row["ORIGIN_HOUR_DEPS"] = self.origin_hour.get((origin, dow, dep_hhmm // 100), 1.0)
        row["ORIGIN_DAY_DEPS"] = self.origin_day.get((origin, dow), 1.0)
        row["DEST_HOUR_ARRS"] = self.dest_hour.get((dest, dow, arr_hhmm // 100), 1.0)

        frame = pd.DataFrame([row])
        if weather is None:
            weather = self.forecast_weather([origin, dest])
        if weather is not None and len(weather):
            frame = add_weather_features(frame, weather, pd.Series([flight_date.normalize()]))
        for col in WEATHER_FEATURES:
            if col not in frame:
                frame[col] = np.nan
        has_weather = frame[WEATHER_FEATURES].notna().any(axis=1).iloc[0]

        X = self.model_input(frame)
        dmatrix = xgb.DMatrix(X, enable_categorical=True)
        raw = float(self.booster.predict(dmatrix)[0])
        contribs = self.booster.predict(dmatrix, pred_contribs=True)[0]

        return {
            "probability": self.calibrate(raw),
            "route_rate": float(route["DELAY_RATE"]) if route is not None else None,
            "route_flights": int(route["FLIGHTS"]) if route is not None else 0,
            "airline_rate": float(self.airline_rates.get(airline, np.nan)),
            "has_weather": bool(has_weather),
            "factors": self.explain(contribs[:-1], origin, dest),
            "inputs": X.iloc[0].to_dict(),
        }

    def model_input(self, frame):
        X = frame[self.meta["features"]].copy()
        # Same category lists as training, so codes line up with what the model learned
        for col, cats in self.meta["categories"].items():
            X[col] = pd.Categorical(X[col], categories=cats)
        for col in X.columns:
            if col not in self.meta["categories"]:
                X[col] = X[col].astype("float32")
        return X

    def calibrate(self, p):
        """Undo the training-time up-weighting of delayed flights (divide the odds)."""
        odds = p / (1 - p) / self.meta["scale_pos_weight"]
        return odds / (1 + odds)

    # Feature -> plain-language group shown in the app
    def explain(self, contribs, origin, dest):
        groups = {}
        for feature, value in zip(self.meta["features"], contribs):
            groups.setdefault(self.factor_name(feature, origin, dest), 0.0)
            groups[self.factor_name(feature, origin, dest)] += float(value)
        return sorted(groups.items(), key=lambda kv: -abs(kv[1]))

    @staticmethod
    def factor_name(feature, origin, dest):
        if feature.startswith("ORIGIN_") and feature.split("_", 1)[1] in {
                c.split("_", 1)[1] for c in WEATHER_FEATURES}:
            return f"Weather in {origin}"
        if feature.startswith("DEST_") and feature.split("_", 1)[1] in {
                c.split("_", 1)[1] for c in WEATHER_FEATURES}:
            return f"Weather in {dest}"
        return {
            "AIRLINE": "Airline", "ORIGIN": f"Departing from {origin}", "DEST": f"Flying into {dest}",
            "DEP_MINUTES": "Time of day", "ARR_MINUTES": "Time of day",
            "DAY_OF_WEEK": "Day of the week", "MONTH": "Time of year",
            "DAYS_TO_HOLIDAY": "Holiday travel", "DISTANCE": "Route",
            "SCHEDULE_PADDING": "Schedule buffer",
            "ORIGIN_HOUR_DEPS": "Airport traffic", "ORIGIN_DAY_DEPS": "Airport traffic",
            "DEST_HOUR_ARRS": "Airport traffic",
        }.get(feature, "Live flight status")

    # --- helpers ---
    def timezone(self, airport):
        lat, lon = self.coords.loc[airport, ["LATITUDE", "LONGITUDE"]]
        return ZoneInfo(self._tz.timezone_at(lat=lat, lng=lon))

    def scheduled_minutes(self, origin, dest, flight_date, dep_hhmm, arr_hhmm):
        """Gate-to-gate minutes from local departure/arrival times in different time zones."""
        try:
            d = flight_date.date()
            dep = datetime.combine(d, datetime.min.time()).replace(tzinfo=self.timezone(origin))
            dep = dep + pd.Timedelta(minutes=hhmm_to_minutes(dep_hhmm))
            arr = datetime.combine(d, datetime.min.time()).replace(tzinfo=self.timezone(dest))
            arr = arr + pd.Timedelta(minutes=hhmm_to_minutes(arr_hhmm))
            minutes = (arr - dep).total_seconds() / 60
            return minutes + 1440 if minutes < 0 else minutes  # lands the next day
        except (KeyError, TypeError, ValueError):
            return None

    def great_circle_miles(self, origin, dest):
        lat1, lon1, lat2, lon2 = np.radians(
            [*self.coords.loc[origin, ["LATITUDE", "LONGITUDE"]], *self.coords.loc[dest, ["LATITUDE", "LONGITUDE"]]])
        a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
        return float(3958.8 * 2 * np.arcsin(np.sqrt(a)))

    def forecast_weather(self, airports):
        frames = []
        for code in dict.fromkeys(airports):
            if code in self.coords.index:
                lat, lon = self.coords.loc[code, ["LATITUDE", "LONGITUDE"]]
                frames.append(fetch_forecast(lat, lon).assign(IATA_CODE=code))
        return pd.concat(frames, ignore_index=True) if frames else None


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        build_tables()
    else:
        p = Predictor()
        print(p.predict("DL", "ATL", "LGA", date.today() + pd.Timedelta(days=2), 1730, 1940))
