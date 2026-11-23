import os
from datetime import datetime
from pathlib import Path

import certifi
import pandas as pd

# This Python install has no CA certificates of its own, so HTTPS downloads fail without this
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import meteostat as ms  # noqa: E402  (must come after the certificate fix)

# -- Historical weather (Meteostat / NOAA stations) --

CACHE_PATH = Path(__file__).parent / "weather_2015.csv.gz"

# Hourly readings we keep. Others (snow, gusts, visibility) are almost always empty for 2015.
PARAMETERS = {
    ms.Parameter.TEMP: "TEMP",   # air temperature, deg C
    ms.Parameter.RHUM: "RHUM",   # relative humidity, %
    ms.Parameter.PRCP: "PRCP",   # precipitation in that hour, mm
    ms.Parameter.WSPD: "WSPD",   # wind speed, km/h
    ms.Parameter.PRES: "PRES",   # sea-level pressure, hPa (drops ahead of storms)
    ms.Parameter.CLDC: "CLDC",   # cloud cover, oktas 0-8
}

# Data is in UTC, so pad the range to cover all of 2015 in every US timezone
START, END = datetime(2015, 1, 1), datetime(2016, 1, 2)

# airports.csv is missing coordinates for these three
MISSING_COORDS = {
    "ECP": (30.3577, -85.7956),  # Panama City, FL
    "PBG": (44.6509, -73.4681),  # Plattsburgh, NY
    "UST": (29.9592, -81.3398),  # St. Augustine, FL
}


def load_weather(airports):
    """Hourly weather per airport in local time: IATA_CODE, TIME, TEMP, RHUM, ...

    Downloads once (a few minutes) and then reads from the local cache.
    """
    if CACHE_PATH.exists():
        return pd.read_csv(CACHE_PATH, parse_dates=["TIME"])

    airports = airports.copy()
    for code, (lat, lon) in MISSING_COORDS.items():
        airports.loc[airports["IATA_CODE"] == code, ["LATITUDE", "LONGITUDE"]] = lat, lon

    frames = []
    for i, row in enumerate(airports.itertuples(), 1):
        print(f"Weather {i}/{len(airports)}: {row.IATA_CODE}", end="\r")
        frame = fetch_airport_weather(row.LATITUDE, row.LONGITUDE)
        if frame is not None:
            frame.insert(0, "IATA_CODE", row.IATA_CODE)
            frames.append(frame)
    print()

    weather = pd.concat(frames, ignore_index=True)
    weather.to_csv(CACHE_PATH, index=False)
    return weather


def fetch_airport_weather(lat, lon):
    stations = ms.stations.nearby(ms.Point(lat, lon), radius=50_000, limit=5)
    if stations.empty:
        return None

    raw = ms.hourly(list(stations.index), START, END, parameters=list(PARAMETERS)).fetch()
    if raw is None or raw.empty:
        return None
    raw = raw.rename(columns={p.value: name for p, name in PARAMETERS.items()}).astype(float)

    # Hour by hour, use the closest station that actually has a reading
    available = raw.index.get_level_values("station")
    hourly = None
    for station_id in stations.index:  # already sorted nearest first
        if station_id in available:
            readings = raw.xs(station_id, level="station")
            hourly = readings if hourly is None else hourly.combine_first(readings)

    # Flight times are local, so convert UTC to the airport's local time
    timezone = stations["timezone"].iloc[0]
    local = hourly.index.tz_localize("UTC").tz_convert(timezone).tz_localize(None)
    hourly = hourly.set_axis(local).rename_axis("TIME").reset_index()

    # The repeated hour when DST ends shows up twice locally; keep the first
    return hourly.drop_duplicates("TIME")


def add_weather_features(flights, airports, dates):
    """Weather at the origin at departure and at the destination at arrival."""
    weather = load_weather(airports)
    weather_cols = list(PARAMETERS.values())

    # Total precipitation that day: storms earlier in the day cause knock-on delays
    weather["DATE"] = weather["TIME"].dt.normalize()
    weather["DAY_PRCP"] = weather.groupby(["IATA_CODE", "DATE"])["PRCP"].transform("sum", min_count=1)
    weather = weather.drop(columns="DATE")
    weather_cols.append("DAY_PRCP")

    # Scheduled departure hour (local, origin). Hour 24 rolls over into the next day.
    dep_time = dates + pd.to_timedelta(flights["SCHEDULED_DEPARTURE"] // 100, unit="h")

    # Scheduled arrival hour (local, destination). Arrival before departure = overnight flight.
    overnight = flights["SCHEDULED_ARRIVAL"] < flights["SCHEDULED_DEPARTURE"]
    arr_time = (dates + pd.to_timedelta(overnight.astype(int), unit="D")
                + pd.to_timedelta(flights["SCHEDULED_ARRIVAL"] // 100, unit="h"))

    for prefix, airport_col, when in [("ORIGIN_", "IATA_CODE", dep_time),
                                      ("DEST_", "DESTINATION_AIRPORT", arr_time)]:
        lookup = weather.rename(columns={"IATA_CODE": airport_col, "TIME": "_TIME"})
        lookup = lookup.rename(columns={c: prefix + c for c in weather_cols})
        keys = flights[[airport_col]].assign(_TIME=when.values)
        matched = keys.merge(lookup, on=[airport_col, "_TIME"], how="left")
        for c in weather_cols:
            flights[prefix + c] = matched[prefix + c].values

    return flights


WEATHER_FEATURES = [p + c for p in ("ORIGIN_", "DEST_")
                    for c in list(PARAMETERS.values()) + ["DAY_PRCP"]]
