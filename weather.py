import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import certifi
import pandas as pd

# This Python install has no CA certificates of its own, so HTTPS downloads fail without this
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

# -- Weather (Open-Meteo) --
#
# Training uses Open-Meteo's archive of past *forecasts*, and the app uses its live forecast.
# Same source, same variables, so the model sees the same kind of numbers in both places.
# Times come back in each airport's local time, matching the BTS schedule times.

DATA_DIR = Path(__file__).parent / "data"
WEATHER_DIR = DATA_DIR / "weather"

HISTORY_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORY_START, HISTORY_END = "2023-01-01", "2025-12-31"

VARIABLES = {
    "temperature_2m": "TEMP",          # deg C
    "cape": "CAPE",                    # J/kg, thunderstorm energy (the forecast archive has
                                       # no thunderstorm weather codes, so this stands in)
    "precipitation": "PRCP",           # mm in that hour (rain + snow water)
    "snowfall": "SNOW",                # cm in that hour
    "wind_speed_10m": "WSPD",          # km/h
    "wind_gusts_10m": "GUST",          # km/h
    "cloud_cover": "CLDC",             # %
    "pressure_msl": "PRES",            # hPa (drops ahead of storms)
    "visibility": "VSBY",              # m
    "weather_code": "WCODE",           # WMO code: 0 clear ... 45 fog, 6x rain, 7x snow, 8x showers
}
HOURLY_COLS = list(VARIABLES.values())
DAILY_COLS = ["DAY_PRCP", "DAY_SNOW"]  # totals for the whole day (storms earlier cause knock-on delays)
WEATHER_FEATURES = [p + c for p in ("ORIGIN_", "DEST_") for c in HOURLY_COLS + DAILY_COLS]

# Free tier: ~10,000 "calls" per day and 5,000 per hour. Three years of hourly data for one
# airport counts as ~78 calls, so pace downloads to stay under the hourly limit.
SECONDS_PER_AIRPORT = 65

# OurAirports is missing the IATA code for this one
MISSING_COORDS = {"PBI": (26.6832, -80.0956)}


def airport_coords():
    """IATA_CODE, LATITUDE, LONGITUDE for every airport with an IATA code."""
    a = pd.read_csv(DATA_DIR / "ourairports.csv", usecols=["iata_code", "latitude_deg", "longitude_deg"])
    a = a.dropna(subset=["iata_code"]).drop_duplicates("iata_code")
    a.columns = ["LATITUDE", "LONGITUDE", "IATA_CODE"]
    extra = pd.DataFrame([(c, lat, lon) for c, (lat, lon) in MISSING_COORDS.items()],
                         columns=["IATA_CODE", "LATITUDE", "LONGITUDE"])
    return pd.concat([a[["IATA_CODE", "LATITUDE", "LONGITUDE"]], extra], ignore_index=True)


def _fetch(url, lat, lon, extra_params):
    params = (f"?latitude={lat}&longitude={lon}&hourly={','.join(VARIABLES)}"
              f"&timezone=auto&{extra_params}")
    with urllib.request.urlopen(url + params, timeout=180) as response:
        hourly = pd.DataFrame(json.load(response)["hourly"])
    hourly = hourly.rename(columns={"time": "TIME", **VARIABLES})
    hourly["TIME"] = pd.to_datetime(hourly["TIME"])
    # When DST ends, the repeated local hour shows up twice; keep the first
    return hourly.drop_duplicates("TIME")


def fetch_forecast(lat, lon):
    """Hourly forecast for the next 16 days in local time (used by the app)."""
    return _fetch(FORECAST_URL, lat, lon, "forecast_days=16")


def download_history(codes):
    """Download history for each airport code not already cached. Stops at the daily limit.

    Safe to re-run: airports already in data/weather/ are skipped.
    """
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    coords = airport_coords().set_index("IATA_CODE")
    todo = [c for c in codes if not (WEATHER_DIR / f"{c}.csv.gz").exists()]
    print(f"{len(codes) - len(todo)} airports cached, {len(todo)} to download", flush=True)

    for i, code in enumerate(todo, 1):
        if code not in coords.index:
            print(f"  {code}: no coordinates, skipping", flush=True)
            continue
        started = time.time()
        lat, lon = coords.loc[code, ["LATITUDE", "LONGITUDE"]]
        hourly = None
        for attempt in range(4):
            try:
                hourly = _fetch(HISTORY_URL, lat, lon, f"start_date={HISTORY_START}&end_date={HISTORY_END}")
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    print(f"Rate limit reached after {i - 1} airports. Run again later to continue.", flush=True)
                    return
                error = e
            except (TimeoutError, urllib.error.URLError, ConnectionError, http.client.HTTPException) as e:
                error = e
            time.sleep(30 * (attempt + 1))  # slow or flaky response: wait longer each time
        if hourly is None:
            print(f"  {code}: failed 4 times ({error}), skipping for now", flush=True)
            continue
        hourly.to_csv(WEATHER_DIR / f"{code}.csv.gz", index=False)
        print(f"  [{i}/{len(todo)}] {code}", flush=True)
        if i < len(todo):
            time.sleep(max(0, SECONDS_PER_AIRPORT - (time.time() - started)))
    print("All airports downloaded.", flush=True)


def load_weather_history():
    """All cached airport weather: IATA_CODE, TIME, TEMP, RHUM, ..."""
    frames = [pd.read_csv(path, parse_dates=["TIME"]).assign(IATA_CODE=path.name.split(".")[0])
              for path in sorted(WEATHER_DIR.glob("*.csv.gz"))]
    return pd.concat(frames, ignore_index=True)


def add_daily_totals(weather):
    date = weather["TIME"].dt.normalize()
    for hourly_col, daily_col in [("PRCP", "DAY_PRCP"), ("SNOW", "DAY_SNOW")]:
        weather[daily_col] = weather.groupby(["IATA_CODE", date])[hourly_col].transform("sum")
    return weather


def add_weather_features(flights, weather, dates):
    """Weather at the origin at departure and at the destination at arrival.

    flights needs ORIGIN_AIRPORT, DESTINATION_AIRPORT, SCHEDULED_DEPARTURE and
    SCHEDULED_ARRIVAL (HHMM, local). dates = each flight's date. weather = output of
    load_weather_history() or a forecast with IATA_CODE added. Airports with no weather get NaN.
    """
    weather = add_daily_totals(weather.copy())
    cols = HOURLY_COLS + DAILY_COLS

    # Scheduled departure hour (local, origin). Hour 24 rolls over into the next day.
    dep_time = dates + pd.to_timedelta(flights["SCHEDULED_DEPARTURE"] // 100, unit="h")
    # Scheduled arrival hour (local, destination). Arrival before departure = overnight flight.
    overnight = flights["SCHEDULED_ARRIVAL"] < flights["SCHEDULED_DEPARTURE"]
    arr_time = (dates + pd.to_timedelta(overnight.astype(int), unit="D")
                + pd.to_timedelta(flights["SCHEDULED_ARRIVAL"] // 100, unit="h"))

    for prefix, airport_col, when in [("ORIGIN_", "ORIGIN_AIRPORT", dep_time),
                                      ("DEST_", "DESTINATION_AIRPORT", arr_time)]:
        lookup = weather.rename(columns={"IATA_CODE": airport_col, "TIME": "_TIME",
                                         **{c: prefix + c for c in cols}})
        keys = pd.DataFrame({airport_col: flights[airport_col].to_numpy(), "_TIME": when.to_numpy()})
        matched = keys.merge(lookup[[airport_col, "_TIME"] + [prefix + c for c in cols]],
                             on=[airport_col, "_TIME"], how="left")
        for c in cols:
            flights[prefix + c] = matched[prefix + c].to_numpy()
    return flights


if __name__ == "__main__":
    # python weather.py  -> download history for all airports, busiest first
    traffic = pd.read_csv(DATA_DIR / "airport_traffic.csv")
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    download_history(traffic["IATA_CODE"].tolist()[:limit])
