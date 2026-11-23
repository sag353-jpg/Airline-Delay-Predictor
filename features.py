import numpy as np
import pandas as pd

from data import load_data
from weather import add_weather_features

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

    # Attach airport info (coordinates for weather). Inner join also drops the few
    # October airport IDs that couldn't be mapped to a code.
    flights = flights.rename(columns={"ORIGIN_AIRPORT": "IATA_CODE"})
    flights = flights.merge(airports, on = "IATA_CODE")

    # Scheduled times are stored as HHMM (e.g. 1359), so 1259 -> 1300 looks like a jump of 41.
    # Convert to hour of day and minutes since midnight.
    flights["DEP_HOUR"] = flights["SCHEDULED_DEPARTURE"] // 100
    flights["DEP_MINUTES"] = hhmm_to_minutes(flights["SCHEDULED_DEPARTURE"])

    # -- Aircraft rotation features --
    # Order each plane's flights within a day so we can look at its previous leg
    flights = flights.sort_values(["TAIL_NUMBER", "MONTH", "DAY", "SCHEDULED_DEPARTURE"])
    by_plane_day = flights.groupby(["TAIL_NUMBER", "MONTH", "DAY"])

    # How many flights this plane has already flown today (0 = first flight of the day)
    flights["LEG_OF_DAY"] = by_plane_day.cumcount()

    # Minutes between the previous leg's scheduled arrival and this scheduled departure.
    # First leg of the day has no previous flight, so give it a full day (1440) of buffer.
    prev_sched_arrival = hhmm_to_minutes(by_plane_day["SCHEDULED_ARRIVAL"].shift())
    flights["TURNAROUND_MIN"] = (flights["DEP_MINUTES"] - prev_sched_arrival).fillna(1440)

    # Previous leg's actual arrival delay. Only valid if predicting AFTER that leg lands
    # (not at booking time). 0 for the first flight of the day.
    flights["PREV_ARRIVAL_DELAY"] = by_plane_day["ARRIVAL_DELAY"].shift().fillna(0)

    # Previous leg's departure delay (known earlier than its arrival delay)
    flights["PREV_DEPARTURE_DELAY"] = by_plane_day["DEPARTURE_DELAY"].shift().fillna(0)

    # Previous leg was cancelled or diverted. Its delays show up as 0 above, which would
    # otherwise look the same as "on time".
    prev_cancelled = by_plane_day["CANCELLED"].shift().fillna(0)
    prev_diverted = by_plane_day["DIVERTED"].shift().fillna(0)
    flights["PREV_DISRUPTED"] = ((prev_cancelled + prev_diverted) > 0).astype(int)

    # Ground time actually left once the plane gets in: scheduled turnaround minus how late
    # it arrived. Small or negative = the late arrival will push this departure late too.
    flights["INBOUND_SLACK"] = flights["TURNAROUND_MIN"] - flights["PREV_ARRIVAL_DELAY"]

    # -- Calendar features --
    dates = pd.to_datetime(flights[["YEAR", "MONTH", "DAY"]])
    flights["DAY_OF_YEAR"] = dates.dt.dayofyear

    # Days to the nearest major travel holiday (computed once per unique date, then mapped)
    unique_dates = pd.Series(dates.unique())
    holidays = pd.to_datetime(MAJOR_HOLIDAYS)
    days_away = unique_dates.apply(lambda d: abs((holidays - d).days).min())
    flights["DAYS_TO_HOLIDAY"] = dates.map(dict(zip(unique_dates, days_away)))

    # -- Schedule padding --
    # Airlines pad the schedule on routes they know run late. Compare this flight's
    # scheduled duration to the typical one for its route (positive = extra padding).
    route = ["IATA_CODE", "DESTINATION_AIRPORT"]
    route_median = flights.groupby(route)["SCHEDULED_TIME"].transform("median")
    flights["SCHEDULE_PADDING"] = flights["SCHEDULED_TIME"] - route_median

    # -- Airport congestion --
    # Counted from the schedule (cancelled flights included), so it is known in advance.
    # Scheduled departures from the origin in the same hour, and across the whole day
    dep_hour = ["IATA_CODE", "DAY_OF_YEAR", "DEP_HOUR"]
    flights["ORIGIN_HOUR_DEPS"] = flights.groupby(dep_hour)["FLIGHT_NUMBER"].transform("size")
    flights["ORIGIN_DAY_DEPS"] = flights.groupby(["IATA_CODE", "DAY_OF_YEAR"])["FLIGHT_NUMBER"].transform("size")

    # Scheduled arrivals into the destination in the same hour (overnight flights land the next day)
    overnight = (flights["SCHEDULED_ARRIVAL"] < flights["SCHEDULED_DEPARTURE"]).astype(int)
    flights["_ARR_DAY"] = flights["DAY_OF_YEAR"] + overnight
    flights["_ARR_HOUR"] = flights["SCHEDULED_ARRIVAL"] // 100
    arr_hour = ["DESTINATION_AIRPORT", "_ARR_DAY", "_ARR_HOUR"]
    flights["DEST_HOUR_ARRS"] = flights.groupby(arr_hour)["FLIGHT_NUMBER"].transform("size")
    flights = flights.drop(columns=["_ARR_DAY", "_ARR_HOUR"])

    # -- Recent delays at the airports (what's happening right now) --
    # Catches ground stops, air traffic slowdowns, and staffing problems that weather misses.
    # Only flights that left/landed between 3h and 1h before this flight's scheduled departure,
    # so everything here is known 1 hour ahead (RECENT_WINDOW_START/END).
    # Times are minutes since Jan 1 so windows can cross midnight.
    sched_dep = (flights["DAY_OF_YEAR"] - 1) * 1440 + flights["DEP_MINUTES"]
    sched_arr = ((flights["DAY_OF_YEAR"] - 1 + overnight) * 1440
                 + hhmm_to_minutes(flights["SCHEDULED_ARRIVAL"]))
    actual_dep = sched_dep + flights["DEPARTURE_DELAY"]  # NaN if cancelled -> ignored
    actual_arr = sched_arr + flights["ARRIVAL_DELAY"]

    flights["ORIGIN_RECENT_DEP_DELAY"], flights["ORIGIN_RECENT_DEPS"] = recent_average(
        flights["IATA_CODE"], actual_dep, flights["DEPARTURE_DELAY"],
        flights["IATA_CODE"], sched_dep)
    flights["DEST_RECENT_ARR_DELAY"], _ = recent_average(
        flights["DESTINATION_AIRPORT"], actual_arr, flights["ARRIVAL_DELAY"],
        flights["DESTINATION_AIRPORT"], sched_dep)

    # -- Weather at origin (departure hour) and destination (arrival hour) --
    flights = add_weather_features(flights, airports, dates)

    # -- Categorical features --
    # Mark these as categories so XGBoost treats them as labels, not numbers
    # (otherwise airline 8 counts as "bigger" than airline 3, which means nothing)
    flights["ORIGIN"] = flights["IATA_CODE"].astype("category")
    flights["DEST"] = flights["DESTINATION_AIRPORT"].astype("category")
    # (.map() above leaves float IDs; XGBoost needs integer or string categories)
    flights["AIRLINE"] = flights["AIRLINE"].astype("Int64").astype("category")

    return flights


# Major US travel holidays around the 2015 data (includes New Year's 2016 for late December)
MAJOR_HOLIDAYS = [
    "2015-01-01",  # New Year's Day
    "2015-05-25",  # Memorial Day
    "2015-07-04",  # Independence Day
    "2015-09-07",  # Labor Day
    "2015-11-26",  # Thanksgiving
    "2015-12-25",  # Christmas
    "2016-01-01",  # New Year's Day
]


RECENT_WINDOW_START = 180  # minutes before scheduled departure
RECENT_WINDOW_END = 60     # = how far ahead the prediction is made

# Features that can legitimately be missing (XGBoost handles NaN), so train.py keeps those rows.
# Recent-delay averages are NaN when nothing left/landed at that airport in the window.
NULLABLE_FEATURES = ["ORIGIN_RECENT_DEP_DELAY", "DEST_RECENT_ARR_DELAY"]


def recent_average(event_airport, event_time, event_value, query_airport, query_time):
    """For each query, average event_value over events at the same airport whose
    event_time falls in [query_time - RECENT_WINDOW_START, query_time - RECENT_WINDOW_END).

    Returns (average, count) arrays aligned with the query rows. Average is NaN if count is 0.
    """
    events = pd.DataFrame({"airport": event_airport.to_numpy(), "t": event_time.to_numpy(),
                           "v": event_value.to_numpy()}).dropna()
    events_by_airport = {a: g.sort_values("t") for a, g in events.groupby("airport")}

    query_t = query_time.to_numpy()
    average = np.full(len(query_t), np.nan)
    count = np.zeros(len(query_t), dtype=int)
    for airport, rows in pd.Series(query_airport.to_numpy()).groupby(query_airport.to_numpy()).indices.items():
        g = events_by_airport.get(airport)
        if g is None:
            continue
        t = g["t"].to_numpy()
        running_total = np.concatenate([[0.0], np.cumsum(g["v"].to_numpy())])
        lo = np.searchsorted(t, query_t[rows] - RECENT_WINDOW_START)
        hi = np.searchsorted(t, query_t[rows] - RECENT_WINDOW_END)
        n = hi - lo
        count[rows] = n
        average[rows] = np.where(n > 0, (running_total[hi] - running_total[lo]) / np.maximum(n, 1), np.nan)
    return average, count


def hhmm_to_minutes(hhmm):
    return (hhmm // 100) * 60 + hhmm % 100
