import numpy as np
import pandas as pd

from data import load_flights
from weather import add_weather_features, load_weather_history

# -- Feature Engineering --

FIRST_DAY = pd.Timestamp("2023-01-01")  # day 0 for absolute day/minute counters


def prepare_features():
    flights = load_flights()

    # Scheduled times are stored as HHMM (e.g. 1359), so 1259 -> 1300 looks like a jump of 41.
    # Convert to hour of day and minutes since midnight.
    flights["DEP_HOUR"] = flights["SCHEDULED_DEPARTURE"] // 100
    flights["DEP_MINUTES"] = hhmm_to_minutes(flights["SCHEDULED_DEPARTURE"])
    flights["ARR_MINUTES"] = hhmm_to_minutes(flights["SCHEDULED_ARRIVAL"])
    # Days since FIRST_DAY: a single counter for dates across all years
    flights["DAY_INDEX"] = (flights["FLIGHT_DATE"] - FIRST_DAY).dt.days

    # -- Aircraft rotation features --
    # Order each plane's flights within a day so we can look at its previous leg
    flights = flights.sort_values(["TAIL_NUMBER", "FLIGHT_DATE", "SCHEDULED_DEPARTURE"]).reset_index(drop=True)
    by_plane_day = flights.groupby(["TAIL_NUMBER", "FLIGHT_DATE"])

    # How many flights this plane has already flown today (0 = first flight of the day)
    flights["LEG_OF_DAY"] = by_plane_day.cumcount()

    # Minutes between the previous leg's scheduled arrival and this scheduled departure.
    # First leg of the day has no previous flight, so give it a full day (1440) of buffer.
    prev_sched_arrival = by_plane_day["ARR_MINUTES"].shift()
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

    # Flights with no tail number can't be linked to a previous leg
    no_tail = flights["TAIL_NUMBER"].isna()
    flights.loc[no_tail, ROTATION_FEATURES] = np.nan

    # -- Calendar features --
    flights["DAYS_TO_HOLIDAY"] = days_to_holiday(flights["FLIGHT_DATE"])

    # -- Schedule padding --
    # Airlines pad the schedule on routes they know run late. Compare this flight's
    # scheduled duration to the typical one for its route (positive = extra padding).
    route = ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
    route_median = flights.groupby(route)["SCHEDULED_TIME"].transform("median")
    flights["SCHEDULE_PADDING"] = flights["SCHEDULED_TIME"] - route_median

    # -- Airport congestion --
    # Counted from the schedule (cancelled flights included), so it is known in advance.
    # Scheduled departures from the origin in the same hour, and across the whole day
    dep_hour = ["ORIGIN_AIRPORT", "DAY_INDEX", "DEP_HOUR"]
    flights["ORIGIN_HOUR_DEPS"] = flights.groupby(dep_hour)["FLIGHT_NUMBER"].transform("size")
    flights["ORIGIN_DAY_DEPS"] = flights.groupby(["ORIGIN_AIRPORT", "DAY_INDEX"])["FLIGHT_NUMBER"].transform("size")

    # Scheduled arrivals into the destination in the same hour (overnight flights land the next day)
    overnight = (flights["SCHEDULED_ARRIVAL"] < flights["SCHEDULED_DEPARTURE"]).astype(int)
    flights["_ARR_DAY"] = flights["DAY_INDEX"] + overnight
    flights["_ARR_HOUR"] = flights["SCHEDULED_ARRIVAL"] // 100
    arr_hour = ["DESTINATION_AIRPORT", "_ARR_DAY", "_ARR_HOUR"]
    flights["DEST_HOUR_ARRS"] = flights.groupby(arr_hour)["FLIGHT_NUMBER"].transform("size")
    flights = flights.drop(columns=["_ARR_DAY", "_ARR_HOUR"])

    # -- Recent delays at the airports (what's happening right now) --
    # Catches ground stops, air traffic slowdowns, and staffing problems that weather misses.
    # Only flights that left/landed between 3h and 1h before this flight's scheduled departure,
    # so everything here is known 1 hour ahead (RECENT_WINDOW_START/END).
    # Times are minutes since FIRST_DAY so windows can cross midnight.
    sched_dep = flights["DAY_INDEX"] * 1440 + flights["DEP_MINUTES"]
    sched_arr = (flights["DAY_INDEX"] + overnight) * 1440 + flights["ARR_MINUTES"]
    actual_dep = sched_dep + flights["DEPARTURE_DELAY"]  # NaN if cancelled -> ignored
    actual_arr = sched_arr + flights["ARRIVAL_DELAY"]

    flights["ORIGIN_RECENT_DEP_DELAY"], flights["ORIGIN_RECENT_DEPS"] = recent_average(
        flights["ORIGIN_AIRPORT"], actual_dep, flights["DEPARTURE_DELAY"],
        flights["ORIGIN_AIRPORT"], sched_dep)
    flights["DEST_RECENT_ARR_DELAY"], _ = recent_average(
        flights["DESTINATION_AIRPORT"], actual_arr, flights["ARRIVAL_DELAY"],
        flights["DESTINATION_AIRPORT"], sched_dep)

    # -- Weather at origin (departure hour) and destination (arrival hour) --
    flights = add_weather_features(flights, load_weather_history(), flights["FLIGHT_DATE"])

    # -- Categorical features --
    # Mark these as categories so XGBoost treats them as labels, not numbers
    flights["ORIGIN"] = flights["ORIGIN_AIRPORT"].astype("category")
    flights["DEST"] = flights["DESTINATION_AIRPORT"].astype("category")
    flights["AIRLINE"] = flights["AIRLINE"].astype("category")

    # 21M rows: store numbers as 32-bit to halve memory (plenty of precision for these values)
    for col in flights.select_dtypes("float64").columns:
        flights[col] = flights[col].astype("float32")
    for col in flights.select_dtypes("int64").columns:
        flights[col] = flights[col].astype("int32")

    return flights


# Features that need the plane's tail number (only assigned close to the day of travel)
ROTATION_FEATURES = ["LEG_OF_DAY", "TURNAROUND_MIN", "PREV_ARRIVAL_DELAY", "PREV_DEPARTURE_DELAY",
                     "PREV_DISRUPTED", "INBOUND_SLACK"]

RECENT_WINDOW_START = 180  # minutes before scheduled departure
RECENT_WINDOW_END = 60     # = how far ahead the prediction is made


def major_holidays(years):
    """Major US travel holidays for the given years."""
    days = []
    for y in years:
        nov = pd.date_range(f"{y}-11-01", f"{y}-11-30", freq="W-THU")
        days += [
            pd.Timestamp(f"{y}-01-01"),                                        # New Year's Day
            pd.date_range(f"{y}-05-01", f"{y}-05-31", freq="W-MON")[-1],       # Memorial Day
            pd.Timestamp(f"{y}-07-04"),                                        # Independence Day
            pd.date_range(f"{y}-09-01", f"{y}-09-30", freq="W-MON")[0],        # Labor Day
            nov[3],                                                            # Thanksgiving
            pd.Timestamp(f"{y}-12-25"),                                        # Christmas
        ]
    return pd.DatetimeIndex(days)


def days_to_holiday(dates):
    """Days to the nearest major holiday (computed once per unique date, then mapped)."""
    unique_dates = pd.Series(pd.to_datetime(dates).unique())
    years = range(unique_dates.dt.year.min() - 1, unique_dates.dt.year.max() + 2)
    holidays = major_holidays(years).values
    days_away = unique_dates.apply(lambda d: np.abs((holidays - d.to_datetime64()) / np.timedelta64(1, "D")).min())
    return pd.to_datetime(dates).map(dict(zip(unique_dates, days_away)))


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
