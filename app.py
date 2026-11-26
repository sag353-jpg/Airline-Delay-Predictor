"""Flight delay predictor. Run with:  streamlit run app.py"""
from datetime import date, time, timedelta

import pandas as pd
import streamlit as st

from predict import Predictor

st.set_page_config(page_title="Will my flight be delayed?", page_icon="✈️", layout="centered")

AIRLINE_NAMES = {
    "9E": "Endeavor Air (Delta Connection)", "AA": "American Airlines", "AS": "Alaska Airlines",
    "B6": "JetBlue", "DL": "Delta Air Lines", "F9": "Frontier", "G4": "Allegiant Air",
    "HA": "Hawaiian Airlines", "MQ": "Envoy Air (American Eagle)", "NK": "Spirit Airlines",
    "OH": "PSA Airlines (American Eagle)", "OO": "SkyWest", "UA": "United Airlines",
    "WN": "Southwest Airlines", "YX": "Republic Airways",
}
FORECAST_DAYS = 16  # Open-Meteo forecast horizon


@st.cache_resource
def get_predictor():
    return Predictor("advance")


@st.cache_data(ttl=3600, show_spinner=False)
def get_forecast(airports):
    # Cached for an hour per airport pair, so repeat lookups don't re-download
    return get_predictor().forecast_weather(list(airports))


def hhmm(t):
    return t.hour * 100 + t.minute


def to_time(hhmm_value):
    hhmm_value = int(hhmm_value) % 2400
    return time(hhmm_value // 100, hhmm_value % 100)


predictor = get_predictor()

st.title("✈️ Will my flight be delayed?")
st.caption("Predicts the chance your flight arrives 15+ minutes late, based on 21 million US "
           "flights (2023–2025), airport traffic, and the weather forecast.")

# ---------- Optional: fill in details from a flight number ----------
airlines = predictor.airlines
airline = st.selectbox("Airline", airlines, format_func=lambda c: f"{AIRLINE_NAMES.get(c, c)} ({c})",
                       index=airlines.index("DL") if "DL" in airlines else 0)

col_num, col_find = st.columns([3, 1])
flight_number = col_num.text_input("Flight number (optional)", placeholder="e.g. 1234",
                                   help="Fills in the route and times this flight usually has. "
                                        "Check them against your ticket.")
defaults = st.session_state.setdefault("defaults", {})
if col_find.button("Look up", use_container_width=True) and flight_number.strip().isdigit():
    legs = predictor.lookup_flight(airline, flight_number.strip())
    if legs.empty:
        st.warning(f"No recent {airline} {flight_number} flights found. Enter the details below.")
    else:
        st.session_state["legs"] = legs
        st.session_state.pop("leg_choice", None)

legs = st.session_state.get("legs")
if legs is not None and not legs.empty:
    labels = [f"{r.ORIGIN_AIRPORT} → {r.DESTINATION_AIRPORT}, departs {to_time(r.SCHEDULED_DEPARTURE):%H:%M}"
              for r in legs.itertuples()]
    choice = st.radio("Which leg?", range(len(labels)), format_func=labels.__getitem__, key="leg_choice") \
        if len(labels) > 1 else 0
    leg = legs.iloc[choice]
    defaults.update(origin=leg.ORIGIN_AIRPORT, dest=leg.DESTINATION_AIRPORT,
                    dep=to_time(leg.SCHEDULED_DEPARTURE), arr=to_time(leg.SCHEDULED_ARRIVAL))

# ---------- Flight details ----------
airports = predictor.airports


def airport_index(code, fallback):
    return airports.index(code) if code in airports else airports.index(fallback)


c1, c2 = st.columns(2)
origin = c1.selectbox("From", airports, index=airport_index(defaults.get("origin"), "ATL"))
dest = c2.selectbox("To", airports, index=airport_index(defaults.get("dest"), "LGA"))
flight_date = st.date_input("Date", value=date.today() + timedelta(days=1), min_value=date.today())
c3, c4 = st.columns(2)
dep = c3.time_input("Scheduled departure (local time)", value=defaults.get("dep", time(17, 30)), step=300)
arr = c4.time_input("Scheduled arrival (local time)", value=defaults.get("arr", time(19, 40)), step=300)

if origin == dest:
    st.error("Origin and destination must be different.")
    st.stop()

# ---------- Prediction ----------
if st.button("Predict", type="primary", use_container_width=True):
    days_out = (flight_date - date.today()).days
    weather = None
    if days_out < FORECAST_DAYS:
        with st.spinner("Getting the weather forecast…"):
            try:
                weather = get_forecast((origin, dest))
            except Exception:
                st.warning("Couldn't reach the weather service; predicting without weather.")
    result = predictor.predict(airline, origin, dest, flight_date, hhmm(dep), hhmm(arr),
                               weather=weather if weather is not None else pd.DataFrame())
    p = result["probability"]
    typical = result["route_rate"] or result["airline_rate"]

    if p < 0.15:
        verdict, color = "Low risk", "green"
    elif p < 0.30:
        verdict, color = "Moderate risk", "orange"
    else:
        verdict, color = "High risk", "red"

    st.subheader(f":{color}[{verdict} of delay]")
    m1, m2 = st.columns(2)
    m1.metric("Chance of arriving 15+ min late", f"{p:.0%}")
    if typical:
        label = (f"Typical for {origin}→{dest}" if result["route_rate"] is not None
                 else f"Typical for {AIRLINE_NAMES.get(airline, airline)}")
        m2.metric(label, f"{typical:.0%}", delta=f"{(p - typical) * 100:+.0f} pts", delta_color="inverse")

    if not result["has_weather"]:
        st.info("No weather forecast is available for this date yet "
                f"(forecasts reach {FORECAST_DAYS} days ahead), so this uses the schedule and "
                "season only. Check again closer to the trip.")
    if result["route_flights"] == 0:
        st.info(f"No {origin}→{dest} flights in our 2023–2025 data, so this estimate is rougher.")

    # What pushed the prediction up or down (model contributions, grouped)
    factors = [(name, v) for name, v in result["factors"] if abs(v) >= 0.05][:5]
    if factors:
        st.markdown("**What's driving this**")
        for name, value in factors:
            arrow = "🔺 raises" if value > 0 else "🔻 lowers"
            st.markdown(f"- **{name}** {arrow} the risk")

    st.caption("Estimates from historical patterns, not a guarantee. Doesn't predict "
               "cancellations, and doesn't yet use live flight status.")
