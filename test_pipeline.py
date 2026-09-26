import datetime as dt

import pandas as pd
import pytest

import update
from cities import CITIES
from naqi import category, naqi, sub_index


def test_sub_index_breakpoints():
    assert sub_index("pm2_5", 30) == pytest.approx(50)
    assert sub_index("pm2_5", 45) == pytest.approx(75)
    assert sub_index("pm10", 250) == pytest.approx(200)
    assert sub_index("co", 1.5) == pytest.approx(75)
    assert sub_index("pm2_5", 10_000) == 500


def test_categories():
    assert category(50) == "Good"
    assert category(51) == "Satisfactory"
    assert category(250) == "Poor"
    assert category(450) == "Severe"


def test_naqi_dominant_and_minimum_pollutants():
    aqi, cat, dom = naqi({"pm2_5": 150, "pm10": 200, "no2": 30})
    assert dom == "PM2.5" and cat == "Very Poor" and 300 < aqi <= 400
    assert naqi({"no2": 30, "so2": 10, "o3": 20}) == (None, None, None)  # no PM
    assert naqi({"pm2_5": 20, "pm10": 30}) == (None, None, None)  # < 3 pollutants


def test_cities_unique_and_in_india():
    names = [c[0] for c in CITIES]
    assert len(names) == len(set(names)) == 50
    for _, _, _, lat, lon in CITIES:
        assert 6 <= lat <= 36 and 68 <= lon <= 98


def test_weight_matches_open_meteo_rules():
    d = dt.date(2025, 1, 1)
    assert update.weight(d, d + dt.timedelta(days=13), 10) == 1.0
    assert update.weight(d, d + dt.timedelta(days=27), 10) == 2.0
    assert update.weight(d, d + dt.timedelta(days=13), 15) == 1.5


@pytest.mark.network
def test_live_fetch_one_city_short_range(monkeypatch):
    monkeypatch.setattr(update, "CALLS_PER_HOUR", 10**9)  # no throttling in tests
    city = CITIES[0]
    end = dt.date.today() - dt.timedelta(days=2)
    w, elev = update.fetch_weather(city, end - dt.timedelta(days=6), end)
    assert len(w) == 7 and list(w.columns[:2]) == ["date", "city"] and elev is not None
    aq = update.fetch_air_quality(city, end - dt.timedelta(days=6), end)
    assert len(aq) == 7
    assert aq["india_aqi"].notna().all()
    assert set(aq["india_aqi_category"]) <= {"Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"}
    assert (aq["hours_available"] >= 18).all()
    print(pd.concat([w.head(3), aq.head(3)], axis=1).to_string())
