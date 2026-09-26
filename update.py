"""Build or update the India daily weather + air quality dataset from Open-Meteo.

    python update.py            # update data/ in place (creates it on first run)

Each run:
  * re-fetches the last REFRESH_DAYS days (recent values are provisional) and adds new days,
  * extends weather history backwards by up to BACKFILL_DAYS until WEATHER_START,
  * validates the result and rewrites data/*.csv and data/metadata.json.

Stays within Open-Meteo's free tier (weighted calls: days/14 x variables/10 per request).
"""

from __future__ import annotations

import datetime as dt
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from cities import CITIES
from naqi import category, naqi, sub_index

DATA = Path("data")
TZ = "Asia/Kolkata"
WEATHER_START = dt.date(2020, 1, 1)
AQ_START = dt.date(2022, 8, 5)  # CAMS global history for India starts 2022-08-04
INITIAL_WEATHER_START = dt.date(2024, 1, 1)  # first build; older years are backfilled daily
REFRESH_DAYS = 10
BACKFILL_DAYS = 366
CALLS_PER_HOUR = 4500  # free tier allows 5000

WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "precipitation_sum",
    "precipitation_hours",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
    "weather_code",
]
AQ_VARS = [
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
    "dust",
    "aerosol_optical_depth",
]

session = requests.Session()
session.headers["User-Agent"] = "india-weather-aqi-dataset (non-commercial, CC BY 4.0)"
calls_used = 0.0


def weight(start: dt.date, end: dt.date, n_vars: int) -> float:
    days = (end - start).days + 1
    return max(1.0, days / 14) * max(1.0, n_vars / 10)


def get(url: str, params: dict, w: float) -> dict:
    global calls_used
    for attempt in range(8):
        try:
            r = session.get(url, params=params, timeout=120)
        except (requests.ConnectionError, requests.Timeout) as e:
            print(f"    connection problem ({e.__class__.__name__}); retrying", flush=True)
            time.sleep(15 * (attempt + 1))
            continue
        if r.status_code == 429 or (r.status_code == 400 and "limit" in r.text.lower()):
            reason = r.json().get("reason", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text
            if "daily" in reason.lower():
                sys.exit(f"Open-Meteo daily limit reached: {reason}. Re-run tomorrow; progress so far is saved.")
            wait = 3600 if "hourly" in reason.lower() else 65
            print(f"    rate limited ({reason}); sleeping {wait}s", flush=True)
            time.sleep(wait)
            continue
        if r.status_code >= 500:
            time.sleep(10 * (attempt + 1))
            continue
        r.raise_for_status()
        calls_used += w
        time.sleep(w * 3600 / CALLS_PER_HOUR)  # spread load under the hourly limit
        return r.json()
    raise RuntimeError(f"giving up on {url} {params}")


def fetch_weather(city: tuple, start: dt.date, end: dt.date) -> tuple[pd.DataFrame, float]:
    name, _, _, lat, lon = city
    j = get(
        WEATHER_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "daily": ",".join(WEATHER_VARS),
            "timezone": TZ,
        },
        weight(start, end, len(WEATHER_VARS)),
    )
    df = pd.DataFrame(j["daily"]).rename(columns={"time": "date"})
    df.insert(1, "city", name)
    return df, j.get("elevation")


def fetch_air_quality(city: tuple, start: dt.date, end: dt.date) -> pd.DataFrame:
    name, _, _, lat, lon = city
    # One extra day before `start` so 8-hour rolling windows are complete at midnight.
    fetch_start = start - dt.timedelta(days=1)
    j = get(
        AQ_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "start_date": fetch_start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": ",".join(AQ_VARS),
            "timezone": TZ,
        },
        weight(fetch_start, end, len(AQ_VARS)),
    )
    h = pd.DataFrame(j["hourly"])
    h["date"] = h["time"].str[:10]
    o3_8h = h["ozone"].rolling(8, min_periods=6).mean()
    co_8h = h["carbon_monoxide"].rolling(8, min_periods=6).mean()
    h = h.assign(o3_8h=o3_8h, co_8h=co_8h)
    g = h.groupby("date")
    daily = pd.DataFrame(
        {
            "pm2_5_mean": g["pm2_5"].mean(),
            "pm2_5_max": g["pm2_5"].max(),
            "pm10_mean": g["pm10"].mean(),
            "pm10_max": g["pm10"].max(),
            "no2_mean": g["nitrogen_dioxide"].mean(),
            "so2_mean": g["sulphur_dioxide"].mean(),
            "o3_8h_max": g["o3_8h"].max(),
            "co_8h_max": g["co_8h"].max(),
            "dust_mean": g["dust"].mean(),
            "aod_mean": g["aerosol_optical_depth"].mean(),
            "us_aqi_max": g["us_aqi"].max(),
            "hours_available": g["pm2_5"].count(),
        }
    ).reset_index()
    daily = daily[(daily["date"] >= start.isoformat()) & (daily["hours_available"] >= 18)]
    idx = daily.apply(
        lambda r: naqi(
            {
                "pm2_5": r.pm2_5_mean,
                "pm10": r.pm10_mean,
                "no2": r.no2_mean,
                "so2": r.so2_mean,
                "o3": r.o3_8h_max,
                "co": r.co_8h_max / 1000 if pd.notna(r.co_8h_max) else None,  # ug/m3 -> mg/m3
            }
        ),
        axis=1,
        result_type="expand",
    )
    if len(daily):
        daily[["india_aqi", "india_aqi_category", "dominant_pollutant"]] = idx
        # PM-only variant: CAMS model ozone runs high over India, so O3 can dominate the
        # full index. This one uses only PM2.5/PM10 (not the official >=3 pollutant rule).
        daily["india_aqi_pm"] = daily.apply(
            lambda r: round(min(500.0, max(sub_index("pm2_5", r.pm2_5_mean), sub_index("pm10", r.pm10_mean)))),
            axis=1,
        )
        daily["india_aqi_pm_category"] = daily["india_aqi_pm"].map(category)
    else:
        daily = daily.assign(
            india_aqi=None, india_aqi_category=None, dominant_pollutant=None,
            india_aqi_pm=None, india_aqi_pm_category=None,
        )
    daily.insert(1, "city", name)
    return daily


def weather_with_elevation(city: tuple, start: dt.date, end: dt.date) -> pd.DataFrame:
    df, elevation = fetch_weather(city, start, end)
    return df.assign(_elevation=elevation)


CACHE = Path("cache")


def cached(kind: str, city: tuple, start: dt.date, end: dt.date, fetch) -> pd.DataFrame:
    """Cache each fetched frame on disk so an interrupted run resumes without re-spending calls."""
    stem = f"{city[0].replace(' ', '_')}_{start}_"
    f = CACHE / kind / f"{stem}{end}.csv"
    if f.exists():
        return pd.read_csv(f, dtype={"date": str})
    # Reuse an older cache file with the same start and fetch only the days after it.
    older = sorted((CACHE / kind).glob(f"{stem}*.csv")) if (CACHE / kind).exists() else []
    if older:
        old = pd.read_csv(older[-1], dtype={"date": str})
        old_end = dt.date.fromisoformat(older[-1].stem.rsplit("_", 1)[1])
        df = pd.concat([old, fetch(city, old_end + dt.timedelta(days=1), end)], ignore_index=True)
        df = df.drop_duplicates("date", keep="last")
    else:
        df = fetch(city, start, end)
    f.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(f, index=False)
    return df


def load(name: str) -> pd.DataFrame | None:
    f = DATA / name
    return pd.read_csv(f, dtype={"date": str}) if f.exists() else None


def upsert(old: pd.DataFrame | None, new: list[pd.DataFrame]) -> pd.DataFrame:
    frames = ([old] if old is not None else []) + [n for n in new if len(n)]
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(["city", "date"], keep="last").sort_values(["city", "date"]).reset_index(drop=True)


def validate(weather: pd.DataFrame, aq: pd.DataFrame, yesterday: dt.date) -> None:
    names = {c[0] for c in CITIES}
    problems = []
    for label, df, start in (("weather", weather, None), ("air_quality", aq, AQ_START)):
        if set(df["city"]) != names:
            problems.append(f"{label}: cities differ: {set(df['city']) ^ names}")
        if df.duplicated(["city", "date"]).any():
            problems.append(f"{label}: duplicate (city, date) rows")
        for city, g in df.groupby("city"):
            dates = pd.to_datetime(g["date"])
            expected = (dates.max() - dates.min()).days + 1
            if len(g) != expected:
                problems.append(f"{label}/{city}: {expected - len(g)} missing days")
            if dates.max().date() < yesterday - dt.timedelta(days=3):
                problems.append(f"{label}/{city}: stale, last date {dates.max().date()}")
    checks = [
        (weather["temperature_2m_max"].between(-45, 55), "temperature_2m_max out of range"),
        (weather["temperature_2m_min"] <= weather["temperature_2m_max"] + 0.01, "min temp > max temp"),
        (weather["precipitation_sum"].fillna(0) >= 0, "negative precipitation"),
        (weather["relative_humidity_2m_mean"].between(0, 100), "humidity out of range"),
        (aq["pm2_5_mean"] >= 0, "negative PM2.5"),
        (aq["india_aqi"].between(0, 500), "India AQI out of range"),
        (aq["india_aqi_pm"].between(0, 500), "India PM AQI out of range"),
    ]
    for ok, msg in checks:
        bad = int((~ok).sum())
        if bad:
            problems.append(f"{msg}: {bad} rows")
    if problems:
        sys.exit("Validation failed:\n  " + "\n  ".join(problems))


def main() -> None:
    DATA.mkdir(exist_ok=True)
    yesterday = dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30))).date() - dt.timedelta(days=1)
    weather, aq = load("weather_daily.csv"), load("air_quality_daily.csv")
    cities_meta = load("cities.csv")
    elevations = dict(zip(cities_meta["city"], cities_meta["elevation_m"])) if cities_meta is not None else {}

    new_weather, new_aq = [], []
    for i, city in enumerate(CITIES, 1):
        name = city[0]
        w_have = weather[weather["city"] == name]["date"] if weather is not None else pd.Series(dtype=str)
        a_have = aq[aq["city"] == name]["date"] if aq is not None else pd.Series(dtype=str)

        # Forward: refresh recent provisional days and add new ones.
        w_from = (
            dt.date.fromisoformat(w_have.max()) - dt.timedelta(days=REFRESH_DAYS)
            if len(w_have)
            else INITIAL_WEATHER_START
        )
        a_from = (
            max(AQ_START, dt.date.fromisoformat(a_have.max()) - dt.timedelta(days=REFRESH_DAYS))
            if len(a_have)
            else AQ_START
        )
        df = cached("weather", city, w_from, yesterday, weather_with_elevation)
        if len(df):
            elevations[name] = df["_elevation"].iloc[0]
        new_weather.append(df.drop(columns="_elevation"))
        new_aq.append(cached("air_quality", city, a_from, yesterday, fetch_air_quality))

        # Backward: extend weather history toward WEATHER_START (not on the first build,
        # which already uses most of the day's free-tier budget).
        earliest = dt.date.fromisoformat(w_have.min()) if len(w_have) else None
        if earliest and earliest > WEATHER_START:
            b_from = max(WEATHER_START, earliest - dt.timedelta(days=BACKFILL_DAYS))
            df = cached("weather", city, b_from, earliest - dt.timedelta(days=1), weather_with_elevation)
            new_weather.append(df.drop(columns="_elevation"))
        print(f"  [{i}/{len(CITIES)}] {name}: calls used so far {calls_used:.0f}", flush=True)

    weather = upsert(weather, new_weather)
    aq = upsert(aq, new_aq)
    for col in aq.columns:
        if aq[col].dtype == float and col not in ("india_aqi", "india_aqi_pm"):
            aq[col] = aq[col].round(2)
    aq["india_aqi"] = aq["india_aqi"].astype("Int64")
    aq["india_aqi_pm"] = aq["india_aqi_pm"].astype("Int64")
    aq["hours_available"] = aq["hours_available"].astype(int)
    validate(weather, aq, yesterday)

    cities = pd.DataFrame(CITIES, columns=["city", "state", "region", "latitude", "longitude"])
    cities["elevation_m"] = cities["city"].map(elevations)
    cities.to_csv(DATA / "cities.csv", index=False)
    weather.to_csv(DATA / "weather_daily.csv", index=False)
    aq.to_csv(DATA / "air_quality_daily.csv", index=False)
    meta = {
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "cities": len(cities),
        "weather": {"rows": len(weather), "first": weather["date"].min(), "last": weather["date"].max()},
        "air_quality": {"rows": len(aq), "first": aq["date"].min(), "last": aq["date"].max()},
        "open_meteo_calls_used": round(calls_used, 1),
    }
    (DATA / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
