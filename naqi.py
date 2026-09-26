"""Indian National Air Quality Index (CPCB, 2014).

Sub-index for each pollutant by linear interpolation between breakpoints; the AQI is
the maximum sub-index, reported only when at least 3 pollutants (one of them PM2.5
or PM10) are available. Concentrations in ug/m3 except CO in mg/m3.
PM, NO2 and SO2 use 24-hour averages; O3 and CO use the maximum 8-hour average.

CPCB leaves the top of the "Severe" band open-ended; like most implementations we
extend the last band linearly (upper concentrations below) and cap the index at 500.
"""

from __future__ import annotations

import math

AQI_BANDS = [0, 50, 100, 200, 300, 400, 500]
CATEGORIES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

# Concentration breakpoints matching AQI_BANDS.
BREAKPOINTS = {
    "pm2_5": [0, 30, 60, 90, 120, 250, 380],
    "pm10": [0, 50, 100, 250, 350, 430, 510],
    "no2": [0, 40, 80, 180, 280, 400, 520],
    "so2": [0, 40, 80, 380, 800, 1600, 2400],
    "o3": [0, 50, 100, 168, 208, 748, 1000],
    "co": [0, 1.0, 2.0, 10, 17, 34, 51],
}

LABELS = {"pm2_5": "PM2.5", "pm10": "PM10", "no2": "NO2", "so2": "SO2", "o3": "O3", "co": "CO"}


def sub_index(pollutant: str, conc: float | None) -> float | None:
    if conc is None or (isinstance(conc, float) and math.isnan(conc)):
        return None
    bp = BREAKPOINTS[pollutant]
    conc = max(conc, 0.0)
    for i in range(1, len(bp)):
        if conc <= bp[i]:
            lo_c, hi_c = bp[i - 1], bp[i]
            lo_i, hi_i = AQI_BANDS[i - 1], AQI_BANDS[i]
            return lo_i + (hi_i - lo_i) * (conc - lo_c) / (hi_c - lo_c)
    return 500.0


def category(aqi: float | None) -> str | None:
    if aqi is None:
        return None
    for upper, name in zip(AQI_BANDS[1:], CATEGORIES):
        if aqi <= upper:
            return name
    return CATEGORIES[-1]


def naqi(concs: dict[str, float | None]) -> tuple[int | None, str | None, str | None]:
    """Return (aqi, category, dominant pollutant) for one day of concentrations."""
    subs = {p: sub_index(p, concs.get(p)) for p in BREAKPOINTS}
    subs = {p: v for p, v in subs.items() if v is not None}
    if len(subs) < 3 or not ({"pm2_5", "pm10"} & subs.keys()):
        return None, None, None
    dominant = max(subs, key=subs.get)
    aqi = round(min(subs[dominant], 500.0))
    return aqi, category(aqi), LABELS[dominant]
