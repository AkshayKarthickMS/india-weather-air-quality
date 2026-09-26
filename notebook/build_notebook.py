"""Generate the Kaggle starter notebook (starter.ipynb) from the cells below."""

from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md(
        """# India's Air & Weather: a Starter Analysis

This notebook explores the **[India Daily Weather & Air Quality (Auto-Updated)](https://www.kaggle.com/datasets/akshaykarthickms007/india-daily-weather-and-air-quality)**
dataset, which has daily weather and air quality for 50 Indian cities and gets new rows every day.

We'll look at:
1. Which cities have the dirtiest and cleanest air
2. Delhi's winter smog season
3. What air pollution looks like by month across India
4. How wind clears the air
5. The monsoon by region
6. A data-quality check: why this notebook uses `india_aqi_pm` rather than `india_aqi`"""
    ),
    code(
        """import glob, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Works on Kaggle (/kaggle/input/...) and locally (./data or ../data)
def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True) + glob.glob(f"data/{name}") + glob.glob(f"../data/{name}")
    return hits[0]

cities = pd.read_csv(find("cities.csv"))
weather = pd.read_csv(find("weather_daily.csv"), parse_dates=["date"])
aq = pd.read_csv(find("air_quality_daily.csv"), parse_dates=["date"])

# Chart style: thin marks, recessive grid, one accent color
BLUE, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({
    "figure.dpi": 110, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e", "axes.titlecolor": INK,
    "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9.5,
})

print(f"{len(cities)} cities")
print(f"weather:     {len(weather):>7,} rows, {weather.date.min():%Y-%m-%d} to {weather.date.max():%Y-%m-%d}")
print(f"air quality: {len(aq):>7,} rows, {aq.date.min():%Y-%m-%d} to {aq.date.max():%Y-%m-%d}")
cities.head()"""
    ),
    md(
        """## 1. Which cities have the dirtiest air?

Average PM-based Indian AQI (`india_aqi_pm`) over the full air-quality period. For reference,
CPCB counts 0–50 as *Good*, 51–100 as *Satisfactory*, 101–200 as *Moderate* and above 200 as *Poor* or worse."""
    ),
    code(
        """avg = aq.groupby("city")["india_aqi_pm"].mean().sort_values()
fig, ax = plt.subplots(figsize=(8, 10))
ax.barh(avg.index, avg.values, color=BLUE, height=0.7)
for band, label in [(100, "Satisfactory | Moderate"), (200, "Moderate | Poor")]:
    ax.axvline(band, color=MUTED, lw=1, ls="--")
    ax.text(band + 2, -0.9, label, color=MUTED, fontsize=8)
for city in list(avg.index[:3]) + list(avg.index[-5:]):  # label the extremes only
    ax.text(avg[city] + 2, city, f"{avg[city]:.0f}", va="center", color=INK, fontsize=8)
ax.set_title("Average PM-based AQI by city")
ax.set_xlabel("Mean daily india_aqi_pm")
ax.grid(axis="y", visible=False)
ax.margins(y=0.01)
plt.tight_layout(); plt.show()"""
    ),
    md(
        """The Indo-Gangetic plain (Delhi, Punjab, Uttar Pradesh, Bihar) sits at the top, and coastal,
hill and island cities sit at the bottom. The pattern is geographic: the plain is a basin where
cold, still winter air traps emissions."""
    ),
    md("## 2. Delhi's smog season"),
    code(
        """d = aq[aq.city == "Delhi"].set_index("date")["pm2_5_mean"]
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(d.index, d.values, color=BLUE, lw=0.6, alpha=0.45, label="Daily mean")
ax.plot(d.index, d.rolling(30, center=True).mean(), color=BLUE, lw=2, label="30-day average")
ax.axhline(60, color=MUTED, lw=1, ls="--")
ax.text(d.index.max(), 60, "CPCB 24-h standard: 60 µg/m³ ", color=MUTED, fontsize=8, ha="right", va="center",
        bbox=dict(facecolor="#fcfcfb", edgecolor="none", pad=1.5))
ax.set_title("Delhi PM2.5, daily")
ax.set_ylabel("PM2.5 (µg/m³)")
ax.legend(frameon=False, loc="upper right")
plt.tight_layout(); plt.show()

by_month = d.groupby(d.index.month).mean()
print("Delhi mean PM2.5 by month (µg/m³):")
print(by_month.round(0).rename(lambda m: pd.Timestamp(2000, m, 1).strftime("%b")).to_string())"""
    ),
    md(
        """Every year, PM2.5 climbs from October, peaks in November–January, and falls away with the
spring winds. The monsoon months (July–September) are the cleanest because rain washes particles
out of the air."""
    ),
    md("## 3. Air pollution by month across India"),
    code(
        """top = aq.groupby("city")["pm2_5_mean"].mean().sort_values(ascending=False).index[:20]
grid = (aq[aq.city.isin(top)]
        .assign(month=aq.date.dt.month)
        .pivot_table(index="city", columns="month", values="pm2_5_mean", aggfunc="mean")
        .loc[top])
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(grid.values, cmap="Blues", aspect="auto")
ax.set_xticks(range(12), [pd.Timestamp(2000, m, 1).strftime("%b") for m in grid.columns])
ax.set_yticks(range(len(grid)), grid.index)
ax.grid(False)
cb = fig.colorbar(im, ax=ax, shrink=0.8)
cb.set_label("Mean PM2.5 (µg/m³)", color="#52514e")
ax.set_title("Mean PM2.5 by month: 20 most polluted cities")
plt.tight_layout(); plt.show()"""
    ),
    md("## 4. Wind clears the air"),
    code(
        """m = weather.merge(aq, on=["city", "date"])
winter = m[(m.city == "Delhi") & (m.date.dt.month.isin([11, 12, 1]))]
bins = pd.cut(winter.wind_speed_10m_max, [0, 6, 9, 12, 15, 20, 40])
binned = winter.groupby(bins, observed=True)["pm2_5_mean"].agg(["mean", "count"])
fig, ax = plt.subplots(figsize=(7, 3.8))
labels = [f"{int(i.left)}–{int(i.right)}" for i in binned.index]
ax.bar(labels, binned["mean"], color=BLUE, width=0.6)
for x, (v, n) in enumerate(zip(binned["mean"], binned["count"])):
    ax.text(x, v + 3, f"{v:.0f}", ha="center", color=INK, fontsize=8)
ax.set_title("Delhi winter PM2.5 by daily max wind speed")
ax.set_xlabel("Max wind speed (km/h)"); ax.set_ylabel("Mean PM2.5 (µg/m³)")
ax.grid(axis="x", visible=False)
plt.tight_layout(); plt.show()
print(binned.round(1))"""
    ),
    md(
        """On calm winter days, PM2.5 is several times higher than on windy days. That makes weather
features essential for any PM2.5 forecasting model."""
    ),
    md("## 5. The monsoon by region"),
    code(
        """rain = (weather.merge(cities[["city", "region"]], on="city")
          .assign(month=weather.date.dt.month)
          .groupby(["region", "month", "city"])["precipitation_sum"].sum()
          / weather.date.dt.year.nunique())
rain = rain.groupby(["region", "month"]).mean()
regions = [r for r in ["North", "Central", "West", "East", "Northeast", "South"] if r in rain.index]
fig, axes = plt.subplots(2, 3, figsize=(11, 5.5), sharey=True)
for ax, region in zip(axes.flat, regions):
    r = rain.loc[region]
    ax.bar(r.index, r.values, color=BLUE, width=0.7)
    ax.set_title(region, fontsize=10)
    ax.set_xticks([1, 4, 7, 10], ["Jan", "Apr", "Jul", "Oct"])
    ax.grid(axis="x", visible=False)
for ax in axes[:, 0]:
    ax.set_ylabel("mm / month")
fig.suptitle("Average monthly rainfall per city, by region", x=0.01, ha="left", fontweight="bold")
plt.tight_layout(); plt.show()"""
    ),
    md(
        """The June–September south-west monsoon dominates everywhere except the South, where the
north-east monsoon brings a second peak in October–December (Chennai, Puducherry)."""
    ),
    md(
        """## 6. Data-quality check: `india_aqi` vs `india_aqi_pm`

The dataset has two Indian AQI columns. `india_aqi` follows the full CPCB formula with all
pollutants, but the air-quality data comes from the CAMS global model, which is known to
**overestimate ozone over India**. Let's see how often ozone drives the index:"""
    ),
    code(
        """share = aq["dominant_pollutant"].value_counts(normalize=True).mul(100).sort_values()
fig, ax = plt.subplots(figsize=(7, 2.8))
ax.barh(share.index, share.values, color=BLUE, height=0.6)
for p, v in share.items():
    ax.text(v + 0.8, p, f"{v:.0f}%", va="center", color=INK, fontsize=8)
ax.set_title("Dominant pollutant in india_aqi (share of city-days)")
ax.set_xlabel("% of city-days"); ax.grid(axis="y", visible=False)
plt.tight_layout(); plt.show()

gap = (aq["india_aqi"] - aq["india_aqi_pm"])
print(f"india_aqi is higher than india_aqi_pm on {100 * (gap > 0).mean():.0f}% of days "
      f"(median gap {gap.median():.0f} points).")"""
    ),
    md(
        """Ozone drives a large share of `india_aqi`, which is unusual for Indian ground-station data,
where PM2.5 and PM10 usually dominate. That's why this notebook uses **`india_aqi_pm`**.
When you use model-based air quality data, check which pollutant drives the index.

## Ideas to try next
- **Forecast tomorrow's PM2.5** from today's air quality and weather (LightGBM with lag features).
- **Diwali effect:** compare PM2.5 around Diwali each year with the weeks before.
- **Heatwaves:** count days with `temperature_2m_max` ≥ 45 °C per city and year as the history grows.
- **Cluster cities** by their seasonal pollution and rainfall profiles.

If this helped, please upvote the dataset. It updates every day."""
    ),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
Path(__file__).with_name("starter.ipynb").write_text(nbf.writes(nb), encoding="utf-8")
print("wrote starter.ipynb")
