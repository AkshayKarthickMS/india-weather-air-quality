"""Publish data/ to Hugging Face and Kaggle.

    python publish.py pull      # download the current published data into data/ (used by CI)
    python publish.py push      # upload data/ + generated cards to both platforms

Needs HF_TOKEN (or a saved `hf auth login`) and Kaggle credentials
(~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

HF_REPO = "AkshayKarthick/india-weather-air-quality"
KAGGLE_ID = "akshaykarthickms007/india-daily-weather-and-air-quality"
TITLE = "India Daily Weather & Air Quality (Auto-Updated)"
SUBTITLE = "Daily weather and PM2.5/PM10/AQI for 50 Indian cities, updated every day"
DATA = Path("data")
FILES = ["cities.csv", "weather_daily.csv", "air_quality_daily.csv", "metadata.json"]
CODE = ["update.py", "naqi.py", "cities.py", "publish.py"]

COLUMNS = {
    "cities.csv": {
        "city": "City name",
        "state": "State or union territory",
        "region": "North / South / East / West / Central / Northeast",
        "latitude": "Latitude used for the query (city centre)",
        "longitude": "Longitude used for the query (city centre)",
        "elevation_m": "Elevation of the model grid point in metres",
    },
    "weather_daily.csv": {
        "date": "Local date (Asia/Kolkata)",
        "city": "City name (join key to cities.csv)",
        "temperature_2m_max": "Maximum air temperature at 2 m (°C)",
        "temperature_2m_min": "Minimum air temperature at 2 m (°C)",
        "temperature_2m_mean": "Mean air temperature at 2 m (°C)",
        "apparent_temperature_max": "Maximum feels-like temperature including humidity and wind (°C)",
        "precipitation_sum": "Total precipitation (mm)",
        "precipitation_hours": "Hours with precipitation (h)",
        "relative_humidity_2m_mean": "Mean relative humidity at 2 m (%)",
        "wind_speed_10m_max": "Maximum wind speed at 10 m (km/h)",
        "wind_direction_10m_dominant": "Dominant wind direction (° from north)",
        "shortwave_radiation_sum": "Total solar radiation (MJ/m²)",
        "et0_fao_evapotranspiration": "Reference evapotranspiration, FAO-56 (mm)",
        "weather_code": "Most severe WMO weather code of the day (0 clear … 95+ thunderstorm)",
    },
    "air_quality_daily.csv": {
        "date": "Local date (Asia/Kolkata)",
        "city": "City name (join key to cities.csv)",
        "pm2_5_mean": "24-hour mean PM2.5 (µg/m³)",
        "pm2_5_max": "Highest hourly PM2.5 (µg/m³)",
        "pm10_mean": "24-hour mean PM10 (µg/m³)",
        "pm10_max": "Highest hourly PM10 (µg/m³)",
        "no2_mean": "24-hour mean nitrogen dioxide (µg/m³)",
        "so2_mean": "24-hour mean sulphur dioxide (µg/m³)",
        "o3_8h_max": "Highest 8-hour mean ozone (µg/m³)",
        "co_8h_max": "Highest 8-hour mean carbon monoxide (µg/m³)",
        "dust_mean": "24-hour mean dust (µg/m³)",
        "aod_mean": "Mean aerosol optical depth at 550 nm",
        "us_aqi_max": "Highest hourly US EPA AQI",
        "hours_available": "Hours of model data behind the daily values (days with < 18 are dropped)",
        "india_aqi": "Indian National AQI (CPCB formula, all pollutants). See the ozone caveat",
        "india_aqi_category": "Good / Satisfactory / Moderate / Poor / Very Poor / Severe",
        "dominant_pollutant": "Pollutant with the highest sub-index",
        "india_aqi_pm": "Indian AQI from PM2.5 and PM10 only (recommended; see caveat)",
        "india_aqi_pm_category": "Category of india_aqi_pm",
    },
}


def stats() -> dict:
    meta = json.loads((DATA / "metadata.json").read_text(encoding="utf-8"))
    aq = pd.read_csv(DATA / "air_quality_daily.csv")
    worst = (
        aq.groupby("city")["india_aqi_pm"].mean().sort_values(ascending=False).round(0).astype(int)
    )
    meta["worst"] = worst.head(5).to_dict()
    meta["cleanest"] = worst.tail(5).iloc[::-1].to_dict()
    meta["o3_dominant_pct"] = round(100 * (aq["dominant_pollutant"] == "O3").mean(), 1)
    return meta


def body(s: dict) -> str:
    def table(file: str) -> str:
        return "\n".join(f"| `{k}` | {v} |" for k, v in COLUMNS[file].items())

    worst = ", ".join(f"{c} ({v})" for c, v in s["worst"].items())
    cleanest = ", ".join(f"{c} ({v})" for c, v in s["cleanest"].items())
    return f"""Daily **weather** and **air quality** for **{s['cities']} Indian cities**, covering every region
from Leh to Port Blair. The dataset **updates automatically every day**.

| File | Rows | Coverage |
|---|---:|---|
| `weather_daily.csv` | {s['weather']['rows']:,} | {s['weather']['first']} → {s['weather']['last']} |
| `air_quality_daily.csv` | {s['air_quality']['rows']:,} | {s['air_quality']['first']} → {s['air_quality']['last']} |
| `cities.csv` | {s['cities']} | city, state, region, coordinates, elevation |

Last updated: **{s['updated_at'][:10]}**. Weather history is being extended back to 2020-01-01,
about one year per day.

Highest average PM-based AQI: {worst}. Lowest: {cleanest}.

## Ideas

- Winter smog in the Indo-Gangetic plain: when does PM2.5 spike, and how does it relate to
  wind speed, temperature inversions and Diwali?
- Heatwaves: days with `temperature_2m_max` ≥ 45 °C, and how `apparent_temperature_max`
  compares across humid coastal and dry inland cities.
- Monsoon onset and withdrawal dates from `precipitation_sum`, by region.
- Forecast tomorrow's PM2.5 from today's weather and air quality (time-series ML).
- Dust storms: `dust_mean` and `aod_mean` over Rajasthan and the north-west.

## Files and columns

Join the files on `city` (and `date` for the two daily files).

### weather_daily.csv
| Column | Description |
|---|---|
{table('weather_daily.csv')}

### air_quality_daily.csv
| Column | Description |
|---|---|
{table('air_quality_daily.csv')}

### cities.csv
| Column | Description |
|---|---|
{table('cities.csv')}

## How the Indian AQI is computed

Open-Meteo provides European and US AQI but not India's. This dataset adds the
**CPCB National AQI**:
- PM2.5, PM10, NO₂ and SO₂ use 24-hour means. O₃ and CO use the highest 8-hour mean.
- Each pollutant gets a sub-index by linear interpolation between the CPCB breakpoints.
- The AQI is the highest sub-index, reported when at least 3 pollutants, including PM, are available.
- The top "Severe" band is extended linearly and capped at 500.

**Ozone caveat.** The air-quality values come from the CAMS global model, not ground stations.
CAMS is known to overestimate surface ozone over India, so O₃ is the dominant pollutant on
{s['o3_dominant_pct']}% of days in `india_aqi`, which pushes that index higher than CPCB station
readings. **For most analyses, use `india_aqi_pm`**, which is computed from PM2.5 and PM10 only.

## Sources and method

- **Weather:** [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api).
  It combines ECMWF IFS and ERA5/ERA5-Land reanalysis from the Copernicus Climate Change Service,
  with a resolution of about 9–25 km.
- **Air quality:** [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api).
  This is the Copernicus Atmosphere Monitoring Service (CAMS) global model at about 45 km. Its
  history for India starts in August 2022.
- Values are for the model grid cell at each city centre, aggregated to local days (IST).
- **Updates:** a GitHub Actions job runs every day. It re-downloads the last 10 days, because
  recent values are provisional and get revised, adds new days, validates the data, and
  publishes a new version. Validation covers missing days, duplicates, value ranges and stale cities.

## Limitations

- This is **model and reanalysis data, not station measurements**. It is smooth over 9–45 km
  and can miss very local effects, like a single busy road or an industrial area.
- The last ~10 days are provisional and can change slightly in later versions.
- Model ozone runs high over India (see above). Model PM tends to be less extreme than
  station peaks in severe winter smog.

## License and attribution

**CC BY 4.0.** Please credit:
- Weather data by [Open-Meteo.com](https://open-meteo.com/), with ERA5 from the Copernicus
  Climate Change Service.
- Air quality data by [Open-Meteo.com](https://open-meteo.com/) and the
  Copernicus Atmosphere Monitoring Service (CAMS).
- Indian AQI calculation and dataset by [AkshayKarthick](https://huggingface.co/AkshayKarthick).

The pipeline code (`update.py`, `naqi.py`, `cities.py`, `publish.py`) is included."""


def hf_card(s: dict) -> str:
    return f"""---
license: cc-by-4.0
language:
  - en
pretty_name: {TITLE}
size_categories:
  - 100K<n<1M
task_categories:
  - time-series-forecasting
  - tabular-regression
tags:
  - weather
  - climate
  - air-quality
  - pollution
  - aqi
  - india
  - time-series
configs:
  - config_name: weather
    data_files: weather_daily.csv
    default: true
  - config_name: air_quality
    data_files: air_quality_daily.csv
  - config_name: cities
    data_files: cities.csv
---

# {TITLE}

Also on Kaggle: [kaggle.com/datasets/{KAGGLE_ID}](https://www.kaggle.com/datasets/{KAGGLE_ID})

{body(s)}

```python
from datasets import load_dataset

weather = load_dataset("{HF_REPO}", "weather", split="train").to_pandas()
aq = load_dataset("{HF_REPO}", "air_quality", split="train").to_pandas()
df = weather.merge(aq, on=["city", "date"])
```
"""


def kaggle_meta(s: dict, license_name: str) -> dict:
    def schema(file: str) -> dict:
        ints = {"precipitation_hours", "weather_code", "hours_available", "india_aqi", "india_aqi_pm", "us_aqi_max"}
        text = {"date", "city", "state", "region", "india_aqi_category", "dominant_pollutant", "india_aqi_pm_category"}
        return {
            "fields": [
                {
                    "name": k,
                    "description": v,
                    "type": "datetime" if k == "date" else "string" if k in text else "integer" if k in ints else "number",
                }
                for k, v in COLUMNS[file].items()
            ]
        }

    return {
        "title": TITLE,
        "subtitle": SUBTITLE,
        "id": KAGGLE_ID,
        "licenses": [{"name": license_name}],
        "keywords": ["india", "weather and climate", "pollution", "environment", "time series analysis"],
        "description": body(s)
        + f"\n\nAlso on Hugging Face: [huggingface.co/datasets/{HF_REPO}](https://huggingface.co/datasets/{HF_REPO})",
        "resources": [
            {"path": f, "description": f"{f} (see column descriptions)", "schema": schema(f)} for f in COLUMNS
        ],
    }


def pull() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(HF_REPO, repo_type="dataset", local_dir=DATA, allow_patterns=FILES)
    print("pulled", sorted(p.name for p in DATA.iterdir()))


def push(message: str) -> None:
    from huggingface_hub import HfApi

    s = stats()

    # Hugging Face
    hf = Path("release/hf")
    shutil.rmtree(hf, ignore_errors=True)
    hf.mkdir(parents=True)
    for f in FILES:
        shutil.copy(DATA / f, hf / f)
    for f in CODE:
        shutil.copy(f, hf / f)
    (hf / "README.md").write_text(hf_card(s), encoding="utf-8")
    api = HfApi()
    api.create_repo(HF_REPO, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=hf, repo_id=HF_REPO, repo_type="dataset", commit_message=message)
    print(f"HF: https://huggingface.co/datasets/{HF_REPO}")

    # Kaggle
    from kaggle.api.kaggle_api_extended import KaggleApi

    kg = Path("release/kaggle")
    shutil.rmtree(kg, ignore_errors=True)
    kg.mkdir(parents=True)
    for f in COLUMNS:
        shutil.copy(DATA / f, kg / f)
    kapi = KaggleApi()
    kapi.authenticate()
    exists = any(d.ref == KAGGLE_ID for d in kapi.dataset_list(user=KAGGLE_ID.split("/")[0]) or [])
    for license_name in ("CC-BY-4.0", "other"):
        (kg / "dataset-metadata.json").write_text(
            json.dumps(kaggle_meta(s, license_name), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        try:
            if exists:
                kapi.dataset_create_version(str(kg), version_notes=message, dir_mode="skip", quiet=True)
                kapi.dataset_metadata_update(KAGGLE_ID, str(kg))
            else:
                kapi.dataset_create_new(str(kg), public=True, dir_mode="skip", quiet=True)
            break
        except Exception as e:  # noqa: BLE001 - retry once with a fallback license
            if "licen" not in str(e).lower() or license_name == "other":
                raise
            print(f"Kaggle rejected license {license_name}; retrying with 'other'")
    print(f"Kaggle: https://www.kaggle.com/datasets/{KAGGLE_ID}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "pull":
        pull()
    elif cmd == "push":
        push(sys.argv[2] if len(sys.argv) > 2 else "Update data")
    else:
        sys.exit(__doc__)
