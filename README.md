# India Daily Weather & Air Quality: data pipeline

This pipeline builds and updates a daily weather and air-quality dataset for 50 Indian cities.
It runs every day on GitHub Actions and publishes to:

- Hugging Face: [AkshayKarthick/india-weather-air-quality](https://huggingface.co/datasets/AkshayKarthick/india-weather-air-quality)
- Kaggle: [akshaykarthickms007/india-daily-weather-and-air-quality](https://www.kaggle.com/datasets/akshaykarthickms007/india-daily-weather-and-air-quality)

## How it works

| File | Role |
|---|---|
| `cities.py` | The 50 cities (state, region, coordinates) |
| `update.py` | Fetches from Open-Meteo, aggregates hourly air quality to daily, validates, writes `data/*.csv` |
| `naqi.py` | Indian National AQI (CPCB formula) |
| `publish.py` | `pull` gets the current data from Hugging Face; `push` publishes to Hugging Face and Kaggle |
| `.github/workflows/daily-update.yml` | Runs pull → update → push every day at 08:00 IST |
| `notebook/build_notebook.py` | Generates the Kaggle starter notebook |

Each run re-fetches the last 10 days, because recent values are provisional, and adds new days.
It also extends the weather history back toward 2020, and stays within Open-Meteo's free-tier limits.

## Run locally

```bash
pip install -r requirements.txt pytest
pytest                       # unit tests + one live API call
python update.py             # builds or updates data/
python publish.py push "msg" # needs HF_TOKEN and Kaggle credentials
```

The workflow needs the repository secrets `HF_TOKEN` (write), `KAGGLE_USERNAME` and `KAGGLE_KEY`.

## Data sources and license

Weather and air quality data come from [Open-Meteo.com](https://open-meteo.com/). The underlying
sources are ERA5 (Copernicus Climate Change Service) and CAMS (Copernicus Atmosphere Monitoring
Service), all under CC BY 4.0. The code in this repo is MIT licensed.
