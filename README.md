# 🌦️ Swiss Weather Forecast System

Automated ML pipeline forecasting temperature, precipitation, and wind speed for **Zürich, Bern, Geneva, and Basel** — entirely free to run.

## What it predicts

| Target | Horizons |
|--------|----------|
| Temperature (°C) | 1 h, 3 h, 6 h, 12 h, 24 h, 48 h |
| Precipitation (mm) | 1 h, 3 h, 6 h, 12 h, 24 h, 48 h |
| Wind speed (km/h) | 1 h, 3 h, 6 h, 12 h, 24 h, 48 h |

One LightGBM model per (target, horizon) pair; all four cities in a single global model with one-hot city encoding.

## Quick start

```bash
pip install -r requirements.txt
python src/data_collector.py      # fetch 5 years of history (~2 min)
python src/feature_engineering.py # build feature matrix
python src/train.py               # train models (~10–20 min)
python src/inference.py           # generate 7-day forecasts
python src/retrain.py             # incremental retrain
python src/retrain.py --full      # full retrain from scratch
```

## Pipeline

```
Open-Meteo API → data_collector.py → feature_engineering.py → train.py → inference.py
                                                                        ↑
                                                              retrain.py (daily, via GitHub Actions)
```

**Features:** lag (1–168 h), rolling mean/std (3–168 h), differences (1/3/24 h), sin/cos time encoding, city dummies.

**Retraining:** daily at 03:00 CET via GitHub Actions. New models are only accepted if they are not more than 5% worse than the current model on a 30-day holdout. Full retrain runs every Sunday.

## Stack (all free)

| Component | Tool |
|-----------|------|
| Weather data | Open-Meteo API |
| Training | LightGBM + scikit-learn |
| Experiment tracking | MLflow (local) or DagsHub |
| Retraining automation | GitHub Actions |
| Model/data versioning | DVC + DagsHub (10 GB free) |

## Optional: DagsHub remote

```bash
dvc remote add -d origin https://dagshub.com/<USER>/<REPO>.dvc
dvc push
```

Set `DAGSHUB_USER` and `DAGSHUB_TOKEN` as GitHub repository secrets to enable DVC pull/push in CI.
