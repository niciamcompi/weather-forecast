"""
Datensammlung via Open-Meteo API (komplett kostenlos, kein API-Key nötig).
Historische + Forecast-Daten für Zürich, Bern, Genf, Basel.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Städte-Konfiguration ──────────────────────────────────────────────────────
CITIES = {
    "zuerich": {"lat": 47.3769, "lon": 8.5417, "name": "Zürich"},
    "bern":    {"lat": 46.9480, "lon": 7.4474,  "name": "Bern"},
    "genf":    {"lat": 46.2044, "lon": 6.1432,  "name": "Genf"},
    "basel":   {"lat": 47.5596, "lon": 7.5886,  "name": "Basel"},
}

# ── Variablen (alle kostenlos via Open-Meteo) ─────────────────────────────────
HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "pressure_msl",
    "cloud_cover",
    "surface_pressure",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation_probability",
    "weather_code",
    "shortwave_radiation",
    "et0_fao_evapotranspiration",
]

DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def fetch_historical(city_key: str, start: str, end: str) -> pd.DataFrame:
    """
    Historische Stundendaten von Open-Meteo Archive API.
    Kostenlos, kein API-Key, bis zu 80 Jahre zurück.
    """
    city = CITIES[city_key]
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "Europe/Zurich",
    }
    log.info(f"Lade historische Daten für {city['name']} ({start} – {end})")
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    raw = r.json()

    df = pd.DataFrame(raw["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df["city"] = city_key
    df["lat"] = city["lat"]
    df["lon"] = city["lon"]
    return df


def fetch_forecast(city_key: str, days: int = 16) -> pd.DataFrame:
    """
    Forecast-Daten (bis 16 Tage) von Open-Meteo — ebenfalls kostenlos.
    Wird für Inference verwendet.
    """
    city = CITIES[city_key]
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "hourly": ",".join(HOURLY_VARS),
        "forecast_days": days,
        "timezone": "Europe/Zurich",
    }
    log.info(f"Lade Forecast-Daten für {city['name']}")
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    raw = r.json()

    df = pd.DataFrame(raw["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df["city"] = city_key
    return df


def collect_all_history(years_back: int = 5) -> pd.DataFrame:
    """Alle Städte, letzten N Jahre historisch sammeln."""
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=365 * years_back)).strftime("%Y-%m-%d")

    frames = []
    for city_key in CITIES:
        df = fetch_historical(city_key, start, end)
        path = DATA_DIR / f"{city_key}_history.parquet"
        df.to_parquet(path, index=False)
        log.info(f"  → {len(df):,} Zeilen gespeichert: {path}")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(DATA_DIR / "all_cities_history.parquet", index=False)
    log.info(f"Gesamt: {len(combined):,} Zeilen")
    return combined


def update_recent(days: int = 7) -> pd.DataFrame:
    """Nur die letzten N Tage nachladen (für tägliches Re-Training)."""
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    frames = []
    for city_key in CITIES:
        df = fetch_historical(city_key, start, end)
        frames.append(df)

    new_data = pd.concat(frames, ignore_index=True)

    # An bestehende Datei anhängen und deduplizieren
    full_path = DATA_DIR / "all_cities_history.parquet"
    if full_path.exists():
        old = pd.read_parquet(full_path)
        combined = pd.concat([old, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=["time", "city"])
        combined = combined.sort_values(["city", "time"]).reset_index(drop=True)
        combined.to_parquet(full_path, index=False)
        log.info(f"Update: +{len(new_data)} neue Zeilen, gesamt {len(combined):,}")
        return new_data

    new_data.to_parquet(full_path, index=False)
    return new_data


if __name__ == "__main__":
    collect_all_history(years_back=5)
