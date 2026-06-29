"""
Feature Engineering für das Wetter-Forecastmodell.
Erstellt lag-Features, rollierende Statistiken und Zeitfeatures.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

log = logging.getLogger(__name__)

DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Ziel-Variablen (was wir vorhersagen wollen)
TARGETS = {
    "temperature_2m":         "Temperatur (°C)",
    "precipitation":          "Niederschlag (mm)",
    "wind_speed_10m":         "Windgeschwindigkeit (km/h)",
    "cloud_cover":            "Bewölkung (%)",
    "relative_humidity_2m":   "Luftfeuchtigkeit (%)",
}

# Für welche Horizonte (in Stunden) wir vorhersagen
FORECAST_HORIZONS = [1, 3, 6, 12, 24, 48]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Zyklische Zeitencoding (sin/cos) für Stunde, Tag, Monat."""
    df = df.copy()
    df["hour"]       = df["time"].dt.hour
    df["dayofyear"]  = df["time"].dt.dayofyear
    df["month"]      = df["time"].dt.month
    df["weekday"]    = df["time"].dt.weekday
    df["year"]       = df["time"].dt.year
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    # Zyklische Encoding
    df["hour_sin"]      = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]      = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"]       = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["doy_cos"]       = np.cos(2 * np.pi * df["dayofyear"] / 365)
    df["month_sin"]     = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]     = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_lag_features(df: pd.DataFrame, target_cols: list, lags: list) -> pd.DataFrame:
    """Lag-Features: vergangene Werte als Prädiktoren."""
    df = df.copy()
    for col in target_cols:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag{lag}h"] = df.groupby("city")[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, target_cols: list, windows: list) -> pd.DataFrame:
    """Rollierende Mittelwerte und Standardabweichungen."""
    df = df.copy()
    for col in target_cols:
        if col not in df.columns:
            continue
        for w in windows:
            df[f"{col}_roll{w}h_mean"] = (
                df.groupby("city")[col]
                  .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            )
            df[f"{col}_roll{w}h_std"] = (
                df.groupby("city")[col]
                  .transform(lambda x: x.shift(1).rolling(w, min_periods=1).std())
            )
    return df


def add_diff_features(df: pd.DataFrame, target_cols: list) -> pd.DataFrame:
    """Differenz-Features: Änderungsraten."""
    df = df.copy()
    for col in target_cols:
        if col not in df.columns:
            continue
        df[f"{col}_diff1h"] = df.groupby("city")[col].diff(1)
        df[f"{col}_diff3h"] = df.groupby("city")[col].diff(3)
        df[f"{col}_diff24h"] = df.groupby("city")[col].diff(24)
    return df


def add_target_horizons(df: pd.DataFrame, target_col: str, horizons: list) -> pd.DataFrame:
    """Erstellt Zielvariablen für mehrere Vorhersagehorizonte."""
    df = df.copy()
    for h in horizons:
        df[f"{target_col}_target_{h}h"] = df.groupby("city")[target_col].shift(-h)
    return df


def city_encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encoding der Städte."""
    dummies = pd.get_dummies(df["city"], prefix="city")
    return pd.concat([df, dummies], axis=1)


def build_feature_matrix(
    df: pd.DataFrame,
    primary_target: str = "temperature_2m",
    horizons: list = FORECAST_HORIZONS,
    lags: list = [1, 2, 3, 6, 12, 24, 48, 168],     # 1h bis 7d
    rolling_windows: list = [3, 6, 12, 24, 48, 168],
) -> pd.DataFrame:
    """
    Vollständige Feature-Matrix aufbauen.
    Gibt ein DataFrame zurück, das sofort für Training genutzt werden kann.
    """
    target_cols = [c for c in TARGETS.keys() if c in df.columns]

    log.info("Zeitfeatures berechnen...")
    df = add_time_features(df)

    log.info("Lag-Features berechnen...")
    df = add_lag_features(df, target_cols, lags)

    log.info("Rolling-Features berechnen...")
    df = add_rolling_features(df, target_cols, rolling_windows)

    log.info("Diff-Features berechnen...")
    df = add_diff_features(df, target_cols)

    log.info("Städte encodieren...")
    df = city_encode(df)

    log.info(f"Zielhorizonte erstellen für '{primary_target}'...")
    df = add_target_horizons(df, primary_target, horizons)

    # Zeilen mit NaN am Rand entfernen (durch Lags/Shifts entstanden)
    # Mindest-Lag ist 168h → erste 168 Zeilen je Stadt verwerfen
    min_lag = max(lags)
    df = df.sort_values(["city", "time"]).reset_index(drop=True)
    df = df.groupby("city").apply(lambda x: x.iloc[min_lag:]).reset_index(drop=True)

    log.info(f"Feature-Matrix: {df.shape[0]:,} Zeilen × {df.shape[1]} Spalten")
    return df


def get_feature_columns(df: pd.DataFrame, horizons: list = FORECAST_HORIZONS) -> list:
    """Gibt alle Input-Feature-Spalten zurück (ohne Targets und Metadaten)."""
    exclude = {"time", "city", "lat", "lon", "name"} | {
        f"{t}_target_{h}h"
        for t in TARGETS
        for h in horizons
    }
    # Auch die rohen Ziel-Variablen ausschliessen (Datenleck-Schutz)
    exclude.update(TARGETS.keys())

    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    import sys

    path = Path("data/raw/all_cities_history.parquet")
    if not path.exists():
        print("Erst data_collector.py ausführen!")
        sys.exit(1)

    raw = pd.read_parquet(path)
    features = build_feature_matrix(raw)
    out = PROCESSED_DIR / "features.parquet"
    features.to_parquet(out, index=False)
    print(f"Gespeichert: {out} ({features.shape})")
