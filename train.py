"""
Inference: Erzeugt Wettervorhersagen für alle vier Schweizer Städte.
Nutzt trainierte LightGBM-Modelle + aktuelle Open-Meteo-Forecast-Daten.
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
import logging

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODELS_DIR  = Path("models")
OUTPUT_DIR  = Path("data/forecasts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_models(target_col: str, horizons: list) -> dict:
    models = {}
    for h in horizons:
        path = MODELS_DIR / f"{target_col}_{h}h.pkl"
        with open(path, "rb") as f:
            models[h] = pickle.load(f)
        log.info(f"  Modell geladen: {path.name}")
    return models


def predict(
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    target_col: str = "temperature_2m",
    horizons: list = [1, 3, 6, 12, 24, 48],
) -> pd.DataFrame:
    """
    Erzeugt Vorhersagen basierend auf:
    - history_df: historische Daten (für Lag-Features)
    - forecast_df: NWP-Rohdaten (Open-Meteo Forecast, als Input-Features)
    """
    from feature_engineering import (
        build_feature_matrix, get_feature_columns, TARGETS
    )
    from data_collector import CITIES

    # Meta laden
    meta_path = MODELS_DIR / "model_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]

    # Historische + Forecast-Daten zusammenführen
    combined = pd.concat([history_df, forecast_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["time", "city"])
    combined = combined.sort_values(["city", "time"]).reset_index(drop=True)

    # Features berechnen
    feat_df = build_feature_matrix(combined, primary_target=target_col, horizons=[])
    feat_df = feat_df[feat_df["time"].isin(forecast_df["time"])]

    # Nur verfügbare Features nutzen
    available_cols = [c for c in feature_cols if c in feat_df.columns]
    missing = set(feature_cols) - set(available_cols)
    if missing:
        log.warning(f"  {len(missing)} Features fehlen, werden mit 0 gefüllt")
        for col in missing:
            feat_df[col] = 0.0

    models = load_models(target_col, horizons)

    results = []
    for h, model in models.items():
        X = feat_df[feature_cols].copy()
        preds = model.predict(X)
        for i, row in feat_df.iterrows():
            results.append({
                "city":           row["city"],
                "forecast_base":  row["time"],
                "target_time":    row["time"] + pd.Timedelta(hours=h),
                "horizon_h":      h,
                "target_col":     target_col,
                "predicted":      preds[feat_df.index.get_loc(i)],
            })

    return pd.DataFrame(results)


def run_forecast_pipeline():
    """Full Pipeline: Daten laden → Features → Vorhersage → speichern."""
    from data_collector import fetch_forecast, CITIES
    import pandas as pd
    from pathlib import Path

    raw_path = Path("data/raw/all_cities_history.parquet")
    if not raw_path.exists():
        raise FileNotFoundError("Historische Daten fehlen. Erst data_collector.py ausführen.")

    history = pd.read_parquet(raw_path)
    # Nur letzte 30 Tage für Lag-Features nötig
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=30)
    history = history[history["time"] >= cutoff]

    log.info("Lade aktuelle NWP-Forecast-Daten...")
    forecast_frames = []
    for city_key in CITIES:
        df = fetch_forecast(city_key, days=7)
        forecast_frames.append(df)
    forecast_df = pd.concat(forecast_frames, ignore_index=True)

    log.info("Erstelle Vorhersagen...")
    results = predict(history, forecast_df)

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = OUTPUT_DIR / f"forecast_{ts}.parquet"
    results.to_parquet(out_path, index=False)
    log.info(f"✓ Vorhersagen gespeichert: {out_path} ({len(results):,} Zeilen)")

    # Auch als CSV (für einfache Inspektion)
    results.to_csv(OUTPUT_DIR / "latest_forecast.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_forecast_pipeline()
    print(results.head(20).to_string())
