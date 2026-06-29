# .github/workflows/daily_retrain.yml
#
# Kostenloser täglicher Re-Training-Job via GitHub Actions.
# GitHub Actions ist für öffentliche Repos vollständig gratis,
# für private Repos: 2000 Minuten/Monat gratis (reicht für diesen Job locker).
#
# Speichert Modelle als GitHub Artifacts (oder DVC remote auf DagsHub).

name: Daily Weather Model Retrain

on:
  schedule:
    # Täglich um 03:00 Uhr Schweizer Zeit (01:00 UTC)
    - cron: "0 1 * * *"
  workflow_dispatch:  # Manueller Start via GitHub UI

jobs:
  retrain:
    runs-on: ubuntu-latest
    timeout-minutes: 60

    steps:
      - name: Repository auschecken
        uses: actions/checkout@v4

      - name: Python 3.11 einrichten
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - name: Abhängigkeiten installieren
        run: pip install -r requirements.txt

      # ── Daten/Modelle via DVC aus DagsHub (gratis) laden ──────────────────
      - name: DVC Pull (Daten & Modelle)
        if: ${{ env.DAGSHUB_TOKEN != '' }}
        env:
          DAGSHUB_TOKEN: ${{ secrets.DAGSHUB_TOKEN }}
        run: |
          pip install dvc dvc-s3
          dvc remote modify origin --local auth basic
          dvc remote modify origin --local user ${{ secrets.DAGSHUB_USER }}
          dvc remote modify origin --local password ${{ secrets.DAGSHUB_TOKEN }}
          dvc pull -r origin || echo "Kein DVC-Remote konfiguriert, überspringe Pull"

      # ── Inkrementelles Re-Training ─────────────────────────────────────────
      - name: Modelle re-trainieren
        run: python src/retrain.py
        env:
          MLFLOW_TRACKING_URI: mlruns

      # ── Vollständiges Re-Training jeden Sonntag ────────────────────────────
      - name: Vollständiges Re-Training (sonntags)
        if: github.event.schedule == '0 1 * * 0'
        run: python src/retrain.py --full

      # ── Modelle als Artifact speichern ─────────────────────────────────────
      - name: Modelle als GitHub Artifact hochladen
        uses: actions/upload-artifact@v4
        with:
          name: weather-models-${{ github.run_number }}
          path: models/
          retention-days: 30

      # ── Forecasts als Artifact speichern ───────────────────────────────────
      - name: Forecasts hochladen
        uses: actions/upload-artifact@v4
        with:
          name: forecasts-${{ github.run_number }}
          path: data/forecasts/
          retention-days: 7

      # ── DVC Push (optional, wenn DagsHub konfiguriert) ────────────────────
      - name: DVC Push (Daten & Modelle sichern)
        if: ${{ env.DAGSHUB_TOKEN != '' }}
        env:
          DAGSHUB_TOKEN: ${{ secrets.DAGSHUB_TOKEN }}
        run: dvc push -r origin || echo "DVC Push übersprungen"

      # ── Repo committen (MLflow-Logs & Reports) ─────────────────────────────
      - name: Änderungen committen
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add reports/ mlruns/ || true
          git diff --cached --quiet || git commit -m "auto: Re-Training $(date +'%Y-%m-%d')"
          git push || echo "Nichts zu pushen"
