# Retail demand forecasting

![License](https://img.shields.io/badge/license-Apache--2.0-blue) ![Python](https://img.shields.io/badge/python-3.10%2B-blue)

Copyright 2026 Parth Maniar. Apache-2.0.
[GitHub](https://github.com/officialpm) | [Portfolio](https://www.parthmaniar.tech/)

A small forecasting project for promotion-sensitive retail demand. It compares seasonal estimates with per-store, per-product-family promotion models, keeps zero-sales days, and checks every output against the expected IDs.

This is a reproducible research workflow, not a deployed inventory system. It can be a foundation for a demand-planning service later; no service, dashboard or production integration is claimed here.

## Measured results

| Measurement | Result |
|---|---:|
| Public benchmark RMSLE, lower is better | **0.40508** |
| Previous public benchmark result | 0.41602 |
| Five-window development RMSLE, 56-day promotion model | 0.435361 |
| Same-window seasonal control | 0.460723 |
| Development windows won against seasonal control | 5 / 5 |

The public score was verified October 7, 2026. Development windows were reused as the method evolved; their results are selection-biased. The 112-day promotion model still wins on the latest development window. These numbers are not a guarantee on new stores, years or demand patterns. [Full result ledger](RESULTS.md).



## How it works

- Predict each store-family series separately, without dropping zero sales.
- Compare three original seasonal baselines, a longer-window seasonal control, and two fixed promotion models.
- Fit weekday intercepts and log promotion counts to log sales. Use fixed ridge penalties, not a parameter search.
- Evaluate five nonoverlapping 16-day horizons using only earlier history.
- Join forecasts to sample IDs one-to-one, preserving order. Reject missing, duplicate, nonfinite or negative output.

## Run locally

Python 3.10 or newer. Download the authorized original dataset yourself; it is not included. Put `train.csv`, `test.csv` and `sample_submission.csv` under `data/`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v
python train.py
```

Set `FORECAST_DATA_DIR` and `FORECAST_OUTPUT_DIR` to use other folders. Outputs include fold/model/family/promotion-group metrics, paired deltas, validation and test predictions, a checked CSV, charts and file hashes. No upload or submission happens automatically.

The portable script is an adaptation of the measured hosted run. Two synthetic tests passed locally: output shape/finiteness for all six models, and invariance to future target changes. Full-dataset retraining of the portable adaptation has **not** been run. Dependencies are compatible ranges, not a claimed exact lockfile for the historical environment.

## Project map

| File | Purpose |
|---|---|
| `train.py` | End-to-end loading, chronological evaluation and output writing |
| `forecast.py` | Forecast implementation used by synthetic tests |
| `test_forecast.py` | Output checks and future-target leakage test |
| `workflow.ipynb` | Notebook entry point for the portable script |
| `RESULTS.md` | Measured result provenance and limits |
| `LICENSE`, `NOTICE` | Code license, attribution and dataset separation |

## Limits and data rights

No holidays, oil prices, transactions or stockout labels are modeled. Promotions are associated with sales; the coefficient is not a causal estimate. Predictions floor negative log sales to zero, but training targets are untouched. No production latency, capacity or business impact has been measured.

This project uses the Corporacion Favorita Store Sales benchmark provided by the original benchmark organizers. Competition data is restricted to permitted competition, academic and noncommercial use; do not redistribute it. Apache-2.0 covers original code and docs only, not the dataset. NumPy, pandas and Matplotlib retain their licenses.

Source notebook remains private; access to it is not required to use the portable code with authorized data. Charts show aggregate model results, not dataset rows. No dataset, secrets, generated predictions or model binaries are bundled.
