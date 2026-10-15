# Hybrid LSTM–XGBoost forecasting of hourly industrial electricity use

Code for a stacked hybrid model that forecasts hourly electricity consumption in a food processing plant, together with the reference models, ablation, statistical tests and a forecast-triggered peak-load reduction rule.

**Status.** This code accompanies a manuscript that is under peer review. It is shared so that the analysis can be examined and run. Please do not reuse or redistribute it without the authors' permission (see `LICENSE`). Once the article is published, please cite it.

The plant data are confidential and are not included. The repository contains a script that creates **synthetic data with the same structure**, so every part of the code can be run and checked. Results obtained with the synthetic data do not reproduce results obtained with the plant data.

## Quick start

```bash
pip install -r requirements.txt
python make_synthetic_data.py                        # writes data/synthetic_*.csv
python run_all.py --quick                            # ~1 min smoke test (1 seed, 2 epochs)
python run_all.py                                    # full protocol (5 seeds, ~15-25 min on a laptop CPU)
python peak_reduction.py --quick
python daily_meter_check.py --daily data/synthetic_daily_meter.csv \
       --production data/synthetic_daily_production.csv --dev data/synthetic_hourly.csv \
       --trial-date 2025-08-06
```

`Hybrid_LSTM_XGBoost_demo.ipynb` contains the same code in one notebook for Google Colab or Jupyter. Run all cells; it creates the synthetic files itself.

## Files

| File | Purpose |
|---|---|
| `run_all.py` | data preparation, models, reference models, ablation, repeated seeds, statistical tests, time-series CV, peak-load rule |
| `peak_reduction.py` | forecast-triggered peak-load reduction on measured load, flagging accuracy, threshold sensitivity |
| `daily_meter_check.py` | optional daily check against utility meter registers (TOU Rate A/B/C): load level, daily peak-load reduction scenario and trial-day check; `--forecast-check` adds an exploratory daily forecasting check that is not part of the main analysis |
| `plots.py` | plots the figures (called from the notebook) |
| `make_synthetic_data.py` | synthetic hourly file `synthetic_hourly.csv` (2022–2023, 10,000 hourly rows with random gaps), daily meter and daily production files |

## Input data

Hourly file (`--data`), one row per hour:

| Column | Meaning | Unit |
|---|---|---|
| `timestamp` | date and hour, `dd/mm/yyyy HH:MM` | – |
| `EN` | electricity consumption (target) | kWh |
| `RH` | outdoor relative humidity | % |
| `Air temp` | outdoor air temperature | °C |
| `Process`, `Paste`, `Cold` | hourly output of three production areas | kg |
| `season` | production season label (`winter`, `spring`, `autumn`) | – |
| `CL` | cooling load (read but not used as an input, because it is calculated from `EN`) | kW |

Daily meter file (`--daily`): `date, kWh_rateA, kWh_rateB, kWh_rateC, kWh_total` (Rate A = weekday On-Peak, Rate B = weekday Off-Peak, Rate C = weekends and holidays).

Daily production file (`--production`): `date` and one or more `*_output_kg` columns, including `total_output_kg`.

## Protocol

* Chronological split: first 80 % of hours for training, last 20 % for testing; no shuffling.
* LSTM: 64 units, ReLU, 24-hour input window (Min–Max scaled with training-period limits), 20 epochs, batch 32, 10 % validation split.
* XGBoost meta-learner: 100 trees, depth 6, learning rate 0.1; inputs = weather, monthly mean temperature (training period), temperature anomaly, production, calendar and shift indicators, weekday On-Peak flag, public-holiday flag, season, EN at t−1 and t−24, and the LSTM prediction.
* Separate models for the day shift (06:00–18:59) and night shift (19:00–05:59).
* Reference models: persistence (t−1), seasonal naive (t−24), standalone LSTM, standalone XGBoost, reciprocal-error weighting of LSTM and XGBoost (Chen et al., 2019; weights from the last 10 % of the training period).
* Five seeds (42, 0, 1, 2, 3); Diebold–Mariano test (HLN correction, Newey–West variance, 24 lags) and Wilcoxon signed-rank test.
* Peak-load rule: in weekday On-Peak hours (09:00–21:59, not public holidays) in which the forecast exceeds the threshold (default 2,200 kWh), the **measured** load is reduced by α (5, 10, 15 %).

## Outputs

| Folder / file | Content |
|---|---|
| `results/metrics_models.csv` | model comparison, whole test period (mean ± SD over seeds) |
| `results/metrics_models_by_shift.csv` | the same by shift |
| `results/metrics_ablation.csv`, `metrics_ablation_by_shift.csv` | ablation |
| `results/significance_tests.csv` | Diebold–Mariano and Wilcoxon tests |
| `results/timeseries_cv.csv` | expanding-window CV of the meta-learner |
| `results/exceedance_flagging.csv`, `peak_reduction_test.csv` | flagging accuracy and peak-load reduction |
| `results/summary.txt` | full log |
| `results_peak/` | peak-load reduction tables and figures |
| `results_meter/` | daily meter check |
| `figures/` | figures (PNG, 300 dpi) |

`results/test_predictions_seed42.csv` and `results_peak/peak_reduction_hourly.csv` contain row-level data. Do not share them when the code is run on confidential data.

## Notes

* MAPE is inflated by near-idle hours (EN < 500 kWh), so RMSE, MAE and R² are the main criteria.
* LSTM-based results can differ slightly between TensorFlow versions and hardware even with fixed seeds; XGBoost and the naive models are deterministic. The versions used are listed in `requirements.txt`.

## Licence

All rights reserved; see `LICENSE`. The code may be viewed, downloaded and run to review or reproduce the analysis. Any other use, modification or redistribution requires the authors' written permission.
