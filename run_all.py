"""
Hybrid LSTM-XGBoost forecasting of hourly electricity consumption in a food processing plant
Reproducibility script

What it does (all in one run):
  1. Prepares features from the plant file (plant_hourly.csv schema, see README).
  2. Splits the hourly series chronologically: first 80 % training, last 20 % testing.
  3. Trains, for each shift (day 06:00-18:59, night 19:00-05:59):
       - LSTM on the previous 24 hourly EN values,
       - XGBoost meta-learner on structured features + the LSTM prediction (stacked hybrid).
  4. Reference models: persistence (lag-1), seasonal naive (lag-24), standalone LSTM,
     standalone XGBoost, Chen et al. (2019) reciprocal-error weighting.
  5. Ablation: no shift split / no temperature anomaly / no LSTM meta-feature.
  6. Repeats every neural model over several random seeds (mean +/- SD).
  7. Diebold-Mariano (HLN-corrected, squared loss, Newey-West HAC) and Wilcoxon signed-rank
     tests of the hybrid against each competitor.
  8. Time-ordered (expanding-window) cross-validation for the XGBoost meta-learner.
  9. Peak-load reduction rule on the MEASURED test-period load (5 %, 10 %, 15 %) and
     accuracy of the hybrid forecast in flagging threshold exceedances.
 10. Figures and CSV tables in the output folder.

Usage
  python run_all.py --data data/plant_hourly.csv --out results
  python run_all.py --data data/synthetic_hourly.csv --out results_demo --quick

Cooling load (CL) is not used as an input, because in the plant file it is largely calculated
from EN.
"""

import argparse
import json
import os
import random
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import holidays  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit  # noqa: E402
from sklearn.preprocessing import MinMaxScaler  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

# ----------------------------------------------------------------------------- settings
SEQ_LEN = 24
LSTM_UNITS = 64
LSTM_EPOCHS = 20
LSTM_BATCH = 32
LSTM_VAL_SPLIT = 0.10
XGB_PARAMS = dict(n_estimators=100, max_depth=6, learning_rate=0.10)
SEEDS = [42, 0, 1, 2, 3]
TRAIN_FRACTION = 0.80
DAY_HOURS = range(6, 19)            # 06:00-18:59 = day shift
ON_PEAK_HOURS = range(9, 22)        # weekday TOU On-Peak 09:00-21:59
THRESHOLD_KWH = 2200.0              # development-period peak-load threshold
ALPHAS = [0.05, 0.10, 0.15]
TARIFF_THB = 5.0
CV_GRID = dict(n_estimators=[100, 200, 300], max_depth=[3, 4, 6, 8], learning_rate=[0.05, 0.10])

FEATURES = [
    "RH", "air_temp", "monthly_avg_temp", "temp_anomaly",
    "Process", "Paste", "Cold",
    "hour", "month", "dow", "daytime", "on_peak", "holiday",
    "autumn", "spring", "winter",
    "lag1", "lag24",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


class Log:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, msg=""):
        print(msg)
        self.f.write(str(msg) + "\n")
        self.f.flush()


# ----------------------------------------------------------------------------- data
def load_plant_file(path, impute="mean_train"):
    raw = pd.read_csv(path, encoding="utf-8-sig")
    raw.columns = [c.strip() for c in raw.columns]
    raw = raw.dropna(subset=["timestamp"]).copy()
    raw["dt"] = pd.to_datetime(raw["timestamp"].astype(str).str.strip(), dayfirst=True, errors="coerce")
    raw = raw.dropna(subset=["dt"]).sort_values("dt").drop_duplicates("dt").reset_index(drop=True)
    d = pd.DataFrame({"dt": raw["dt"], "EN": raw["EN"], "RH": raw["RH"], "air_temp": raw["Air temp"],
                      "Process": raw["Process"], "Paste": raw["Paste"], "Cold": raw["Cold"],
                      "season": raw["season"].astype(str).str.strip().str.lower()})
    n = len(d)
    cut = int(n * TRAIN_FRACTION)
    train_mask = np.arange(n) < cut
    missing = d.isna().sum()
    num = ["EN", "RH", "air_temp", "Process", "Paste", "Cold"]
    if impute == "interpolate":
        d[num] = d.set_index("dt")[num].interpolate(method="time").bfill().ffill().values
    else:  # training-period means only
        d[num] = d[num].fillna(d.loc[train_mask, num].mean())
    t = d["dt"]
    th = holidays.Thailand(years=sorted(t.dt.year.unique().tolist()))
    d["holiday"] = t.dt.date.map(lambda x: int(x in th))
    d["hour"] = t.dt.hour
    d["month"] = t.dt.month
    d["dow"] = t.dt.dayofweek
    d["daytime"] = t.dt.hour.isin(DAY_HOURS).astype(int)
    d["on_peak"] = ((d["dow"] < 5) & d["hour"].isin(ON_PEAK_HOURS) & (d["holiday"] == 0)).astype(int)
    # monthly mean temperature from the training period only
    mtemp = d.loc[train_mask].groupby("month")["air_temp"].mean()
    d["monthly_avg_temp"] = d["month"].map(mtemp).fillna(d.loc[train_mask, "air_temp"].mean())
    d["temp_anomaly"] = d["air_temp"] - d["monthly_avg_temp"]
    for s in ["autumn", "spring", "winter"]:
        d[s] = (d["season"] == s).astype(int)
    # lags on the true hourly clock (a gap gives the last available value)
    s = d.set_index("dt")["EN"]
    d["lag1"] = s.shift(1, freq="h").reindex(d["dt"]).values
    d["lag24"] = s.shift(24, freq="h").reindex(d["dt"]).values
    d["lag1"] = d["lag1"].fillna(d["EN"].shift(1))
    d["lag24"] = d["lag24"].fillna(d["EN"].shift(24))
    return d, cut, missing


class Data:
    def __init__(self, d, cut):
        self.full = d
        self.target_full = d["EN"].values.astype(float)
        self.scaler = MinMaxScaler().fit(self.target_full[:cut].reshape(-1, 1))
        s = self.scaler.transform(self.target_full.reshape(-1, 1)).ravel()
        n = len(d)
        self.Xseq = np.stack([s[i - SEQ_LEN:i] for i in range(SEQ_LEN, n)])[..., None]
        self.df = d.iloc[SEQ_LEN:].reset_index(drop=True)
        self.y = self.df["EN"].values.astype(float)
        self.y_s = s[SEQ_LEN:]
        self.train = (np.arange(SEQ_LEN, n) < cut)
        self.test = ~self.train
        self.day = self.df["daytime"].values == 1


# ----------------------------------------------------------------------------- models
def build_lstm():
    m = tf.keras.Sequential([tf.keras.layers.Input(shape=(SEQ_LEN, 1)),
                             tf.keras.layers.LSTM(LSTM_UNITS, activation="relu"),
                             tf.keras.layers.Dense(1)])
    m.compile(optimizer="adam", loss="mse")
    return m


def groups(D, split):
    return [(1, D.day), (0, ~D.day)] if split else [("all", np.ones(len(D.y), bool))]


def lstm_pred(D, seed, split=True, fit=None):
    fit = D.train if fit is None else fit
    out = np.zeros(len(D.y))
    for _, g in groups(D, split):
        set_seed(seed)
        m = build_lstm()
        m.fit(D.Xseq[fit & g], D.y_s[fit & g], epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH,
              validation_split=LSTM_VAL_SPLIT, verbose=0, shuffle=True)
        out[g] = m.predict(D.Xseq[g], verbose=0).ravel()
    return D.scaler.inverse_transform(out.reshape(-1, 1)).ravel()


def xgb_pred(D, feats, seed, split=True, meta=None, fit=None, params=None):
    fit = D.train if fit is None else fit
    X = D.df[feats].copy()
    if meta is not None:
        X["lstm_pred"] = meta
    out = np.zeros(len(D.y))
    for _, g in groups(D, split):
        m = XGBRegressor(random_state=seed, n_jobs=-1, **(params or XGB_PARAMS))
        m.fit(X[fit & g], D.y[fit & g])
        out[g] = m.predict(X[g])
    return out


# ----------------------------------------------------------------------------- metrics & tests
def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = y - p
    nz = np.abs(y) > 1e-9
    return dict(RMSE=float(np.sqrt(np.mean(e ** 2))), MAE=float(np.mean(np.abs(e))),
                MAPE=float(np.mean(np.abs(e[nz] / y[nz])) * 100),
                R2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)))


def dm_test(e1, e2, h=1, hac_lags=None):
    """Diebold-Mariano, squared-error loss, HLN correction. Negative stat: model 1 more accurate."""
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    dbar = d.mean()
    L = (h - 1) if hac_lags is None else hac_lags
    var = np.mean((d - dbar) ** 2)
    for k in range(1, L + 1):
        w = 1.0 if hac_lags is None else 1 - k / (L + 1)
        var += 2 * w * np.mean((d[k:] - dbar) * (d[:-k] - dbar))
    stat = dbar / np.sqrt(var / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(stat), float(2 * stats.t.sf(abs(stat), df=n - 1))


def shift_table(D, preds, mask):
    rows = []
    for name, p in preds.items():
        for sh, g in (("All", np.ones(len(D.y), bool)), ("Day", D.day), ("Night", ~D.day)):
            m = mask & g
            rows.append(dict(Model=name, Shift=sh, **metrics(D.y[m], p[m])))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic_hourly.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true", help="1 seed, 2 epochs, small CV grid (smoke test)")
    a = ap.parse_args()
    global LSTM_EPOCHS, SEEDS, CV_GRID
    if a.quick:
        LSTM_EPOCHS, SEEDS = 2, [42]
        CV_GRID = dict(n_estimators=[100], max_depth=[3, 6], learning_rate=[0.10])
    os.makedirs(a.out, exist_ok=True)
    log = Log(os.path.join(a.out, "summary.txt"))
    t0 = time.time()

    d, cut, missing = load_plant_file(a.data)
    D = Data(d, cut)
    log("=" * 78)
    log("Hybrid LSTM-XGBoost - reproducibility run")
    log("=" * 78)
    log(f"Data: {len(d)} hourly rows, {d.dt.min()} to {d.dt.max()}")
    log(f"Training: {d.dt.iloc[0]} to {d.dt.iloc[cut - 1]} ({cut} rows)")
    log(f"Testing : {d.dt.iloc[cut]} to {d.dt.iloc[-1]} ({len(d) - cut} rows)")
    log(f"Rows used after the 24-h warm-up: train {D.train.sum()}, test {D.test.sum()}")
    log("Missing values in the input file: " + (", ".join(f"{k}={v}" for k, v in missing.items() if v) or "none"))
    ytr = D.y[D.train]
    log(f"EN training period: mean {ytr.mean():.1f}, max {ytr.max():.1f}, min {ytr.min():.1f} kWh")
    log(f"Test hours below 500 kWh (inflate MAPE): {(D.y[D.test] < 500).mean() * 100:.1f} %")

    feats = FEATURES
    feats_noanom = [f for f in feats if f != "temp_anomaly"]
    tr_idx = np.where(D.train)[0]
    fit90 = D.train & (np.arange(len(D.y)) < tr_idx[int(len(tr_idx) * 0.9)])
    val10 = D.train & ~fit90

    # ------------------------------------------------ models over seeds
    log("\n[1] Models over seeds " + str(SEEDS))
    tabs, errs, keep, lp_seed = [], {}, {}, {}
    for seed in SEEDS:
        log(f"  seed {seed}")
        lp = lstm_pred(D, seed, split=True)
        lp_seed[seed] = lp
        xo = xgb_pred(D, feats, seed, split=True)
        hy = xgb_pred(D, feats, seed, split=True, meta=lp)
        lp_all = lstm_pred(D, seed, split=False)
        a_nosplit = xgb_pred(D, feats, seed, split=False, meta=lp_all)
        a_noanom = xgb_pred(D, feats_noanom, seed, split=True, meta=lp)
        l90 = lstm_pred(D, seed, split=True, fit=fit90)
        x90 = xgb_pred(D, feats, seed, split=True, fit=fit90)
        chen = np.zeros(len(D.y))
        for sh, g in ((1, D.day), (0, ~D.day)):
            el = metrics(D.y[val10 & g], l90[val10 & g])["MAPE"]
            ex = metrics(D.y[val10 & g], x90[val10 & g])["MAPE"]
            w = (1 / el) / (1 / el + 1 / ex)
            chen[g] = w * lp[g] + (1 - w) * xo[g]
            if seed == SEEDS[0]:
                log(f"    Chen weights ({'day' if sh else 'night'}): LSTM {w:.3f}, XGBoost {1 - w:.3f}")
        P = {"Persistence (lag-1)": D.df["lag1"].values, "Seasonal naive (lag-24)": D.df["lag24"].values,
             "LSTM": lp, "XGBoost": xo, "Chen et al. weighting": chen, "Hybrid LSTM-XGBoost": hy,
             "Ablation: no shift split": a_nosplit, "Ablation: no temperature anomaly": a_noanom,
             "Ablation: no LSTM meta-feature": xo}
        t = shift_table(D, P, D.test)
        t["seed"] = seed
        tabs.append(t)
        errs[seed] = {k: D.y - v for k, v in P.items()}
        if seed == SEEDS[0]:
            keep = dict(lstm=lp, hybrid=hy)
            pred_out = pd.DataFrame({"dt": D.df["dt"], "measured": D.y, "shift": np.where(D.day, "day", "night"),
                                     **{k: v for k, v in P.items() if not k.startswith("Ablation")}})[D.test]
            pred_out.to_csv(os.path.join(a.out, "test_predictions_seed42.csv"), index=False)

    res = pd.concat(tabs)
    res.to_csv(os.path.join(a.out, "all_seeds_metrics.csv"), index=False)
    agg = res.groupby(["Model", "Shift"], sort=False)[["RMSE", "MAE", "MAPE", "R2"]].agg(["mean", "std"]).fillna(0.0)
    agg.to_csv(os.path.join(a.out, "table_mean_sd.csv"))

    def cell(r, c, d1):
        d = 3 if c == "R2" else d1
        m, sd = r[(c, "mean")], r[(c, "std")]
        return f"{m:.{d}f}" if sd == 0 else f"{m:.{d}f} ± {sd:.{d}f}"

    def fmt(name, d1=2, shifts=("All", "Day", "Night")):
        rows = []
        for sh in shifts:
            r = agg.loc[(name, sh)]
            rows.append(dict(Model=name, Shift=sh, **{c: cell(r, c, d1) for c in ("RMSE", "MAE", "MAPE", "R2")}))
        return rows

    main_models = ["Persistence (lag-1)", "Seasonal naive (lag-24)", "LSTM", "XGBoost",
                   "Chen et al. weighting", "Hybrid LSTM-XGBoost"]
    abl_models = ["Hybrid LSTM-XGBoost", "Ablation: no shift split", "Ablation: no temperature anomaly",
                  "Ablation: no LSTM meta-feature"]
    t4 = pd.DataFrame(sum([fmt(m) for m in main_models], []))
    t6 = pd.DataFrame(sum([fmt(m) for m in abl_models], []))
    t4.to_csv(os.path.join(a.out, "metrics_models_by_shift.csv"), index=False)
    t6.to_csv(os.path.join(a.out, "metrics_ablation_by_shift.csv"), index=False)
    # compact tables: whole test period (both shifts), mean (± SD over seeds)
    t4c = pd.DataFrame(sum([fmt(m, shifts=("All",)) for m in main_models], [])).drop(columns="Shift")
    t4c.to_csv(os.path.join(a.out, "metrics_models.csv"), index=False)
    full = agg.loc[("Hybrid LSTM-XGBoost", "All"), ("RMSE", "mean")]
    t6c = pd.DataFrame(sum([fmt(m, shifts=("All",)) for m in abl_models], [])).drop(columns=["Shift", "MAPE"])
    t6c["Change in RMSE (%)"] = [round((agg.loc[(m, "All"), ("RMSE", "mean")] - full) / full * 100, 1) for m in abl_models]
    t6c.to_csv(os.path.join(a.out, "metrics_ablation.csv"), index=False)
    log("\nModel comparison - whole test period, mean ± SD over seeds (RMSE, MAE in kWh; MAPE in %)")
    log(t4c.to_string(index=False))
    log("\nAblation - whole test period")
    log(t6c.to_string(index=False))
    log("\nModel comparison by shift")
    log(t4.to_string(index=False))
    log("\nAblation by shift")
    log(t6.to_string(index=False))
    log("Note: XGBoost and the naive models are deterministic, so their SD is zero.")

    # ------------------------------------------------ significance tests
    log("\n[2] Diebold-Mariano and Wilcoxon tests: hybrid vs competitor (test period)")
    rows = []
    for sh, g in (("All", np.ones(len(D.y), bool)), ("Day", D.day), ("Night", ~D.day)):
        m = D.test & g
        for c in ["Persistence (lag-1)", "Seasonal naive (lag-24)", "LSTM", "XGBoost", "Chen et al. weighting"]:
            e1, e2 = errs[SEEDS[0]]["Hybrid LSTM-XGBoost"][m], errs[SEEDS[0]][c][m]
            dm, p = dm_test(e1, e2)
            dmh, ph = dm_test(e1, e2, hac_lags=24)
            w = stats.wilcoxon(np.abs(e1), np.abs(e2))
            nsig = sum(dm_test(errs[s]["Hybrid LSTM-XGBoost"][m], errs[s][c][m], hac_lags=24)[1] < 0.05 for s in SEEDS)
            rows.append(dict(Shift=sh, Competitor=c, n=int(m.sum()), DM=round(dm, 3), p_DM=p,
                             DM_HAC24=round(dmh, 3), p_DM_HAC24=ph, Wilcoxon_p=float(w.pvalue),
                             seeds_with_p_below_005=f"{nsig}/{len(SEEDS)}"))
    t7 = pd.DataFrame(rows)
    t7.to_csv(os.path.join(a.out, "significance_tests.csv"), index=False)
    log(t7.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    log("Negative DM: the hybrid has the smaller squared error. Statistics shown for seed "
        f"{SEEDS[0]}; last column counts seeds with HAC p < 0.05.")

    # ------------------------------------------------ time-ordered CV
    log("\n[3] Expanding-window time-series CV for the XGBoost meta-learner (training period only)")
    X = D.df[feats].copy()
    X["lstm_pred"] = keep["lstm"]
    tuned = np.zeros(len(D.y))
    cv_rows = []
    for sh, g in (("Day", D.day), ("Night", ~D.day)):
        gs = GridSearchCV(XGBRegressor(random_state=42, n_jobs=-1), CV_GRID, cv=TimeSeriesSplit(n_splits=5),
                          scoring="neg_root_mean_squared_error", n_jobs=-1)
        gs.fit(X[D.train & g], D.y[D.train & g])
        tuned[g] = gs.best_estimator_.predict(X[g])
        cv_rows.append(dict(Shift=sh, **gs.best_params_, cv_RMSE=round(-gs.best_score_, 2)))
        log(f"  {sh}: best {gs.best_params_}, CV RMSE {-gs.best_score_:.2f}")
    pd.DataFrame(cv_rows).to_csv(os.path.join(a.out, "timeseries_cv.csv"), index=False)
    log(shift_table(D, {"Hybrid, fixed settings": keep["hybrid"], "Hybrid, CV-selected": tuned},
                    D.test).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # CV-selected settings repeated over all seeds (same LSTM predictions as the main runs)
    best = {r["Shift"]: {k: r[k] for k in CV_GRID} for r in cv_rows}
    cv_tabs, cv_rows_dm = [], []
    for seed in SEEDS:
        Xs = D.df[feats].copy()
        Xs["lstm_pred"] = lp_seed[seed]
        hy_cv = np.zeros(len(D.y))
        for sh, g in (("Day", D.day), ("Night", ~D.day)):
            m = XGBRegressor(random_state=seed, n_jobs=-1, **best[sh])
            m.fit(Xs[D.train & g], D.y[D.train & g])
            hy_cv[g] = m.predict(Xs[g])
        t = shift_table(D, {"Hybrid, CV-selected": hy_cv}, D.test)
        t["seed"] = seed
        cv_tabs.append(t)
        mt = D.test
        e_cv = (D.y - hy_cv)[mt]
        for comp in ("LSTM", "Hybrid LSTM-XGBoost"):
            dm, p = dm_test(e_cv, errs[seed][comp][mt], hac_lags=24)
            cv_rows_dm.append(dict(seed=seed, Competitor=comp, DM_HAC24=round(dm, 3), p_DM_HAC24=p))
    cvs = pd.concat(cv_tabs)
    cvs.to_csv(os.path.join(a.out, "timeseries_cv_all_seeds.csv"), index=False)
    cvd = pd.DataFrame(cv_rows_dm)
    cvd.to_csv(os.path.join(a.out, "timeseries_cv_dm.csv"), index=False)
    cva = cvs.groupby("Shift", sort=False)[["RMSE", "MAE", "MAPE", "R2"]].agg(["mean", "std"]).fillna(0.0)
    log("\n  CV-selected settings over seeds " + str(SEEDS) + " (mean ± SD, test period)")
    for sh, r in cva.iterrows():
        log(f"    {sh:5s} RMSE {r[('RMSE','mean')]:.2f} ± {r[('RMSE','std')]:.2f}  MAE {r[('MAE','mean')]:.2f} ± {r[('MAE','std')]:.2f}"
            f"  MAPE {r[('MAPE','mean')]:.2f} ± {r[('MAPE','std')]:.2f}  R2 {r[('R2','mean')]:.3f} ± {r[('R2','std')]:.3f}")
    for comp in ("LSTM", "Hybrid LSTM-XGBoost"):
        sub = cvd[cvd.Competitor == comp]
        nneg = int(((sub.p_DM_HAC24 < 0.05) & (sub.DM_HAC24 < 0)).sum())
        npos = int(((sub.p_DM_HAC24 < 0.05) & (sub.DM_HAC24 > 0)).sum())
        log(f"    DM (HAC 24) CV-selected hybrid vs {comp}: significantly lower error in {nneg}/{len(SEEDS)} seeds, "
            f"significantly higher in {npos}/{len(SEEDS)} seeds")

    # ------------------------------------------------ imputation sensitivity
    if missing.sum() > 0:
        log("\n[4] Imputation sensitivity (seed 42)")
        d2, cut2, _ = load_plant_file(a.data, impute="interpolate")
        D2 = Data(d2, cut2)
        h2 = xgb_pred(D2, feats, 42, meta=lstm_pred(D2, 42))
        log(shift_table(D2, {"Hybrid, time interpolation": h2}, D2.test).to_string(index=False))
    else:
        log("\n[4] No missing values in the input file; imputation has no effect.")

    # ------------------------------------------------ threshold and DSM on measured test load
    log("\n[5] Peak-load threshold and reduction rule (measured load, test period)")
    on_tr = D.train & (D.df["on_peak"].values == 1)
    on_te = D.test & (D.df["on_peak"].values == 1)
    pct = stats.percentileofscore(D.y[on_tr], THRESHOLD_KWH)
    log(f"  {THRESHOLD_KWH:,.0f} kWh = {pct:.1f}th percentile of training weekday On-Peak load")
    log(f"  Share of test weekday On-Peak hours above threshold: {(D.y[on_te] > THRESHOLD_KWH).mean() * 100:.1f} %")
    y, f = D.y, keep["hybrid"]
    dates = D.df["dt"].dt.date.values
    base_peak = pd.Series(y[D.test]).groupby(dates[D.test]).max()
    dsm_rows = []
    for alpha in ALPHAS:
        for trig, lab in ((y > THRESHOLD_KWH, "measured exceeds threshold"),
                          (f > THRESHOLD_KWH, "forecast exceeds threshold")):
            flag = on_te & trig
            adj = np.where(flag, y * (1 - alpha), y)
            red = (y - adj)[D.test].sum()
            new_peak = pd.Series(adj[D.test]).groupby(dates[D.test]).max()
            dsm_rows.append(dict(alpha=alpha, trigger=lab, flagged_hours=int(flag.sum()),
                                 total_kWh=round(y[D.test].sum(), 1), reduction_kWh=round(red, 1),
                                 reduction_pct=round(red / y[D.test].sum() * 100, 2),
                                 mean_daily_peak_reduction_pct=round(((base_peak - new_peak) / base_peak).mean() * 100, 2),
                                 saving_THB=round(red * TARIFF_THB)))
    t9 = pd.DataFrame(dsm_rows)
    t9.to_csv(os.path.join(a.out, "peak_reduction_test.csv"), index=False)
    log(t9.to_string(index=False))
    T, Pf = y[on_te] > THRESHOLD_KWH, f[on_te] > THRESHOLD_KWH
    tp, fp, fn, tn = int((T & Pf).sum()), int((~T & Pf).sum()), int((T & ~Pf).sum()), int((~T & ~Pf).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    t8 = dict(TP=tp, FP=fp, FN=fn, TN=tn, precision=round(prec, 3), recall=round(rec, 3),
              F1=round(2 * prec * rec / (prec + rec), 3) if tp else float("nan"),
              accuracy=round((tp + tn) / len(T), 3), false_alarm_rate=round(fp / (fp + tn), 3) if fp + tn else float("nan"))
    pd.DataFrame([t8]).to_csv(os.path.join(a.out, "exceedance_flagging.csv"), index=False)
    log("  Forecast flagging of exceedances (test weekday On-Peak hours): " + json.dumps(t8))

    # ------------------------------------------------ figures
    figures(D, keep, a.out, log)
    log(f"\nDone in {(time.time() - t0) / 60:.1f} min. Outputs in {a.out}/")
    return dict(D=D, keep=keep, feats=feats, agg=agg)


def figures(D, keep, out, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tr = D.df[D.train]
    wk = tr[tr["dow"] < 5]
    g = wk.groupby("hour")["EN"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.axvspan(9, 22, color="0.92", label="Weekday On-Peak (09:00-22:00)")
    ax.fill_between(g.mean().index, g.quantile(0.1), g.quantile(0.9), alpha=0.25, label="10th-90th percentile")
    ax.plot(g.mean().index, g.mean().values, marker="o", ms=3, label="Weekday mean")
    ax.axhline(THRESHOLD_KWH, color="tab:red", lw=1, label=f"Threshold {THRESHOLD_KWH:,.0f} kWh")
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Hourly consumption (kWh)"); ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(out, "training_profile.png"), dpi=300); plt.close(fig)

    te = D.test
    for sh, gm in (("day", D.day), ("night", ~D.day)):
        m = te & gm
        fig, ax = plt.subplots(figsize=(10, 3.2))
        ax.plot(D.df["dt"][m], D.y[m], lw=0.7, label="Measured")
        ax.plot(D.df["dt"][m], keep["hybrid"][m], lw=0.7, label="Hybrid forecast")
        ax.set_ylabel("kWh"); ax.legend(fontsize=7); fig.tight_layout()
        fig.savefig(os.path.join(out, f"test_{sh}.png"), dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(D.y[te], keep["hybrid"][te], s=4, alpha=0.4)
    lim = [0, max(D.y[te].max(), keep["hybrid"][te].max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1, label="y = x")
    ax.axhline(D.y[D.train].max(), color="tab:red", lw=1, label="Training maximum")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel("Measured (kWh)"); ax.set_ylabel("Hybrid forecast (kWh)")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(out, "scatter_test_quick.png"), dpi=300); plt.close(fig)
    log("  Figures written.")


if __name__ == "__main__":
    main()
