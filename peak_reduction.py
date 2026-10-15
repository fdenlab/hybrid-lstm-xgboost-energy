"""
Peak-load reduction scenario: forecast-triggered peak-load reduction (weekday TOU On-Peak)

Decision rule, evaluated hour by hour on the test period:
    flag_t  = 1  if  t is a weekday On-Peak hour (09:00-21:59, not a public holiday)
                 and the hybrid forecast F_t > E_th
    E_adj,t = E_t (1 - alpha)   if flag_t = 1
    E_adj,t = E_t               otherwise
E_t is the MEASURED consumption, F_t the one-step-ahead hybrid forecast, alpha the reduction
ratio and E_th the threshold (default 2,200 kWh, reported with its percentile in the training
data). The forecast only decides WHEN to act; savings are computed on measured load.
An oracle version (flag from measured load) is reported for comparison.

Outputs (in --out): peak_reduction_summary.csv, peak_reduction_monthly.csv, peak_reduction_flagging.csv,
peak_reduction_hourly.csv (row-level, confidential if run on plant data), and figures.

Usage
  python peak_reduction.py --data data/plant_hourly.csv --out results_peak
  python peak_reduction.py --data data/synthetic_hourly.csv --out results_peak_demo --quick
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

import run_all as R


def evaluate(df, alphas, th, tariff_on, tariff_off):
    on = df["on_peak"].values == 1
    y, f = df["measured"].values, df["forecast"].values
    dates = df["dt"].dt.date.values
    months = df["dt"].dt.to_period("M").values
    base_daily_peak = pd.Series(y).groupby(dates).max()
    base_monthly_max = pd.Series(y).groupby(months).max()
    cost = np.where(on, tariff_on, tariff_off)
    rows, monthly = [], []
    for a in alphas:
        for trig_name, trig in (("forecast", f > th), ("oracle (measured)", y > th)):
            flag = on & trig
            adj = np.where(flag, y * (1 - a), y)
            red = y - adj
            dp = pd.Series(adj).groupby(dates).max()
            mm = pd.Series(adj).groupby(months).max()
            rows.append(dict(
                alpha=a, trigger=trig_name, flagged_hours=int(flag.sum()),
                flagged_days=int(pd.Series(flag).groupby(dates).any().sum()),
                baseline_kWh=round(y.sum(), 1), scenario_kWh=round(adj.sum(), 1),
                reduction_kWh=round(red.sum(), 1), reduction_pct=round(red.sum() / y.sum() * 100, 2),
                onpeak_reduction_pct=round(red[on].sum() / y[on].sum() * 100, 2),
                mean_daily_peak_reduction_pct=round(((base_daily_peak - dp) / base_daily_peak).mean() * 100, 2),
                mean_monthly_max_reduction_pct=round(((base_monthly_max - mm) / base_monthly_max).mean() * 100, 2),
                energy_cost_saving_THB=round((red * cost).sum()),
            ))
            if trig_name == "forecast":
                m = pd.DataFrame({"month": months, "base": y, "adj": adj}).groupby("month").sum()
                m["alpha"] = a
                m["reduction_kWh"] = m["base"] - m["adj"]
                m["reduction_pct"] = m["reduction_kWh"] / m["base"] * 100
                monthly.append(m.reset_index())
    return pd.DataFrame(rows), pd.concat(monthly)


def flagging(df, th):
    on = df["on_peak"].values == 1
    T = df["measured"].values[on] > th
    P = df["forecast"].values[on] > th
    tp, fp = int((T & P).sum()), int((~T & P).sum())
    fn, tn = int((T & ~P).sum()), int((~T & ~P).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(onpeak_hours=int(on.sum()), TP=tp, FP=fp, FN=fn, TN=tn, precision=round(prec, 3),
                recall=round(rec, 3), F1=round(2 * prec * rec / max(prec + rec, 1e-9), 3),
                accuracy=round((tp + tn) / max(len(T), 1), 3), false_alarm_rate=round(fp / max(fp + tn, 1), 3))


def threshold_sensitivity(df, y_train_onpeak, alpha=0.10,
                          percentiles=(55, 60, 64.4, 70, 75), fixed=(2000, 2100, 2200, 2300, 2400)):
    """Repeat the rule for several thresholds (training percentiles and fixed values)."""
    on = df["on_peak"].values == 1
    y, f = df["measured"].values, df["forecast"].values
    dates = df["dt"].dt.date.values
    bp = pd.Series(y).groupby(dates).max()
    ths = sorted(set([round(float(np.percentile(y_train_onpeak, p))) for p in percentiles] + list(fixed)))
    rows = []
    for th in ths:
        T, P = y[on] > th, f[on] > th
        tp, fp, fn = int((T & P).sum()), int((~T & P).sum()), int((T & ~P).sum())
        flag = on & (f > th)
        adj = np.where(flag, y * (1 - alpha), y)
        npk = pd.Series(adj).groupby(dates).max()
        rows.append(dict(threshold_kWh=th, train_percentile=round(stats.percentileofscore(y_train_onpeak, th), 1),
                         flagged_hours=int(flag.sum()), reduction_pct=round((y - adj).sum() / y.sum() * 100, 2),
                         mean_daily_peak_reduction_pct=round(((bp - npk) / bp).mean() * 100, 2),
                         precision=round(tp / max(tp + fp, 1), 3), recall=round(tp / max(tp + fn, 1), 3)))
    return pd.DataFrame(rows)


def figures(df, th, alpha, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    on = df["on_peak"].values == 1
    flag = on & (df["forecast"].values > th)
    df = df.assign(scenario=np.where(flag, df["measured"] * (1 - alpha), df["measured"]))
    wk = df[df["dt"].dt.dayofweek < 5]
    prof = wk.groupby(wk["dt"].dt.hour)[["measured", "forecast", "scenario"]].mean()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.axvspan(9, 22, color="0.93", label="On-Peak 09:00-22:00")
    ax.plot(prof.index, prof["measured"], "o-", ms=3, label="Measured (baseline)")
    ax.plot(prof.index, prof["forecast"], "s-.", ms=3, label="Hybrid forecast")
    ax.plot(prof.index, prof["scenario"], "^--", ms=3, label=f"Peak-load reduction scenario ({int(alpha * 100)} %)")
    ax.axhline(th, color="tab:red", lw=0.8, label=f"Threshold {th:,.0f} kWh")
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Mean weekday consumption (kWh)")
    ax.set_xticks(range(0, 24, 2)); ax.legend(fontsize=7, ncol=2); fig.tight_layout()
    fig.savefig(os.path.join(out, "peak_reduction_weekday_profile.png"), dpi=300); plt.close(fig)

    daily = df.groupby(df["dt"].dt.date).agg(base=("measured", "max"), s1=("scenario", "max"))
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(pd.to_datetime(daily.index), daily["base"], lw=0.9, label="Baseline daily peak")
    ax.plot(pd.to_datetime(daily.index), daily["s1"], lw=0.9, ls="--", label=f"Peak-load reduction scenario daily peak ({int(alpha * 100)} %)")
    ax.set_ylabel("Daily peak (kWh)"); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(os.path.join(out, "peak_reduction_daily_peak.png"), dpi=300); plt.close(fig)

    # example day: weekday with the most flagged hours
    fd = pd.Series(flag).groupby(df["dt"].dt.date.values).sum()
    day = fd.idxmax()
    d = df[df["dt"].dt.date == day]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(d["dt"].dt.hour, d["measured"], "o-", ms=3, label="Measured")
    ax.plot(d["dt"].dt.hour, d["forecast"], "s-.", ms=3, label="Hybrid forecast")
    ax.plot(d["dt"].dt.hour, d["scenario"], "^--", ms=3, label=f"Peak-load reduction scenario ({int(alpha * 100)} %)")
    ax.axhline(th, color="tab:red", lw=0.8)
    ax.set_title(f"Example day {day}", fontsize=9); ax.set_xlabel("Hour"); ax.set_ylabel("kWh")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(os.path.join(out, "peak_reduction_example_day.png"), dpi=300); plt.close(fig)
    return day


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic_hourly.csv")
    ap.add_argument("--out", default="results_peak")
    ap.add_argument("--threshold", type=float, default=R.THRESHOLD_KWH)
    ap.add_argument("--alphas", default="0.05,0.10,0.15")
    ap.add_argument("--tariff_on", type=float, default=R.TARIFF_THB, help="THB/kWh in On-Peak hours")
    ap.add_argument("--tariff_off", type=float, default=R.TARIFF_THB, help="THB/kWh in Off-Peak hours")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        R.LSTM_EPOCHS = 2
    os.makedirs(a.out, exist_ok=True)
    alphas = [float(x) for x in a.alphas.split(",")]

    d, cut, _ = R.load_plant_file(a.data)
    D = R.Data(d, cut)
    lp = R.lstm_pred(D, a.seed)
    fc = R.xgb_pred(D, R.FEATURES, a.seed, meta=lp)
    te = D.test
    df = pd.DataFrame({"dt": D.df["dt"][te].values, "measured": D.y[te], "forecast": fc[te],
                       "on_peak": D.df["on_peak"][te].values, "shift": np.where(D.day[te], "day", "night")})
    df.to_csv(os.path.join(a.out, "peak_reduction_hourly.csv"), index=False)

    on_tr = D.train & (D.df["on_peak"].values == 1)
    pct = stats.percentileofscore(D.y[on_tr], a.threshold)
    summ, monthly = evaluate(df, alphas, a.threshold, a.tariff_on, a.tariff_off)
    fl = flagging(df, a.threshold)
    summ.to_csv(os.path.join(a.out, "peak_reduction_summary.csv"), index=False)
    monthly.to_csv(os.path.join(a.out, "peak_reduction_monthly.csv"), index=False)
    pd.DataFrame([fl]).to_csv(os.path.join(a.out, "peak_reduction_flagging.csv"), index=False)
    sens = threshold_sensitivity(df, D.y[on_tr], alpha=0.10)
    sens.to_csv(os.path.join(a.out, "peak_reduction_threshold_sensitivity.csv"), index=False)
    day = figures(df, a.threshold, 0.10, a.out)

    with open(os.path.join(a.out, "peak_reduction_report.txt"), "w", encoding="utf-8") as f:
        def w(s=""):
            print(s); f.write(s + "\n")
        w("Peak-load reduction scenario - forecast-triggered peak-load reduction")
        w(f"Test period: {df.dt.min()} to {df.dt.max()} ({len(df)} hours)")
        w(f"Threshold {a.threshold:,.0f} kWh = {pct:.1f}th percentile of training weekday On-Peak load")
        w(f"Tariff: On-Peak {a.tariff_on} THB/kWh, Off-Peak {a.tariff_off} THB/kWh (energy charge only)")
        w("\n" + summ.to_string(index=False))
        w("\nForecast flagging of threshold exceedances (weekday On-Peak hours):")
        w(str(fl))
        w("\nThreshold sensitivity (alpha = 0.10, forecast trigger):")
        w(sens.to_string(index=False))
        w(f"\nExample day in peak_reduction_example_day.png: {day}")
        w("\nMonthly results (forecast trigger):")
        w(monthly.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
