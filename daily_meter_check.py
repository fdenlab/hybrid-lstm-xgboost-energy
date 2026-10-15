"""
2025 cross-check with the utility monthly meter reports

The 2025 metering available for this study consists of the utility's monthly reports, which give
DAILY energy per TOU register (Rate A = weekday On-Peak, Rate B = weekday Off-Peak, Rate C =
weekends/holidays). Hourly 2025 values are not available, so this script works at daily level:

  1. Load level: mean daily energy in 2022-2023 (plant file) vs 2025 (meter reports)
  2. Peak-load reduction scenario at daily level on measured 2025 On-Peak energy: on weekdays whose On-Peak energy
     exceeds the chosen percentile, On-Peak energy is reduced by alpha (5, 10, 15 %)
     plus a percentile sensitivity table                                            
  3. Trial day: On-Peak energy and energy per kg of output on the trial date compared with the
     other weekdays of the same month (needs the daily production file)
  4. (optional, --forecast-check) Daily forecasting check for 2025 with production data
     (chronological 80/20 within 2025): persistence, 7-day naive, ridge regression, XGBoost.
     This is an additional exploratory analysis and is not part of the main analysis.

Usage
  python daily_meter_check.py --reports "reports/*.xlsx" --dev data/plant_hourly.csv \
         --production data/daily_production.csv --trial-date YYYY-MM-DD --out results_meter
  (or --daily a CSV with columns date,kWh_rateA,kWh_rateB,kWh_rateC,kWh_total)
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd


def read_reports(pattern):
    import openpyxl
    rows = []
    for f in sorted(glob.glob(pattern)):
        ws = openpyxl.load_workbook(f, data_only=True).active
        for r in ws.iter_rows(min_row=1, values_only=True):
            first = str(r[0]).strip() if r[0] is not None else ""
            if len(first) == 10 and first[2] == "/" and first[5] == "/":
                rows.append(dict(date=pd.to_datetime(first, dayfirst=True), kWh_rateA=r[1], kWh_rateB=r[2],
                                 kWh_rateC=r[3], kWh_total=r[4], source_file=os.path.basename(f)))
    return pd.DataFrame(rows).set_index("date").sort_index()


def load_level(daily, dev_path):
    span = f"{daily.index.min():%b %Y} to {daily.index.max():%b %Y}"
    out = {f"Meter reports ({span}) mean daily kWh": daily["kWh_total"].mean()}
    if dev_path:
        d = pd.read_csv(dev_path, encoding="utf-8-sig")
        d.columns = [c.strip() for c in d.columns]
        d = d.dropna(subset=["timestamp"])
        d["dt"] = pd.to_datetime(d["timestamp"].astype(str).str.strip(), dayfirst=True, errors="coerce")
        g = d.groupby(d["dt"].dt.date)["EN"].agg(["sum", "size"])
        full = g[g["size"] == 24]["sum"]
        out["2022-2023 mean daily kWh (complete days)"] = full.mean()
        out["2022-2023 training-period max hourly kWh"] = d["EN"].max()
        out["ratio meter reports / 2022-2023 (mean daily)"] = daily["kWh_total"].mean() / full.mean()
    return pd.Series(out).round(3)


def peak_reduction_daily(daily, percentile, alphas):
    wk = daily[daily["kWh_rateA"].notna() & (daily["kWh_rateA"] > 0)]
    th = np.percentile(wk["kWh_rateA"], percentile)
    rows = []
    for a in alphas:
        flag = wk["kWh_rateA"] > th
        red = (wk["kWh_rateA"][flag] * a).sum()
        rows.append(dict(alpha=a, percentile=percentile, threshold_onpeak_kWh_per_day=round(th),
                         weekdays=len(wk), flagged_days=int(flag.sum()), reduction_kWh=round(red),
                         reduction_pct_of_total=round(red / daily["kWh_total"].sum() * 100, 2),
                         reduction_pct_of_onpeak=round(red / wk["kWh_rateA"].sum() * 100, 2)))
    return pd.DataFrame(rows)


def trial_day(daily, prod, date):
    date = pd.Timestamp(date)
    m = daily.loc[str(date.to_period("M"))]
    wk = m[m["kWh_rateA"].notna() & (m["kWh_rateA"] > 0)].copy()
    if prod is not None:
        wk = wk.join(prod[["total_output_kg"]], how="left")
        wk["kWh_per_kg"] = wk["kWh_total"] / wk["total_output_kg"].replace(0, np.nan)
        wk["onpeak_kWh_per_kg"] = wk["kWh_rateA"] / wk["total_output_kg"].replace(0, np.nan)
    res = {"trial date": str(date.date()), "weekdays in month": len(wk),
           "trial On-Peak kWh": wk.loc[date, "kWh_rateA"], "other weekdays mean On-Peak kWh": wk.drop(date)["kWh_rateA"].mean(),
           "trial rank by On-Peak kWh (1 = highest)": int((wk["kWh_rateA"] > wk.loc[date, "kWh_rateA"]).sum() + 1)}
    if prod is not None:
        for c in ("kWh_per_kg", "onpeak_kWh_per_kg"):
            res[f"trial {c}"] = wk.loc[date, c]
            res[f"other weekdays mean {c}"] = wk.drop(date)[c].mean()
            res[f"trial rank by {c} (1 = highest)"] = int((wk[c] > wk.loc[date, c]).sum() + 1)
        res["trial total output kg"] = wk.loc[date, "total_output_kg"]
        res["other weekdays mean output kg"] = wk.drop(date)["total_output_kg"].mean()
        o = wk.drop(date).dropna(subset=["total_output_kg"])
        for c in ("kWh_total", "kWh_rateA"):
            b = np.polyfit(o["total_output_kg"], o[c], 1)        # energy vs output, other weekdays
            exp = float(np.polyval(b, wk.loc[date, "total_output_kg"]))
            sd = float((o[c] - np.polyval(b, o["total_output_kg"])).std())
            res[f"{c}: expected from output (linear fit, other weekdays)"] = round(exp)
            res[f"{c}: actual minus expected (%)"] = round((wk.loc[date, c] - exp) / exp * 100, 1)
            res[f"{c}: z-score vs residual SD"] = round((wk.loc[date, c] - exp) / sd, 2)
    return pd.Series(res), wk


def daily_forecast_check(daily, prod, holidays_years=(2025,)):
    import holidays
    from sklearn.linear_model import RidgeCV
    from xgboost import XGBRegressor
    d = prod.join(daily[["kWh_total"]], how="inner", rsuffix="_meter")
    d["y"] = d["kWh_total_meter"] if "kWh_total_meter" in d else d["kWh_total"]
    th = holidays.Thailand(years=list(holidays_years))
    d["dow"] = d.index.dayofweek
    d["weekend"] = (d["dow"] >= 5).astype(int)
    d["holiday"] = [int(x.date() in th) for x in d.index]
    d["lag1"], d["lag2"], d["lag7"] = d["y"].shift(1), d["y"].shift(2), d["y"].shift(7)
    d["roll7_prev"] = d["y"].shift(1).rolling(7, min_periods=3).mean()      # previous days only
    d["out_roll7_prev"] = d["total_output_kg"].shift(1).rolling(7, min_periods=3).mean()
    d = d.dropna(subset=["lag7", "roll7_prev", "out_roll7_prev"])
    prod_cols = [c for c in ("paste_output_kg", "packing_output_kg", "freezing_output_kg", "total_output_kg") if c in d]
    F = prod_cols + ["dow", "weekend", "holiday", "lag1", "lag2", "roll7_prev", "out_roll7_prev"]
    cut = int(len(d) * 0.8)
    tr, te = d.iloc[:cut], d.iloc[cut:]

    def met(y, p):
        e = y - p
        return dict(RMSE=round(float(np.sqrt((e ** 2).mean()))), MAE=round(float(np.abs(e).mean())),
                    MAPE=round(float((np.abs(e) / y).mean() * 100), 2),
                    R2=round(float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()), 3))
    mu, sd = tr[F].mean(), tr[F].std().replace(0, 1)
    ridge = RidgeCV(alphas=np.logspace(-2, 4, 30)).fit((tr[F] - mu) / sd, tr["y"])
    xgb = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42).fit(tr[F], tr["y"])
    rows = [dict(Model="Persistence (previous day)", **met(te["y"], te["lag1"])),
            dict(Model="Naive (same weekday, previous week)", **met(te["y"], te["lag7"])),
            dict(Model="Mean of previous 7 days", **met(te["y"], te["roll7_prev"])),
            dict(Model="Ridge regression (production + lags)", **met(te["y"], ridge.predict((te[F] - mu) / sd))),
            dict(Model="XGBoost (production + lags)", **met(te["y"], xgb.predict(te[F])))]
    span = f"train {tr.index.min().date()} to {tr.index.max().date()} ({len(tr)} days), test {te.index.min().date()} to {te.index.max().date()} ({len(te)} days)"
    return pd.DataFrame(rows), span


def daily_check(out, reports=None, daily_csv=None, dev=None, production=None, trial_date=None,
                percentile=64.4, alphas=(0.05, 0.10, 0.15), forecast_check=False):
    os.makedirs(out, exist_ok=True)
    if reports:
        daily = read_reports(reports)
    else:
        daily = pd.read_csv(daily_csv, parse_dates=["date"]).set_index("date").sort_index()
    for c in ("kWh_rateA", "kWh_rateB", "kWh_rateC", "kWh_total"):
        daily[c] = pd.to_numeric(daily[c], errors="coerce")
    daily.to_csv(os.path.join(out, "meter_daily.csv"))
    prod = None
    if production:
        prod = pd.read_csv(production, parse_dates=["date"]).set_index("date").sort_index()
    lines = []

    def w(s=""):
        print(s)
        lines.append(str(s))
    w("2025 cross-check with utility meter reports (daily TOU registers)")
    w(f"Report days: {len(daily)} ({daily.index.min().date()} to {daily.index.max().date()}), "
      f"total {daily['kWh_total'].sum() / 1e6:.2f} GWh, On-Peak share {daily['kWh_rateA'].sum() / daily['kWh_total'].sum() * 100:.1f} %")

    lv = load_level(daily, dev)
    lv.to_csv(os.path.join(out, "load_level_comparison.csv"), header=["value"])
    w("\n[1] Load level")
    w(lv.to_string())

    s1 = peak_reduction_daily(daily, percentile, alphas)
    s1.to_csv(os.path.join(out, "peak_reduction_daily.csv"), index=False)
    w(f"\n[2] Peak-load reduction scenario at daily level on measured 2025 On-Peak energy (percentile {percentile})")
    w(s1.to_string(index=False))
    sens = pd.concat([peak_reduction_daily(daily, p, [0.10]) for p in (55, 60, 64.4, 70, 75)])
    sens.to_csv(os.path.join(out, "peak_reduction_daily_sensitivity.csv"), index=False)
    w("\n    Percentile sensitivity (alpha = 0.10)")
    w(sens[["percentile", "threshold_onpeak_kWh_per_day", "flagged_days", "reduction_pct_of_total", "reduction_pct_of_onpeak"]].to_string(index=False))

    if trial_date:
        tres, tab = trial_day(daily, prod, trial_date)
        tres.to_csv(os.path.join(out, "trial_day_check.csv"), header=["value"])
        tab.to_csv(os.path.join(out, "trial_month_weekdays.csv"))
        w(f"\n[3] Trial day {trial_date} compared with other weekdays of the month")
        w(tres.to_string())

    if forecast_check and prod is not None:   # exploratory; not part of the main analysis
        fc, span = daily_forecast_check(daily, prod)
        fc.to_csv(os.path.join(out, "daily_forecast_check.csv"), index=False)
        w(f"\n[4] Daily forecasting check within 2025, exploratory, not part of the main analysis ({span})")
        w(fc.to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        wk = daily[daily["kWh_rateA"].notna() & (daily["kWh_rateA"] > 0)]
        th = np.percentile(wk["kWh_rateA"], percentile)
        fig, ax = plt.subplots(figsize=(9, 3.4))
        ax.plot(wk.index, wk["kWh_rateA"], ".-", ms=3, lw=0.7, label="Measured weekday On-Peak energy")
        ax.plot(wk.index, np.where(wk["kWh_rateA"] > th, wk["kWh_rateA"] * 0.9, wk["kWh_rateA"]), lw=0.7, ls="--",
                label="Peak-load reduction scenario (10 %)")
        ax.axhline(th, color="tab:red", lw=0.8, label=f"{percentile}th percentile ({th:,.0f} kWh/day)")
        if trial_date:
            ax.axvline(pd.Timestamp(trial_date), color="0.5", ls=":", lw=1, label="Trial day")
        ax.set_ylabel("kWh per day"); ax.legend(fontsize=7); fig.tight_layout()
        fig.savefig(os.path.join(out, "daily_onpeak_meter_check.png"), dpi=300); plt.close(fig)
    except Exception as ex:
        w(f"(figure skipped: {ex})")
    open(os.path.join(out, "daily_meter_check_report.txt"), "w", encoding="utf-8").write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", help='glob of monthly report files, e.g. "reports/*.xlsx"')
    ap.add_argument("--daily", help="CSV with date,kWh_rateA,kWh_rateB,kWh_rateC,kWh_total")
    ap.add_argument("--dev", help="2022-2023 plant file (plant_hourly.csv) for the load-level comparison")
    ap.add_argument("--production", help="daily production CSV with date and *_output_kg columns")
    ap.add_argument("--trial-date", default=None, help="optional date (YYYY-MM-DD) on which the measure was trialled")
    ap.add_argument("--percentile", type=float, default=64.4, help="percentile of weekday On-Peak days used as threshold")
    ap.add_argument("--forecast-check", action="store_true",
                    help="also run the exploratory daily forecasting check (not part of the main analysis)")
    ap.add_argument("--out", default="results_meter")
    a = ap.parse_args()
    if not (a.reports or a.daily):
        ap.error("give --reports or --daily")
    daily_check(a.out, reports=a.reports, daily_csv=a.daily, dev=a.dev, production=a.production,
                trial_date=a.trial_date, percentile=a.percentile, forecast_check=a.forecast_check)


if __name__ == "__main__":
    main()
