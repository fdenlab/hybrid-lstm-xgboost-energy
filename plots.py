"""
Figures (default Matplotlib font; blue measured /
orange dashed forecast, pink On-Peak shading, red dashed threshold).

Use after run_all.main(...) which returns a dict with D, keep, feats and agg:
    res = main(DATA, OUT)
    make_figures(res, out="figures")
Optional: daily_onpeak_csv = meter_daily.csv written by daily_meter_check.py
"""
import os

import numpy as np
import pandas as pd

BLUE, ORANGE, MAGENTA, RED = "#1565C0", "#F57C20", "#C2185B", "#E53935"
BAR_DAY, BAR_NIGHT = "#90C2F5", "#F8B26A"
PINK = "#F8D7DA"


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 12, "axes.linewidth": 1.2, "legend.fontsize": 10,
        "xtick.major.width": 1.0, "ytick.major.width": 1.0, "savefig.dpi": 300,
    })
    return plt


def _save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name + ".png"), dpi=300)


def fig_timeseries(D, out, plt):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(D.full["dt"], D.full["EN"], color=BLUE, lw=0.5)
    cut = D.full["dt"].iloc[int(len(D.full) * 0.8)]
    ax.axvline(cut, color=RED, ls="--", lw=1.2)
    ax.text(cut, ax.get_ylim()[1] * 0.95, "  Test period", color=RED)
    ax.set_xlabel("Date"); ax.set_ylabel("Energy consumption (kWh)")
    _save(fig, out, "time_series"); plt.close(fig)


def fig_actual_vs_pred(D, keep, out, plt, n=300):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    for ax, (lab, g, tag) in zip(axes, (("day", D.day, "(a) Hybrid model performance during day shift"),
                                        ("night", ~D.day, "(b) Hybrid model performance during night shift"))):
        m = np.where(D.test & g)[0][:n]
        ax.plot(range(len(m)), D.y[m], color=BLUE, lw=1.6, label="Actual")
        ax.plot(range(len(m)), keep["hybrid"][m], color=ORANGE, lw=1.6, ls="--", label="Predicted (Hybrid)")
        ax.set_xlabel("Hourly observation in the test period (this shift only)")
        ax.set_ylabel("Energy consumption (kWh)"); ax.legend(loc="upper right")
        ax.set_title(tag, y=-0.33, fontsize=13)
    fig.subplots_adjust(hspace=0.55)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "actual_vs_predicted.png"), dpi=300)
    plt.close(fig)


def fig_metric_bars(agg, out, plt, models=("LSTM", "XGBoost", "Hybrid LSTM-XGBoost")):
    labels = ["LSTM", "XGBoost", "Hybrid"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    for ax, (met, title) in zip(axes.ravel(), (("RMSE", "RMSE"), ("MAE", "MAE"), ("MAPE", "MAPE"), ("R2", "R$^2$"))):
        x = np.arange(len(models))
        d = [agg.loc[(m, "Day"), (met, "mean")] for m in models]
        n = [agg.loc[(m, "Night"), (met, "mean")] for m in models]
        ed = [agg.loc[(m, "Day"), (met, "std")] for m in models]
        en = [agg.loc[(m, "Night"), (met, "std")] for m in models]
        ax.bar(x - 0.18, d, 0.34, yerr=ed, capsize=3, color=BAR_DAY, label="Day")
        ax.bar(x + 0.18, n, 0.34, yerr=en, capsize=3, color=BAR_NIGHT, label="Night")
        ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_title(title, fontsize=15)
        if met == "R2":
            ax.set_ylim(0, 1.0)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(out, "metric_comparison.png"), dpi=300)
    plt.close(fig)


NICE = {"RH": "RH", "air_temp": "Air temp", "monthly_avg_temp": "monthly avg temp", "temp_anomaly": "temp anomaly",
        "Process": "process", "Paste": "paste", "Cold": "cold", "hour": "hour", "month": "month", "dow": "day of week",
        "daytime": "daytime", "on_peak": "on_peak", "holiday": "holiday", "autumn": "autumn", "spring": "spring",
        "winter": "winter", "lag1": "lag1", "lag24": "lag24", "lstm_pred": "LSTM prediction"}


def hybrid_gain(D, feats, lstm, seed=42, params=None):
    """Gain importance of the hybrid XGBoost meta-learner (mean of day and night models)."""
    from xgboost import XGBRegressor
    X = D.df[feats].copy()
    X["lstm_pred"] = lstm
    tot = pd.Series(0.0, index=X.columns)
    for g in (D.day, ~D.day):
        m = XGBRegressor(random_state=seed, n_jobs=-1, n_estimators=100, max_depth=6, learning_rate=0.10,
                         **(params or {}))
        m.fit(X[D.train & g], D.y[D.train & g])
        gain = pd.Series(m.get_booster().get_score(importance_type="gain"))
        tot = tot.add(gain.reindex(X.columns).fillna(0) / 2, fill_value=0)
    return tot.sort_values(ascending=False)


def fig_importance(gain, out, plt, top=12):
    g = gain.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh([NICE.get(k, k) for k in g.index], g.values, color="#64B5F6")
    ax.set_xlabel("Importance Score (gain)"); ax.set_ylabel("Features")
    _save(fig, out, "feature_importance"); plt.close(fig)


def fig_threshold_profile(D, out, plt, th=2200.0):
    tr = D.df[D.train]
    wk = tr[tr["dow"] < 5]
    g = wk.groupby("hour")["EN"]
    mean, lo, hi = g.mean(), g.quantile(0.10), g.quantile(0.90)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.axvspan(9, 22, color=PINK, alpha=0.8, lw=0)
    ax.fill_between(mean.index, lo, hi, color=BLUE, alpha=0.12, lw=0, label="10th–90th percentile")
    ax.plot(mean.index, mean.values, "o-", color=BLUE, lw=1.8, ms=5, label="Mean weekday energy consumption (kWh)")
    ax.axhline(th, color=RED, ls="--", lw=1.4)
    ax.text(0.5, th * 1.01, f"Threshold = {th:,.0f} kWh", color=RED, fontsize=11)
    share = g.apply(lambda v: (v > th).mean() * 100)
    ax.text(15.5, ax.get_ylim()[0] + 0.06 * (ax.get_ylim()[1] - ax.get_ylim()[0]), "On-Peak", color="#3A1B8C", fontsize=14, ha="center")
    ax.set_xticks(range(24)); ax.set_xlabel("Hour of Day"); ax.set_ylabel("Energy consumption (kWh)")
    ax.legend(loc="upper right")
    _save(fig, out, "threshold_profile"); plt.close(fig)
    return share


def _scenario(D, keep, th, alpha):
    te = D.test
    df = pd.DataFrame({"dt": D.df["dt"][te].values, "y": D.y[te], "f": keep["hybrid"][te],
                       "on": D.df["on_peak"][te].values == 1})
    df["s1"] = np.where(df["on"] & (df["f"] > th), df["y"] * (1 - alpha), df["y"])
    return df


def fig_hourly_reduction(D, keep, out, plt, th=2200.0, alpha=0.10):
    df = _scenario(D, keep, th, alpha)
    wk = df[pd.to_datetime(df["dt"]).dt.dayofweek < 5]
    p = wk.groupby(pd.to_datetime(wk["dt"]).dt.hour)[["y", "s1"]].mean()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axvspan(9, 22, color=PINK, alpha=0.8, lw=0)
    ax.plot(p.index, p["y"], "o-", color=BLUE, lw=1.8, ms=5, label="Baseline (measured)")
    ax.plot(p.index, p["s1"], "o:", color=MAGENTA, lw=1.8, ms=5, label=f"After {int(alpha * 100)}% peak reduction")
    ax.axhline(th, color=RED, ls="--", lw=1.4)
    ax.text(0.3, th + 8, f"Threshold = {th:,.0f} kWh", color=RED, fontsize=11)
    ax.text(15.5, ax.get_ylim()[0] + 0.08 * (ax.get_ylim()[1] - ax.get_ylim()[0]), "On-peak", color="#3A1B8C", fontsize=14, ha="center")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, max(hi, th + 0.08 * (th - lo)))
    ax.set_xticks(range(24)); ax.set_xlabel("Hour of day"); ax.set_ylabel("Energy consumption (kWh)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2)
    _save(fig, out, "hourly_peak_reduction"); plt.close(fig)


def fig_example_day(df, out, plt, day=None, th=2200.0, alpha=0.10):
    """One weekday of the test period: measured load, hybrid forecast and load after the rule.
    df needs columns dt, y (measured), f (hybrid forecast), on (weekday On-Peak flag)."""
    df = df.copy()
    df["dt"] = pd.to_datetime(df["dt"])
    if day is None:   # first test day on which the rule flags 4-9 On-Peak hours (else the day with most flags)
        n = df.assign(f=df["on"] & (df["f"] > th)).groupby(df["dt"].dt.date)["f"].sum()
        cand = n[(n >= 4) & (n <= 9)]
        day = (cand.index[0] if len(cand) else n.idxmax())
    g = df[df["dt"].dt.date == pd.Timestamp(day).date()].sort_values("dt")
    if g.empty:
        return None
    h = g["dt"].dt.hour.values
    flag = (g["on"].values) & (g["f"].values > th)
    s1 = np.where(flag, g["y"].values * (1 - alpha), g["y"].values)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axvspan(9, 22, color=PINK, alpha=0.8, lw=0)
    ax.plot(h, g["y"], "o-", color=BLUE, lw=1.8, ms=5, label="Measured load")
    ax.plot(h, g["f"], "s--", color=ORANGE, lw=1.5, ms=4, label="Hybrid forecast")
    ax.plot(h, s1, "o:", color=MAGENTA, lw=1.8, ms=5, label=f"After {int(alpha * 100)}% reduction")
    ax.scatter(h[flag], s1[flag], s=90, facecolors="none", edgecolors=MAGENTA, lw=1.4, zorder=5, label="Flagged hour (forecast > threshold)")
    ax.axhline(th, color=RED, ls="--", lw=1.4)
    ax.text(0.2, th + 0.035 * (ax.get_ylim()[1] - ax.get_ylim()[0]), f"Threshold = {th:,.0f} kWh", color=RED, fontsize=11)
    lo, hi = ax.get_ylim()
    ax.text(15.5, lo + 0.06 * (hi - lo), "On-Peak", color="#3A1B8C", fontsize=14, ha="center")
    ax.set_xticks(range(24)); ax.set_xlabel("Hour of day"); ax.set_ylabel("Energy consumption (kWh)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2)
    fig.tight_layout()
    _save(fig, out, "example_day"); plt.close(fig)
    return int(flag.sum())


def fig_daily_meter(daily_csv, out, plt, percentile=64.4, alpha=0.10, trial_date=None):
    """2025 utility meter: weekday On-Peak energy per day with the daily Peak-load reduction scenario rule."""
    if daily_csv is None or not os.path.exists(daily_csv):
        return None
    d = pd.read_csv(daily_csv, parse_dates=["date"]).set_index("date").sort_index()
    wk = d[d["kWh_rateA"].notna() & (d["kWh_rateA"] > 0)]
    th = np.percentile(wk["kWh_rateA"], percentile)
    s1 = np.where(wk["kWh_rateA"] > th, wk["kWh_rateA"] * (1 - alpha), wk["kWh_rateA"])
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(wk.index, wk["kWh_rateA"], color=BLUE, lw=1.3, label="Measured On-Peak energy (utility meter)")
    ax.plot(wk.index, s1, color=ORANGE, lw=1.3, ls="--", label=f"Peak-load reduction ({int(alpha * 100)}%)")
    ax.axhline(th, color=RED, ls="--", lw=1.2)
    ax.text(wk.index[0], th * 1.01, f"  {percentile}th percentile = {th:,.0f} kWh/day", color=RED, fontsize=10)
    red = (wk["kWh_rateA"] - s1).sum()
    ax.text(0.02, 0.95, f"On-Peak reduction = {red / wk['kWh_rateA'].sum() * 100:.2f}%  |  total = {red / d['kWh_total'].sum() * 100:.2f}%",
            transform=ax.transAxes, va="top", fontsize=11)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.25 * (hi - lo), hi)
    ax.set_xlabel("Date"); ax.set_ylabel("Weekday On-Peak energy (kWh/day)")
    ax.legend(loc="lower right")
    _save(fig, out, "daily_onpeak_meter"); plt.close(fig)
    return th


def fig_scatter(D, keep, out, plt):
    te = D.test
    y, p = D.y[te], keep["hybrid"][te]
    lim = [0, max(y.max(), p.max(), D.y[D.train].max()) * 1.05]
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.scatter(y, p, s=8, alpha=0.45, color=BLUE, edgecolors="none")
    ax.plot(lim, lim, "k--", lw=1.2, label="y = x")
    ax.axhline(D.y[D.train].max(), color=RED, lw=1, ls=":", label=f"Training maximum ({D.y[D.train].max():,.0f} kWh)")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Measured energy consumption (kWh)"); ax.set_ylabel("Predicted energy consumption (kWh)")
    ax.legend(loc="upper left", fontsize=9)
    _save(fig, out, "scatter_test"); plt.close(fig)


def fig_daily_peak(D, keep, out, plt, th=2200.0, alpha=0.10):
    df = _scenario(D, keep, th, alpha)
    day = pd.to_datetime(df["dt"]).dt.date
    dp = df.groupby(day).agg(base=("y", "max"), s1=("s1", "max"))
    red = ((dp["base"] - dp["s1"]) / dp["base"]).mean() * 100
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(pd.to_datetime(dp.index), dp["base"], color=BLUE, lw=1.2, label="Baseline operation")
    ax.plot(pd.to_datetime(dp.index), dp["s1"], color=ORANGE, lw=1.2, ls="--", label="Peak-load reduction strategy")
    ax.text(0.02, 0.95, f"Mean daily peak reduction = {red:.2f}%", transform=ax.transAxes, va="top", fontsize=12)
    ax.set_ylim(0, dp["base"].max() * 1.15)
    ax.set_xlabel("Date"); ax.set_ylabel("Daily peak hourly energy consumption (kWh)")
    ax.legend(loc="upper right")
    _save(fig, out, "daily_peak"); plt.close(fig)
    return red


def make_figures(res, out="figures", threshold=2200.0, alpha=0.10, daily_meter_csv=None):
    os.makedirs(out, exist_ok=True)
    plt = _style()
    D, keep, feats, agg = res["D"], res["keep"], res["feats"], res["agg"]
    fig_timeseries(D, out, plt)
    fig_actual_vs_pred(D, keep, out, plt)
    fig_metric_bars(agg, out, plt)
    gain = hybrid_gain(D, feats, keep["lstm"])
    gain.to_csv(os.path.join(out, "feature_importance_gain.csv"), header=["gain"])
    fig_importance(gain, out, plt)
    share = fig_threshold_profile(D, out, plt, threshold)
    fig_hourly_reduction(D, keep, out, plt, threshold, alpha)
    fig_example_day(_scenario(D, keep, threshold, alpha), out, plt, th=threshold, alpha=alpha)
    fig_daily_meter(daily_meter_csv, out, plt)
    fig_scatter(D, keep, out, plt)
    red = fig_daily_peak(D, keep, out, plt, threshold, alpha)
    print("Figures saved to", out)
    print("Top features (gain):", ", ".join(f"{k} {v:,.0f}" for k, v in gain.head(6).items()))
    print(f"Mean daily peak reduction (test period, {int(alpha * 100)}%): {red:.2f}%")
    return dict(gain=gain, share_above_threshold_by_hour=share, mean_daily_peak_reduction=red)
