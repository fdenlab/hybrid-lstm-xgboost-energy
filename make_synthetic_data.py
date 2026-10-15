"""Create a synthetic file with the same columns as the confidential plant file.
It lets anyone run run_all.py end to end. The numbers are artificial and do not
reproduce results obtained with the confidential plant data."""
import argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--start", default="2022-01-01 00:00")
ap.add_argument("--end", default="2023-08-25 09:00", help="last hour of the synthetic series")
ap.add_argument("--hours", type=int, default=10000,
                help="number of hourly rows to keep; the rest are removed as random gaps (like meter outages)")
ap.add_argument("--test-start", default="2023-04-26 07:00",
                help="optional timestamp of the first test hour; gaps are placed so that the last 20 %% of rows start here")
a, _ = ap.parse_known_args()   # tolerant of notebook kernels

rng = np.random.default_rng(2026)
t = pd.date_range(a.start, a.end, freq="h")
n = len(t)
h, m, dow = t.hour.values, t.month.values, t.dayofweek.values
season = np.where(np.isin(m, [1, 2, 3]), "winter", np.where(np.isin(m, range(4, 9)), "spring", "autumn"))
temp = 26.8 + 3 * np.sin(2 * np.pi * (m - 4) / 12) + 3 * np.sin(2 * np.pi * (h - 9) / 24) + rng.normal(0, 1, n)
rh = np.clip(70 - 1.5 * (temp - 26.8) + rng.normal(0, 8, n), 30, 100)
campaign = 1 + 0.5 * np.isin(m, [6, 7, 10])                      # seasonal campaign months
work = ((h >= 8) & (h < 20) & (dow < 6)).astype(float)
process = np.clip(rng.normal(15000, 300, n) * (0.98 + 0.04 * work), 8000, 20000)
paste = np.clip(rng.normal(1500, 400, n) * work * campaign, 0, None)
cold = np.clip(rng.normal(1000, 900, n), 500, 15000)
shut = rng.random(n) < (0.015 + 0.01 * ((h >= 19) | (h < 6)))   # unplanned stops occur mostly at night
# observed drivers (production, schedule, weather)
base = 1500 * campaign * (1 + 0.03 * work) + 8 * (temp - 26.8) + 0.03 * paste   # plant profile is almost flat over the day
# unobserved refrigeration state (compressors, defrost, rooms in use): a persistent AR(1) component,
# so that consumption depends on the preceding hours as in the plant data
z = np.zeros(n)
e = rng.normal(0, 1, n)
for i in range(1, n):
    z[i] = 0.97 * z[i - 1] + e[i]
night = ((h >= 19) | (h < 6)).astype(float)
# the plant load is more variable at night (persistence error is about 1.3 times the day-shift value)
raw = base + 120 * z + rng.normal(0, 150, n) * (1 + 0.6 * night)
raw = pd.Series(raw).ewm(alpha=0.6).mean().to_numpy()
# rescale to a typical plant load level (mean about 2,090 kWh, SD about 850 kWh)
en = 2090 + 850 * (raw - raw.mean()) / raw.std()
en[shut] = rng.uniform(20, 400, shut.sum())
en = np.clip(en, 18, 5200)
df = pd.DataFrame({"timestamp": t.strftime("%d/%m/%Y %H:%M"), "EN": en.round(2), "CL": (0.75 * en).round(4),
                   "RH": rh.round(1), "Air temp": temp.round(2), "Process": process.round(2),
                   "Paste": paste.round(0), "Cold": cold.round(0), "season": season})
# remove random blocks of hours (2-48 h) so the file has gaps like a real meter log
keep = np.ones(n, bool)
g = np.random.default_rng(7)
def drop_gaps(lo, hi, k):
    """remove k hours as random 2-48 h blocks inside rows [lo, hi)"""
    while (~keep[lo:hi]).sum() < k:
        s0 = int(g.integers(lo, hi))
        idx = np.arange(s0, min(s0 + int(g.integers(2, 49)), hi))
        idx = idx[keep[idx]][: k - (~keep[lo:hi]).sum()]
        keep[idx] = False
if a.test_start is None:
    drop_gaps(1, n - 1, max(0, n - a.hours))
else:
    cut = int(np.searchsorted(t, pd.Timestamp(a.test_start)))
    n_test = a.hours - int(a.hours * 0.8)
    drop_gaps(1, cut, max(0, cut - int(a.hours * 0.8)))
    drop_gaps(cut + 1, n - 1, max(0, (n - cut) - n_test))
df = df[keep].reset_index(drop=True)
df.to_csv("data/synthetic_hourly.csv", index=False)
print("written", len(df), "rows,", df.timestamp.iloc[0], "to", df.timestamp.iloc[-1])

# ---- synthetic 2025 daily meter registers and daily production (for daily_meter_check.py)
days = pd.date_range("2025-06-01", "2025-10-31", freq="D")
th = __import__("holidays").Thailand(years=2025)
out = np.clip(rng.normal(150000, 60000, len(days)), 9000, 420000)
kwh = 45000 + 0.15 * out + rng.normal(0, 6000, len(days))
work = np.array([(d.dayofweek < 5) and (d.date() not in th) for d in days])
rate_a = np.where(work, kwh * rng.uniform(0.40, 0.52, len(days)), np.nan)
rate_b = np.where(work, kwh - np.nan_to_num(rate_a), np.nan)
rate_c = np.where(work, np.nan, kwh)
pd.DataFrame({"date": days.date, "kWh_rateA": rate_a.round(2), "kWh_rateB": rate_b.round(2),
              "kWh_rateC": rate_c.round(2), "kWh_total": kwh.round(2)}).to_csv("data/synthetic_daily_meter.csv", index=False)
share = rng.dirichlet([3, 3, 4], len(days))
pd.DataFrame({"date": days.date, "paste_output_kg": (out * share[:, 0]).round(1), "packing_output_kg": (out * share[:, 1]).round(1),
              "freezing_output_kg": (out * share[:, 2]).round(1), "total_output_kg": out.round(1)}).to_csv("data/synthetic_daily_production.csv", index=False)
print("written synthetic 2025 daily files")
