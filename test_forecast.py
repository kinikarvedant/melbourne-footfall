"""Checks the places this analysis can quietly cheat or rot: data cleaning, lag alignment, distances. Run: python test_forecast.py"""
import numpy as np
import pandas as pd

from fetch import hourly_grid
from forecast import hourly_panel
from metro_tunnel import haversine_m

days = pd.date_range("2026-01-01", periods=35, freq="D")
counts = pd.DataFrame(
    [(1, d.strftime("%Y-%m-%d"), h, 0 if d == days[20] else i * 24 + h + 1) for i, d in enumerate(days) for h in range(24)],
    columns=["location_id", "sensing_date", "hourday", "pedestriancount"],
)
drop = lambda date, hour: counts[(counts.sensing_date == date) & (counts.hourday == hour)].index
counts = counts.drop(drop("2026-01-30", 3))   # a night hour the feed left out: nobody walked past
counts = counts.drop(drop("2026-01-29", 12))  # a midday gap: the sensor was down

grid = hourly_grid(counts)
assert grid.loc["2026-01-30 03:00", 1] == 0, "a missing night hour is a zero, not a gap"
assert grid.loc["2026-01-29", 1].isna().all(), "a missing 8am-6pm hour makes the whole day an outage"
assert grid.loc[days[20].strftime("%Y-%m-%d"), 1].isna().all(), "an all-zero day is an outage"

p = hourly_panel(grid).set_index("time")
t = pd.Timestamp("2026-01-31 10:00")
assert p.loc[t, "lag_168"] == p.loc[t - pd.Timedelta(hours=168), "count"], "lag_168 must be the same hour a week earlier"
assert abs(haversine_m(-37.8136, 144.9631, -37.8136, 144.9731) - 879) < 5, "0.01 deg of longitude in Melbourne is about 879 m"
print("ok")
