"""Forecast hourly foot traffic per CBD sensor up to a week ahead, and score it against naive baselines.

Every feature is known a week in advance (calendar, holidays, counts from 7+ days ago), except weather:
the "+ weather" model uses observed weather as a stand-in for a forecast, so treat it as an upper bound.
"""
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from fetch import hourly_grid

ROOT = Path(__file__).parent
DATA, RESULTS = ROOT / "data", ROOT / "results"
TEST_WEEKS = 8
LAGS = [168, 336, 504, 672]  # same hour 1-4 weeks back; the shortest lag that's known a week ahead
BASE = ["sensor", "hour", "dow", "month", "holiday", "lag_168", "lag_336", "mean_4w"]
WEATHER = ["temperature_2m", "precipitation"]
GRAND_FINALS = {"2025-09-27": "AFL Grand Final", "2026-09-26": "AFL Grand Final"}  # chart labels only, not a feature


def hourly_panel(wide):
    """One row per sensor per hour from a cleaned hourly_grid, plus lag features."""
    lags = {f"lag_{k}": wide.shift(k) for k in LAGS}
    panel = pd.concat({"count": wide, **lags}, axis=1).stack(future_stack=True).rename_axis(["time", "sensor"]).reset_index()
    panel["mean_4w"] = panel[[f"lag_{k}" for k in LAGS]].mean(axis=1)
    return panel


def add_features(panel, weather):
    vic = holidays.Australia(subdiv="VIC", years=range(panel.time.dt.year.min(), panel.time.dt.year.max() + 1))
    p = panel.assign(
        hour=panel.time.dt.hour,
        dow=panel.time.dt.dayofweek,
        month=panel.time.dt.month,
        holiday=panel.time.dt.date.map(lambda d: d in vic).astype(int),
    )
    return p.merge(weather, on="time", how="left")


def holiday_effects(grid):
    """Each VIC public holiday vs a normal day: summed over sensors working that day, against each sensor's
    median for the same weekday (non-holidays) within four weeks either side."""
    daily = grid.groupby(grid.index.normalize()).sum(min_count=1)
    vic = holidays.Australia(subdiv="VIC", years=range(daily.index.year.min(), daily.index.year.max() + 1))
    is_hol = np.array([d.date() in vic for d in daily.index])
    rows = []
    for day in daily.index[is_hol]:
        normal = (abs(daily.index - day) <= pd.Timedelta(days=28)) & (daily.index.dayofweek == day.dayofweek) & ~is_hol
        ref, today = daily[normal].median(), daily.loc[day]
        ok = today.notna() & ref.notna()
        rows.append({"date": day.date(), "holiday": vic[day.date()], "sensors": int(ok.sum()),
                     "change_vs_normal_pct": round(100 * (today[ok].sum() / ref[ok].sum() - 1), 1)})
    return pd.DataFrame(rows)


def event_days(test, sensor_names, days, sensor=1):
    """Actual vs forecast totals on the given days, across all sensors and at one named sensor."""
    rows = []
    for day in days:
        d = test[test.time.dt.normalize() == pd.Timestamp(day)]
        for where, sub in [("all sensors", d), (sensor_names[sensor], d[d.sensor == sensor])]:
            if len(sub):
                rows.append({"date": day, "where": where, "sensors": sub.sensor.nunique(), "actual": int(sub["count"].sum()),
                             "forecast": int(sub.pred_model.sum()), "gap_pct": round(100 * (sub["count"].sum() / sub.pred_model.sum() - 1), 1)})
    return pd.DataFrame(rows)


def wape(y, yhat):
    return 100 * np.abs(y - yhat).sum() / y.sum()


def fit(train, features):
    model = HistGradientBoostingRegressor(
        loss="poisson", max_iter=400, learning_rate=0.08, categorical_features=["sensor"], random_state=0
    )
    return model.fit(train[features], train["count"])


def main():
    RESULTS.mkdir(exist_ok=True)
    counts = pd.read_csv(DATA / "counts.csv", encoding="utf-8-sig")
    weather = pd.read_csv(DATA / "weather.csv", parse_dates=["time"])
    outdoor = pd.read_csv(DATA / "sensors.csv", encoding="utf-8-sig").query("location_type == 'Outdoor'")
    sensor_names = outdoor.set_index("location_id").sensor_description

    grid = hourly_grid(counts[counts.location_id.isin(outdoor.location_id)])
    hol = holiday_effects(grid)
    hol.to_csv(RESULTS / "holiday_effects.csv", index=False)
    df = add_features(hourly_panel(grid), weather)
    df = df.dropna(subset=["count", "lag_168"])
    df["sensor"] = df.sensor.astype("category")

    test_start = df.time.max().normalize() - pd.Timedelta(weeks=TEST_WEEKS) + pd.Timedelta(days=1)
    train, test = df[df.time < test_start], df[df.time >= test_start].copy()

    test["pred_naive_1w"] = test.lag_168
    test["pred_mean_4w"] = test.mean_4w.fillna(test.lag_168)
    test["pred_model"] = fit(train, BASE).predict(test[BASE])
    test["pred_model_weather"] = fit(train, BASE + WEATHER).predict(test[BASE + WEATHER])

    preds = {
        "same hour last week": "pred_naive_1w",
        "average of last 4 weeks": "pred_mean_4w",
        "gradient boosting": "pred_model",
        "gradient boosting + weather*": "pred_model_weather",
    }
    holiday = test[test.holiday == 1]
    metrics = pd.DataFrame([
        {
            "method": name,
            "wape_pct": round(wape(test["count"], test[col]), 1),
            "mae_per_hour": round(np.abs(test["count"] - test[col]).mean(), 1),
            "holiday_wape_pct": round(wape(holiday["count"], holiday[col]), 1) if len(holiday) else None,
        }
        for name, col in preds.items()
    ])
    metrics.to_csv(RESULTS / "forecast_metrics.csv", index=False)

    by_sensor = (
        test.groupby("sensor", observed=True)
        .apply(lambda g: pd.Series({"wape_naive_1w": wape(g["count"], g.pred_naive_1w), "wape_model": wape(g["count"], g.pred_model), "test_hours": len(g)}))
        .join(sensor_names.rename("sensor_description"))
        .round(1)
        .sort_values("wape_model")
    )
    by_sensor.to_csv(RESULTS / "forecast_by_sensor.csv")

    days = sorted({str(d.date()) for d in holiday.time.dt.normalize().unique()} | set(GRAND_FINALS))
    events = event_days(test, sensor_names, days)
    events.to_csv(RESULTS / "forecast_event_days.csv", index=False)
    plot_week(test, sensor_names, RESULTS / "forecast_week.png")
    span = f"{train.time.min():%d %b %Y} to {train.time.max():%d %b %Y}, test {test_start:%d %b %Y} to {test.time.max():%d %b %Y}"
    print(f"train {span}; {test.sensor.nunique()} sensors, {len(test):,} test hours, {len(holiday):,} on public holidays")
    print(metrics.to_string(index=False), end="\n\n")
    print(events.to_string(index=False), end="\n\n")
    print(f"{(hol.change_vs_normal_pct < 0).sum()} of {len(hol)} public holidays were quieter than a normal day")
    print(hol.to_string(index=False))
    return metrics


def plot_week(test, sensor_names, path, sensor=1):
    """Actual vs forecast for one sensor over the test week that holds a public holiday, if there is one."""
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter

    s = test[test.sensor == sensor].set_index("time").sort_index()
    hol = s[s.holiday == 1]
    start = (hol.index.min() if len(hol) else s.index.min()).normalize()
    start -= pd.Timedelta(days=start.dayofweek)  # back to Monday
    w = s[start : start + pd.Timedelta(days=7) - pd.Timedelta(hours=1)]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(w.index, w["count"], color="#222", lw=1.6, label="actual")
    ax.plot(w.index, w.pred_naive_1w, color="#c8553d", lw=1, ls="--", label="same hour last week")
    ax.plot(w.index, w.pred_model, color="#2a7ab9", lw=1.4, label="gradient boosting")
    top = ax.get_ylim()[1]
    for d in sorted({t.normalize() for t in w[w.holiday == 1].index}):
        ax.axvspan(d, d + pd.Timedelta(days=1), color="#f2c14e", alpha=0.25, lw=0)
        ax.text(d + pd.Timedelta(hours=1), top * 0.93, "public holiday", fontsize=8, color="#8a6d00")
    for day, label in GRAND_FINALS.items():
        if start <= pd.Timestamp(day) < start + pd.Timedelta(days=7):
            ax.text(pd.Timestamp(day) + pd.Timedelta(hours=1), top * 0.93, label, fontsize=8, color="#555")
    ax.xaxis.set_major_formatter(DateFormatter("%a %d %b"))
    ax.set_title(f"{sensor_names.get(sensor, sensor)}, hourly pedestrians, week of {start:%d %b %Y}", loc="left", fontsize=11)
    ax.set_ylabel("pedestrians per hour")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
