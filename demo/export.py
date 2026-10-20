"""Write demo/data.json for the static demo page, from the analysis code and results/. Run: python demo/export.py"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fetch import DAYTIME  # noqa: E402
from forecast import GRAND_FINALS, LAGS, run  # noqa: E402
from metro_tunnel import AFTER, BEFORE, ENTRANCES, FAR_M, NEAR_M, RESULTS, SITE_M, load_sensors  # noqa: E402

OUT = Path(__file__).parent / "data.json"
TOP_N = 12  # busiest sensors in the test window get a full hourly series


def records(df):
    """Rows as plain dicts, NaN as None (null in JSON)."""
    return df.astype(object).where(df.notna(), None).to_dict("records")


def ints(values):
    return [None if pd.isna(v) else int(round(v)) for v in values]


def main():
    changes = pd.read_csv(RESULTS / "metro_tunnel_by_sensor.csv", index_col="location_id")
    sensors = (
        load_sensors().join(changes[["change_pct", "am_change_pct"]]).reset_index()
        .rename(columns={"location_id": "id", "sensor_description": "name", "latitude": "lat", "longitude": "lon",
                         "change_pct": "daily_change_pct"})
        .drop(columns="site").round({"lat": 6, "lon": 6}).astype({"dist_m": int})
    )

    _, names, train, test, test_start = run()
    hours = pd.date_range(test_start, test.time.max(), freq="h")
    per_sensor = pd.read_csv(RESULTS / "forecast_by_sensor.csv", index_col="sensor")
    series = []
    for sid in test.groupby("sensor", observed=True)["count"].sum().nlargest(TOP_N).index:
        s = test[test.sensor == sid].set_index("time").reindex(hours)
        series.append({
            "id": int(sid), "name": names[sid],
            "wape_model": per_sensor.loc[sid, "wape_model"], "wape_naive_1w": per_sensor.loc[sid, "wape_naive_1w"],
            "actual": ints(s["count"]), "model": ints(s.pred_model), "naive": ints(s.pred_naive_1w),
        })
    assert all(len(s[k]) == len(hours) for s in series for k in ("actual", "model", "naive")), "series lengths differ"

    data = {
        "generated_at": str(pd.Timestamp.now(tz="Australia/Melbourne").date()),
        "data_through": str(test.time.max().date()),
        "entrances": [{"name": n, "lat": la, "lon": lo} for n, (la, lo) in ENTRANCES.items()],
        "sensors": records(sensors),
        "metro_tunnel": {
            "before": BEFORE, "after": AFTER, "near_m": NEAR_M, "far_m": FAR_M, "site_m": SITE_M,
            "summary": records(pd.read_csv(RESULTS / "metro_tunnel_summary.csv")),
        },
        "forecast": {
            "train_start": str(train.time.min().date()), "train_end": str(train.time.max().date()),
            "test_start": str(test_start), "step_hours": 1, "test_sensors": int(test.sensor.nunique()),
            "test_hours": len(test), "lag_weeks": [k // 168 for k in LAGS], "daytime_hours": [DAYTIME.start, DAYTIME.stop],
            "holidays": sorted({str(d.date()) for d in test.loc[test.holiday == 1, "time"].dt.normalize()}),
            "grand_finals": GRAND_FINALS,
            "metrics": records(pd.read_csv(RESULTS / "forecast_metrics.csv")),
            "event_days": records(pd.read_csv(RESULTS / "forecast_event_days.csv")),
            "series": series,
        },
    }
    OUT.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    size = OUT.stat().st_size
    assert size < 1.5e6, f"data.json is {size / 1e6:.2f} MB, over the 1.5 MB budget"
    print(f"{OUT.name}: {size / 1e3:.0f} kB, {len(sensors)} sensors, {len(series)} series x {len(hours)} hours")


if __name__ == "__main__":
    main()
