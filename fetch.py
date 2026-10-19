"""Download City of Melbourne pedestrian counts, sensor locations and Melbourne hourly weather into data/."""
from pathlib import Path

import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
COM = "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets"
LAT, LON = -37.8136, 144.9631  # Melbourne CBD
DAYTIME = range(8, 18)  # 8am-6pm: a gap here means the sensor was down, not that the street was empty


def hourly_grid(counts):
    """Hourly counts as a time x sensor table, cleaned.

    The feed leaves out hours with zero pedestrians (true zeros almost never appear, while 1s are
    common at 3am), so a missing hour on a working day is a 0. A day missing any 8am-6pm hour, or
    with a zero total, is an outage and the whole day becomes NaN.
    """
    c = counts.assign(time=pd.to_datetime(counts.sensing_date) + pd.to_timedelta(counts.hourday, unit="h"))
    wide = c.pivot_table(index="time", columns="location_id", values="pedestriancount", aggfunc="sum")
    start, end = wide.index.min().normalize(), wide.index.max().normalize() + pd.Timedelta(hours=23)
    wide = wide.reindex(pd.date_range(start, end, freq="h"))
    day = wide.index.normalize()
    daytime = wide[wide.index.hour.isin(DAYTIME)]
    working = daytime.notna().groupby(daytime.index.normalize()).all() & (wide.fillna(0).groupby(day).sum() > 0)
    return wide.fillna(0).where(working.reindex(day).to_numpy())


def download(url, path, **params):
    with requests.get(url, params=params, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def weather(start, end):
    hourly = "temperature_2m,precipitation"
    tz = "Australia/Melbourne"
    frames = []
    # The archive lags a few days behind, so recent hours come from the forecast API's past_days.
    for url, params in [
        ("https://archive-api.open-meteo.com/v1/archive", {"start_date": start, "end_date": end}),
        ("https://api.open-meteo.com/v1/forecast", {"past_days": 92, "forecast_days": 1}),
    ]:
        r = requests.get(url, params={"latitude": LAT, "longitude": LON, "hourly": hourly, "timezone": tz, **params}, timeout=120)
        r.raise_for_status()
        frames.append(pd.DataFrame(r.json()["hourly"]).dropna())
    w = pd.concat(frames).drop_duplicates("time", keep="first")  # archive wins where both exist
    w["time"] = pd.to_datetime(w["time"])
    return w[(w.time >= start) & (w.time < pd.Timestamp(end) + pd.Timedelta(days=1))].sort_values("time")


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    download(f"{COM}/pedestrian-counting-system-monthly-counts-per-hour/exports/csv", DATA / "counts.csv",
             select="location_id,sensing_date,hourday,pedestriancount", delimiter=",")
    download(f"{COM}/pedestrian-counting-system-sensor-locations/exports/csv", DATA / "sensors.csv", delimiter=",")
    dates = pd.read_csv(DATA / "counts.csv", usecols=["sensing_date"])["sensing_date"]
    weather(dates.min(), dates.max()).to_csv(DATA / "weather.csv", index=False)
    print({p.name: f"{p.stat().st_size / 1e6:.1f} MB" for p in DATA.iterdir()})
