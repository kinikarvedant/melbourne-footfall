"""Did foot traffic change near the new Metro Tunnel stations? Year-on-year change, near vs far sites.

The tunnel opened 30 Nov 2025 and all its lines moved in on 1 Feb 2026, so we compare
1 Feb - 18 Oct 2026 with the same dates in 2025. Same calendar window, so seasonality cancels.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

from fetch import hourly_grid

ROOT = Path(__file__).parent
DATA, RESULTS = ROOT / "data", ROOT / "results"

# Metro Tunnel entrances, from OpenStreetMap (railway=subway_entrance).
ENTRANCES = {
    "State Library - Franklin St": (-37.8074727, 144.9631607),
    "State Library - Swanston St": (-37.8094103, 144.9635607),
    "Town Hall - Collins St": (-37.8156149, 144.9669515),
    "Town Hall - Swanston St": (-37.8170107, 144.9670546),
}
NEAR_M, FAR_M = 150, 500  # sites in between are left out as a buffer
SITE_M = 25  # sensors closer than this count as one site (e.g. the three counters at Degraves St / Flinders Ln)
BEFORE = ("2025-02-01", "2025-10-18")
AFTER = ("2026-02-01", "2026-10-18")
MIN_COVERAGE = 0.8  # share of days a sensor must be working, in both windows
MEASURES = {
    "all day, every day": {},
    "weekday 7-10am": {"hours": [7, 8, 9], "weekdays_only": True},
    "weekday 4-7pm": {"hours": [16, 17, 18], "weekdays_only": True},
}


def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6_371_000 * 2 * np.arcsin(np.sqrt(a))


def daily_totals(grid, hours=range(24), weekdays_only=False):
    """Date x sensor totals over the given hours. Outage days stay NaN (hourly_grid blanks whole days)."""
    g = grid[grid.index.hour.isin(hours)]
    if weekdays_only:
        g = g[g.index.dayofweek < 5]
    return g.groupby(g.index.normalize()).sum(min_count=1)


def window_mean(daily, start, end):
    w = daily.loc[start:end]
    return w.mean().where(w.notna().mean() >= MIN_COVERAGE)


def sensor_changes(grid, sensors, hours=range(24), weekdays_only=False):
    d = daily_totals(grid, hours, weekdays_only)
    both = pd.concat({"before": window_mean(d, *BEFORE), "after": window_mean(d, *AFTER)}, axis=1).dropna()
    both["change_pct"] = 100 * (both.after / both.before - 1)
    return both.join(sensors, how="inner")


def by_site(df):
    """One row per site: mean change of its sensors, grouped by its closest sensor's distance."""
    return df.sort_values("dist_m").groupby("site").agg(change_pct=("change_pct", "mean"), group=("group", "first"))


def diff_in_medians(sites, rng, n_perm=10_000):
    """Median change at near sites minus far sites, with a permutation p-value over site labels."""
    sub = sites[sites.group != "buffer"]
    near = (sub.group == "near").to_numpy()
    x = sub.change_pct.to_numpy()
    observed = np.median(x[near]) - np.median(x[~near])
    perms = np.array([np.median(x[p]) - np.median(x[~p]) for p in (rng.permutation(near) for _ in range(n_perm))])
    p = (np.sum(np.abs(perms) >= abs(observed)) + 1) / (n_perm + 1)
    return observed, p, int(near.sum()), int((~near).sum())


def load_sensors(near_m=NEAR_M):
    sensors = pd.read_csv(DATA / "sensors.csv", encoding="utf-8-sig")
    sensors = sensors[sensors.location_type == "Outdoor"].set_index("location_id")
    dist = np.min([haversine_m(sensors.latitude, sensors.longitude, la, lo) for la, lo in ENTRANCES.values()], axis=0)
    sensors["dist_m"] = dist.round()
    sensors["group"] = np.where(dist <= near_m, "near", np.where(dist >= FAR_M, "far", "buffer"))
    xy = np.c_[sensors.longitude * 111_320 * np.cos(np.radians(-37.81)), sensors.latitude * 110_574]  # metres
    sensors["site"] = fcluster(linkage(xy, "single"), t=SITE_M, criterion="distance")
    return sensors[["sensor_description", "latitude", "longitude", "dist_m", "group", "site"]]


def compare(grid, sensors, rng, n_perm=10_000):
    rows = []
    for name, kw in MEASURES.items():
        sites = by_site(sensor_changes(grid, sensors, **kw))
        diff, p, n_near, n_far = diff_in_medians(sites, rng, n_perm)
        rows.append({
            "measure": name,
            "near_median_change_pct": round(sites[sites.group == "near"].change_pct.median(), 1),
            "far_median_change_pct": round(sites[sites.group == "far"].change_pct.median(), 1),
            "difference_pts": round(diff, 1),
            "permutation_p": round(p, 3),
            "n_near_sites": n_near,
            "n_far_sites": n_far,
        })
    return pd.DataFrame(rows)


def plot_am_peak(per_sensor, far_median, path):
    import matplotlib.pyplot as plt

    d = per_sensor[per_sensor.dist_m <= 300].dropna(subset=["am_change_pct"]).sort_values("am_change_pct")
    fig, ax = plt.subplots(figsize=(8.5, 0.32 * len(d) + 1.8))
    colours = np.where(d.am_change_pct >= 0, "#2a7ab9", "#c8553d")
    ax.barh(d.sensor_description + "  (" + d.dist_m.astype(int).astype(str) + " m)", d.am_change_pct, color=colours)
    ax.axvline(far_median, color="#555", ls="--", lw=1)
    ax.text(far_median, -0.9, f" sites 500 m+ away: {far_median:+.1f}%", fontsize=8, color="#555", va="top")
    ax.axvline(0, color="#222", lw=0.8)
    ax.set_xlabel("% change, weekday 7-10am, Feb-Oct 2026 vs Feb-Oct 2025")
    ax.set_title("Morning foot traffic near the new Metro Tunnel stations\n"
                 "(City of Melbourne sensors within 300 m of an entrance)", loc="left", fontsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    RESULTS.mkdir(exist_ok=True)
    grid = hourly_grid(pd.read_csv(DATA / "counts.csv", encoding="utf-8-sig"))
    sensors = load_sensors()

    summary = compare(grid, sensors, np.random.default_rng(42))  # primary spec: 150 m, fixed before looking at results
    summary.to_csv(RESULTS / "metro_tunnel_summary.csv", index=False)

    # How much the answer depends on the radius. Report this, don't pick the best-looking one.
    sens = [compare(grid, load_sensors(r), np.random.default_rng(42), 2_000).assign(near_radius_m=r) for r in (100, 150, 200, 250, 300)]
    sensitivity = pd.concat(sens)[["near_radius_m", "measure", "near_median_change_pct", "far_median_change_pct", "difference_pts", "permutation_p", "n_near_sites"]]
    sensitivity.to_csv(RESULTS / "metro_tunnel_sensitivity.csv", index=False)

    daily = sensor_changes(grid, sensors).rename(columns={"before": "daily_before", "after": "daily_after"})
    am = sensor_changes(grid, sensors, **MEASURES["weekday 7-10am"])
    per_sensor = daily[["daily_before", "daily_after", "change_pct"]].join(
        am[["before", "after", "change_pct"]].rename(columns=lambda c: "am_" + c), how="outer"
    ).join(sensors, how="inner").sort_values("dist_m")
    per_sensor.round({c: 1 for c in per_sensor.columns if c not in ("latitude", "longitude")}).to_csv(RESULTS / "metro_tunnel_by_sensor.csv")

    far_am = summary.set_index("measure").loc["weekday 7-10am", "far_median_change_pct"]
    plot_am_peak(per_sensor, far_am, RESULTS / "metro_tunnel_am_peak.png")

    print(summary.to_string(index=False), end="\n\n")
    print(sensitivity.to_string(index=False))
    return summary, sensitivity, per_sensor


if __name__ == "__main__":
    main()
