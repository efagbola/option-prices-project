import numpy as np
import pandas as pd

from config import DATA_PATH, OUTPUT_PATH
from utils.price_curves import load_data, get_available_dates


def normalize_strikes(k: np.ndarray) -> np.ndarray:
    """
    Normalize strike units to decimals.
    Heuristic: if median strike > 1, interpret as bps-like (e.g., 300 -> 0.03).
    """
    k = np.asarray(k, dtype=float)
    k = k[np.isfinite(k)]
    if k.size == 0:
        return k
    if np.nanmedian(k) > 1.0:
        k = k / 10000.0
    return k


def get_swap_row(swaps, date, area):
    s = swaps[(swaps["date"] == date) & (swaps["area"] == area)]
    if s.empty:
        return None
    return s.iloc[0]


def undiscount_prices(df, B):
    """
    If price_per_1 is PV and B is the discount factor, then
    undiscounted expected payoff = PV / B.
    """
    out = df.copy()
    out["undisc"] = out["price_per_1"].astype(float) / float(B)
    return out


def trapz_integral(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return 0.0
    return float(np.trapezoid(y, x))


def raw_moment_from_options(k0, puts, calls, moment_order):
    """
    Approximate raw moments using static replication with g(x)=x^n and k0 ~ E[X].

    For n in {2,3,4}:
      g''(K) = 2
      g''(K) = 6K
      g''(K) = 12K^2
    """
    n = moment_order
    k_put = puts["k"].to_numpy(dtype=float)
    k_call = calls["k"].to_numpy(dtype=float)

    if n == 2:
        g2_put = 2.0 * np.ones_like(k_put)
        g2_call = 2.0 * np.ones_like(k_call)
    elif n == 3:
        g2_put = 6.0 * k_put
        g2_call = 6.0 * k_call
    elif n == 4:
        g2_put = 12.0 * (k_put ** 2)
        g2_call = 12.0 * (k_call ** 2)
    else:
        raise ValueError("moment_order must be 2, 3, or 4")

    term_put = trapz_integral(k_put, g2_put * puts["undisc"].to_numpy(dtype=float))
    term_call = trapz_integral(k_call, g2_call * calls["undisc"].to_numpy(dtype=float))

    return float((k0 ** n) + term_put + term_call)


def central_moments_from_raw(m1, m2, m3, m4):
    var = m2 - m1**2
    if not np.isfinite(var) or var <= 1e-14:
        return None

    mu3 = m3 - 3*m2*m1 + 2*(m1**3)
    mu4 = m4 - 4*m3*m1 + 6*m2*(m1**2) - 3*(m1**4)

    skew = mu3 / (var ** 1.5)
    kurt = mu4 / (var ** 2)  # non-excess kurtosis

    if not (np.isfinite(skew) and np.isfinite(kurt)):
        return None

    return float(var), float(skew), float(kurt)


def clean_option_df(df):
    """
    Keep finite rows, normalize strikes, deduplicate strikes by averaging price.
    """
    df = df.copy()
    df["k"] = normalize_strikes(df["k"].to_numpy())
    df["price_per_1"] = pd.to_numeric(df["price_per_1"], errors="coerce")
    df = df[np.isfinite(df["k"]) & np.isfinite(df["price_per_1"])]
    if df.empty:
        return df

    df = df.groupby("k", as_index=False)["price_per_1"].mean()
    df = df.sort_values("k")
    return df


def run_method(min_points_each_side=3):
    caps, floors, swaps = load_data(DATA_PATH)
    dates = get_available_dates(caps, floors)
    areas = sorted(pd.Series(caps["area"]).dropna().unique())

    results = []

    for date in dates:
        for area in areas:
            swap_row = get_swap_row(swaps, date, area)
            if swap_row is None:
                continue

            # Swap-implied mean proxy + discount factor
            k0 = float(swap_row["ypi_n"])
            B = float(swap_row["B"])

            if not (np.isfinite(k0) and np.isfinite(B) and B > 0):
                continue

            caps_d = caps[(caps["date"] == date) & (caps["area"] == area)][["k", "price_per_1"]]
            floors_d = floors[(floors["date"] == date) & (floors["area"] == area)][["k", "price_per_1"]]
            if caps_d.empty or floors_d.empty:
                continue

            caps_d = clean_option_df(caps_d)
            floors_d = clean_option_df(floors_d)
            if caps_d.empty or floors_d.empty:
                continue

            # Undiscounted expected payoffs
            caps_u = undiscount_prices(caps_d, B)
            floors_u = undiscount_prices(floors_d, B)

            # Split around k0 (proxy-forward)
            puts = floors_u[floors_u["k"] <= k0].sort_values("k")
            calls = caps_u[caps_u["k"] >= k0].sort_values("k")

            if len(puts) < min_points_each_side or len(calls) < min_points_each_side:
                continue

            # Mean proxy
            m1 = k0

            # Raw moments
            m2 = raw_moment_from_options(k0, puts, calls, 2)
            m3 = raw_moment_from_options(k0, puts, calls, 3)
            m4 = raw_moment_from_options(k0, puts, calls, 4)

            cm = central_moments_from_raw(m1, m2, m3, m4)
            if cm is None:
                continue
            var, skew, kurt = cm

            results.append({
                "date": date,
                "area": area,
                "k0_swap_mean": m1,
                "raw_m2": m2,
                "raw_m3": m3,
                "raw_m4": m4,
                "mean": m1,
                "variance": var,
                "skewness": skew,
                "kurtosis": kurt,
                "Kmin_used_put": float(puts["k"].min()),
                "Kmax_used_call": float(calls["k"].max()),
                "n_puts": int(len(puts)),
                "n_calls": int(len(calls)),
            })

    df = pd.DataFrame(results).sort_values(["area", "date"])
    out = OUTPUT_PATH + "moments_direct_bkm.csv"
    df.to_csv(out, index=False)
    print("Saved:", out)
    print("Rows:", len(df))


if __name__ == "__main__":
    run_method()
