import numpy as np
import pandas as pd

from config import DATA_PATH, OUTPUT_PATH
from utils.price_curves import load_data, get_available_dates

def get_swap_row(swaps, date, area):
    s = swaps[(swaps["date"] == date) & (swaps["area"] == area)]
    if s.empty:
        return None
    return s.iloc[0]

def undiscount_prices(df, B):
    # price_per_1 is PV; undiscounted expectation of payoff = price / B
    # (B = discount factor)
    out = df.copy()
    out["undisc"] = out["price_per_1"] / float(B)
    return out

def trapz_integral(x, y):
    if len(x) < 2:
        return 0.0
    return float(np.trapezoid(y, x))

def raw_moment_from_options(k0, puts, calls, moment_order):
    """
    Compute E[X^n] using static replication:
    E[g(X)] = g(k0) + g'(k0)E[X-k0] + ∫_{K<k0} g''(K) PutUndisc(K) dK + ∫_{K>k0} g''(K) CallUndisc(K) dK
    We set k0 ~ E[X] => E[X-k0] ~ 0, so the linear term drops.
    Here g(x)=x^n, so:
      n=1: g''=0 (handled outside)
      n=2: g''(K)=2
      n=3: g''(K)=6K
      n=4: g''(K)=12K^2
    """
    n = moment_order
    if n == 2:
        g2_put = 2.0 * np.ones_like(puts["k"].values, dtype=float)
        g2_call = 2.0 * np.ones_like(calls["k"].values, dtype=float)
    elif n == 3:
        g2_put = 6.0 * puts["k"].values.astype(float)
        g2_call = 6.0 * calls["k"].values.astype(float)
    elif n == 4:
        g2_put = 12.0 * (puts["k"].values.astype(float) ** 2)
        g2_call = 12.0 * (calls["k"].values.astype(float) ** 2)
    else:
        raise ValueError("moment_order must be 2,3,4 here")

    term_put = trapz_integral(puts["k"].values, g2_put * puts["undisc"].values)
    term_call = trapz_integral(calls["k"].values, g2_call * calls["undisc"].values)

    return float((k0 ** n) + term_put + term_call)

def central_moments_from_raw(m1, m2, m3, m4):
    var = m2 - m1**2
    if var <= 0 or not np.isfinite(var):
        return None

    mu3 = m3 - 3*m2*m1 + 2*(m1**3)
    mu4 = m4 - 4*m3*m1 + 6*m2*(m1**2) - 3*(m1**4)

    skew = mu3 / (var ** 1.5)
    kurt = mu4 / (var ** 2)

    return float(var), float(skew), float(kurt)

def run_method(min_points_each_side=3):
    caps, floors, swaps = load_data(DATA_PATH)
    dates = get_available_dates(caps, floors)
    areas = sorted(caps["area"].unique())

    results = []

    for date in dates:
        for area in areas:
            swap_row = get_swap_row(swaps, date, area)
            if swap_row is None:
                continue

            # k0 ~ E[X] under pricing measure (proxy by 1Y inflation swap rate)
            k0 = float(swap_row["ypi_n"])
            B = float(swap_row["B"])

            caps_d = caps[(caps["date"] == date) & (caps["area"] == area)][["k", "price_per_1"]].copy()
            floors_d = floors[(floors["date"] == date) & (floors["area"] == area)][["k", "price_per_1"]].copy()

            if caps_d.empty or floors_d.empty:
                continue

            # Undiscount prices -> expected payoffs
            caps_u = undiscount_prices(caps_d, B)
            floors_u = undiscount_prices(floors_d, B)

            # Build OTM sets around k0:
            # puts (floors) for K <= k0, calls (caps) for K >= k0
            puts = floors_u[floors_u["k"] <= k0].sort_values("k")
            calls = caps_u[caps_u["k"] >= k0].sort_values("k")

            # Need enough points on each side for stable integration
            if len(puts) < min_points_each_side or len(calls) < min_points_each_side:
                continue

            # Moment 1 (mean): proxy by swap rate (forward expectation)
            m1 = k0

            # Moments 2-4 from option prices (static replication)
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

    df = pd.DataFrame(results)
    out = OUTPUT_PATH + "moments_direct_bkm.csv"
    df.to_csv(out, index=False)
    print("Saved:", out)
    print("Rows:", len(df))

if __name__ == "__main__":
    run_method()
