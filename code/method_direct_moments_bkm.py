import numpy as np
import pandas as pd
from pathlib import Path

from config import DATA_PATH, OUTPUT_PATH
from utils.price_curves import load_data, get_available_dates


# -----------------------------
# Swap helper
# -----------------------------
def get_swap_row(swaps: pd.DataFrame, date, area):
    s = swaps[(swaps["date"] == str(date)) & (swaps["area"] == area)]
    if s.empty:
        return None
    return s.iloc[0]


# -----------------------------
# Data helpers
# -----------------------------
def clean_option_df(df: pd.DataFrame, strike_col="K"):
    """
    Clean one option surface (caps or floors) for a given date/area.
    Uses strike column 'K' (index ratio strike).
    Deduplicates strikes by averaging price_per_1.
    """
    out = df[[strike_col, "price_per_1"]].copy()
    out[strike_col] = pd.to_numeric(out[strike_col], errors="coerce")
    out["price_per_1"] = pd.to_numeric(out["price_per_1"], errors="coerce")
    out = out[np.isfinite(out[strike_col]) & np.isfinite(out["price_per_1"])]

    if out.empty:
        return out

    out = out.groupby(strike_col, as_index=False)["price_per_1"].mean()
    out = out.sort_values(strike_col)
    return out


def undiscount_prices(df: pd.DataFrame, B: float):
    """
    If price_per_1 is PV and B is the discount factor:
        undiscounted expected payoff = PV / B
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


# -----------------------------
# Tail stabilization (new)
# -----------------------------
def add_tail_point_puts(puts: pd.DataFrame, K0: float, strike_col="K", tail_frac=0.60):
    """
    Add a synthetic far-left tail point for puts (floors):
      - extend strike down by tail_frac of observed span below K0
      - set option price at that tail strike to 0 (OTM very far away)
    This reduces sensitivity to missing far OTM strikes in the integral.
    """
    if puts.empty:
        return puts

    Kmin = float(puts[strike_col].min())
    span = max(K0 - Kmin, 0.0)
    if span <= 0:
        return puts

    K_tail = max(1e-6, K0 - (1.0 + tail_frac) * span)

    # Append a zero-payoff tail point
    tail = pd.DataFrame({strike_col: [K_tail], "undisc": [0.0]})
    out = pd.concat([tail, puts[[strike_col, "undisc"]]], ignore_index=True)
    out = out.groupby(strike_col, as_index=False)["undisc"].mean().sort_values(strike_col)
    return out


def add_tail_point_calls(calls: pd.DataFrame, K0: float, strike_col="K", tail_frac=0.60):
    """
    Add a synthetic far-right tail point for calls (caps):
      - extend strike up by tail_frac of observed span above K0
      - set option price at that tail strike to 0 (OTM very far away)
    """
    if calls.empty:
        return calls

    Kmax = float(calls[strike_col].max())
    span = max(Kmax - K0, 0.0)
    if span <= 0:
        return calls

    K_tail = K0 + (1.0 + tail_frac) * span

    tail = pd.DataFrame({strike_col: [K_tail], "undisc": [0.0]})
    out = pd.concat([calls[[strike_col, "undisc"]], tail], ignore_index=True)
    out = out.groupby(strike_col, as_index=False)["undisc"].mean().sort_values(strike_col)
    return out


def winsorize_payoffs(df: pd.DataFrame, q=0.98):
    """
    Optional: winsorize extreme undiscounted payoffs to reduce noisy tail impact.
    (Useful when data has occasional bad quotes.)
    """
    if df.empty:
        return df
    cap = float(df["undisc"].quantile(q))
    out = df.copy()
    out["undisc"] = out["undisc"].clip(lower=0.0, upper=cap)
    return out


# -----------------------------
# BKM-style raw moments
# -----------------------------
def raw_moment_from_options(K0, puts, calls, moment_order, strike_col="K"):
    """
    Compute E[X^n] for n in {2,3,4} using a static replication identity.

    With K0 ≈ E[X], the linear term is dropped.

    g(x)=x^n -> g''(K):
      n=2: 2
      n=3: 6K
      n=4: 12K^2
    """
    n = int(moment_order)

    K_put = puts[strike_col].to_numpy(dtype=float)
    K_call = calls[strike_col].to_numpy(dtype=float)

    if n == 2:
        g2_put = 2.0 * np.ones_like(K_put)
        g2_call = 2.0 * np.ones_like(K_call)
    elif n == 3:
        g2_put = 6.0 * K_put
        g2_call = 6.0 * K_call
    elif n == 4:
        g2_put = 12.0 * (K_put ** 2)
        g2_call = 12.0 * (K_call ** 2)
    else:
        raise ValueError("moment_order must be 2, 3, or 4")

    term_put = trapz_integral(K_put, g2_put * puts["undisc"].to_numpy(dtype=float))
    term_call = trapz_integral(K_call, g2_call * calls["undisc"].to_numpy(dtype=float))

    return float((K0 ** n) + term_put + term_call)


def central_moments_from_raw(m1, m2, m3, m4):
    var = m2 - m1**2
    if (not np.isfinite(var)) or var <= 1e-14:
        return None

    mu3 = m3 - 3*m2*m1 + 2*(m1**3)
    mu4 = m4 - 4*m3*m1 + 6*m2*(m1**2) - 3*(m1**4)

    skew = mu3 / (var ** 1.5)
    kurt = mu4 / (var ** 2)  # non-excess

    if not (np.isfinite(skew) and np.isfinite(kurt)):
        return None

    return float(var), float(skew), float(kurt)


# -----------------------------
# Runner
# -----------------------------
def run_method(
    min_points_each_side=3,
    strike_col="K",
    tail_frac=0.60,
    winsor_q=0.98,
):
    """
    Tail-stabilized BKM-style direct moments.

    What we change vs baseline:
      1) Add synthetic far OTM tail points with zero payoff on each side
         to reduce sensitivity to missing far strikes.
      2) Winsorize undiscounted payoffs to reduce the effect of bad quotes.

    These are pragmatic stabilizers for sparse strike grids.
    """
    caps, floors, swaps = load_data(DATA_PATH)
    dates = get_available_dates(caps, floors)
    areas = sorted(pd.Series(caps["area"]).dropna().unique())

    results = []

    for date in dates:
        for area in areas:
            swap_row = get_swap_row(swaps, date, area)
            if swap_row is None:
                continue

            B = float(swap_row["B"])
            K0 = float(swap_row["K_star"])  # mean proxy (index ratio)

            if not (np.isfinite(B) and B > 0 and np.isfinite(K0) and K0 > 0):
                continue

            caps_d = caps[(caps["date"] == str(date)) & (caps["area"] == area)]
            floors_d = floors[(floors["date"] == str(date)) & (floors["area"] == area)]
            if caps_d.empty or floors_d.empty:
                continue

            caps_d = clean_option_df(caps_d, strike_col=strike_col)
            floors_d = clean_option_df(floors_d, strike_col=strike_col)
            if caps_d.empty or floors_d.empty:
                continue

            caps_u = undiscount_prices(caps_d, B)
            floors_u = undiscount_prices(floors_d, B)

            # OTM around K0
            puts = floors_u[floors_u[strike_col] <= K0][[strike_col, "undisc"]].sort_values(strike_col)
            calls = caps_u[caps_u[strike_col] >= K0][[strike_col, "undisc"]].sort_values(strike_col)

            if len(puts) < min_points_each_side or len(calls) < min_points_each_side:
                continue

            # Winsorize payoffs (reduces impact of bad quotes)
            puts = winsorize_payoffs(puts, q=winsor_q)
            calls = winsorize_payoffs(calls, q=winsor_q)

            # Add synthetic tail points (reduces tail sensitivity)
            puts = add_tail_point_puts(puts, K0, strike_col=strike_col, tail_frac=tail_frac)
            calls = add_tail_point_calls(calls, K0, strike_col=strike_col, tail_frac=tail_frac)

            # Mean proxy
            m1 = K0

            # Raw moments
            m2 = raw_moment_from_options(K0, puts, calls, 2, strike_col=strike_col)
            m3 = raw_moment_from_options(K0, puts, calls, 3, strike_col=strike_col)
            m4 = raw_moment_from_options(K0, puts, calls, 4, strike_col=strike_col)

            cm = central_moments_from_raw(m1, m2, m3, m4)
            if cm is None:
                continue
            var, skew, kurt = cm

            results.append({
                "date": str(date),
                "area": area,
                "K0_swap_mean": m1,
                "raw_m2": m2,
                "raw_m3": m3,
                "raw_m4": m4,
                "mean": m1,
                "variance": var,
                "skewness": skew,
                "kurtosis": kurt,
                "tail_frac": float(tail_frac),
                "winsor_q": float(winsor_q),
                "Kmin_used_put": float(puts[strike_col].min()),
                "Kmax_used_call": float(calls[strike_col].max()),
                "n_puts": int(len(puts)),
                "n_calls": int(len(calls)),
            })

    df = pd.DataFrame(results)
    out_path = Path(OUTPUT_PATH) / "moments_direct_bkm.csv"

    if df.empty:
        print("No results produced — insufficient OTM strikes or missing swaps.")
        df.to_csv(out_path, index=False)
        print("Saved empty file:", str(out_path))
        return

    df = df.sort_values(["area", "date"])
    df.to_csv(out_path, index=False)

    print("Saved:", str(out_path))
    print("Rows:", len(df))


if __name__ == "__main__":
    run_method(min_points_each_side=3, strike_col="K", tail_frac=0.60, winsor_q=0.98)
