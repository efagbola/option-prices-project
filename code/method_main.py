import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import UnivariateSpline

from config import DATA_PATH, OUTPUT_PATH
from utils.price_curves import load_data, get_available_dates, build_price_curve

# -----------------------------
# Quality gates (tuneable)
# -----------------------------
PARITY_EPS = 5e-4   # PV-per-1 units (after your /100 scaling)
MONO_TOL   = 1e-10  # monotonicity tolerance
CONV_TOL   = 1e-10  # convexity tolerance


def is_monotone_decreasing(y, tol=MONO_TOL) -> bool:
    y = np.asarray(y, dtype=float)
    dy = np.diff(y)
    return bool(np.all(dy <= tol))


def is_convex(y, x, tol=CONV_TOL) -> bool:
    """
    Discrete convexity check:
      call curve must be convex in strike => slopes are non-decreasing.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return False
    dx = np.diff(x)
    if np.any(dx <= 0):
        return False
    slopes = np.diff(y) / dx
    return bool(np.all(np.diff(slopes) >= -tol))


# -----------------------------
# Core BL helpers
# -----------------------------
def smooth_price_curve(strikes, prices, smooth_factor=None):
    """
    Smooth the option price curve across strikes with a scale-aware smoothing factor.
    """
    strikes = np.asarray(strikes, dtype=float)
    prices = np.asarray(prices, dtype=float)

    if smooth_factor is None:
        smooth_factor = 1e-3 * len(strikes) * float(np.nanvar(prices))

    return UnivariateSpline(strikes, prices, s=float(smooth_factor))


def implied_pdf_from_prices(spline, grid):
    """
    Breeden–Litzenberger: pdf is proportional to second derivative of option price wrt strike.
    We clip negative values (spline oscillation) and renormalize on the observed grid.
    """
    grid = np.asarray(grid, dtype=float)

    second_derivative = spline.derivative(n=2)(grid)
    pdf = np.asarray(second_derivative, dtype=float)
    pdf[~np.isfinite(pdf)] = 0.0
    pdf = np.maximum(pdf, 0.0)

    area = np.trapz(pdf, grid)
    if area <= 0 or not np.isfinite(area):
        return None

    return pdf / area


def compute_moments(grid, pdf):
    """
    Compute mean, variance, skewness, kurtosis from discrete pdf.
    Kurtosis is non-excess (Normal=3).
    """
    grid = np.asarray(grid, dtype=float)
    pdf = np.asarray(pdf, dtype=float)

    mean = float(np.trapz(grid * pdf, grid))
    var = float(np.trapz(((grid - mean) ** 2) * pdf, grid))
    if not np.isfinite(var) or var <= 1e-14:
        return None

    std = np.sqrt(var)
    skew = float(np.trapz((((grid - mean) / std) ** 3) * pdf, grid))
    kurt = float(np.trapz((((grid - mean) / std) ** 4) * pdf, grid))

    return mean, var, skew, kurt


# -----------------------------
# Two-sided curve construction
# -----------------------------
def build_two_sided_price_curve(caps, floors, swaps, date, area, min_points_each_side=3):
    """
    Construct a two-sided option curve around the ATM forward strike (K_star):

      - Calls side: caps with K >= K_star
      - Puts side:  floors with K <= K_star

    Then we convert puts into "call-equivalent" prices using put-call parity
    under the simple X=I_T/I_0 payoff convention:

        Call(K) - Put(K) = (E[X] - K) * B  (undiscounted forward times discount)
    => Call_equiv_from_put(K) = Put(K) + B*(K_star - K)

    where K_star is the swap-implied forward mean of X, and B is the discount factor.

    Returns:
        (K_all, C_all) merged, deduped, sorted
        or None if insufficient data.
    """
    swap_row = swaps[(swaps["date"] == str(date)) & (swaps["area"] == area)]
    if swap_row.empty:
        return None
    swap_row = swap_row.iloc[0]

    B = float(swap_row["B"])
    K_star = float(swap_row["K_star"])

    if not (np.isfinite(B) and B > 0 and np.isfinite(K_star) and K_star > 0):
        return None

    cap_curve = build_price_curve(caps, floors, swaps, date, area, instrument="cap", min_points=1)
    floor_curve = build_price_curve(caps, floors, swaps, date, area, instrument="floor", min_points=1)
    if cap_curve is None or floor_curve is None:
        return None

    Kc, Pc = cap_curve
    Kp, Pp = floor_curve

    # Parity consistency check on overlapping strikes (caps and floors share K grid)
    cap_df = pd.DataFrame({"K": Kc, "C": Pc})
    put_df = pd.DataFrame({"K": Kp, "P": Pp})
    over = cap_df.merge(put_df, on="K", how="inner")

    if len(over) >= 2:
        resid = (over["C"] - over["P"]) - (B * (K_star - over["K"]))
        med_abs = float(np.nanmedian(np.abs(resid)))
        if not np.isfinite(med_abs) or med_abs > PARITY_EPS:
            return None

    # OTM selection around K_star
    calls_mask = (Kc >= K_star) & np.isfinite(Pc) & (Pc >= -1e-12)
    puts_mask = (Kp <= K_star) & np.isfinite(Pp) & (Pp >= -1e-12)

    K_calls = Kc[calls_mask]
    C_calls = Pc[calls_mask]  # cap PV acts like call PV

    K_puts = Kp[puts_mask]
    P_puts = Pp[puts_mask]

    if len(K_calls) < min_points_each_side or len(K_puts) < min_points_each_side:
        return None

    # Convert puts into call-equivalent prices via parity:
    # C(K) = P(K) + B*(K_star - K)
    C_from_puts = P_puts + B * (K_star - K_puts)

    # Combine
    K_all = np.concatenate([K_puts, K_calls])
    C_all = np.concatenate([C_from_puts, C_calls])

    # Clean + dedup + sort
    mask = np.isfinite(K_all) & np.isfinite(C_all)
    K_all = K_all[mask]
    C_all = C_all[mask]

    if K_all.size < 6:
        return None

    tmp = pd.DataFrame({"K": K_all, "C": C_all}).groupby("K", as_index=False)["C"].mean()
    tmp = tmp.sort_values("K")

    if len(tmp) < 6:
        return None

    K_sorted = tmp["K"].to_numpy(dtype=float)
    C_sorted = tmp["C"].to_numpy(dtype=float)

    # No-arbitrage shape checks for a call curve
    if not is_monotone_decreasing(C_sorted):
        return None
    if not is_convex(C_sorted, K_sorted):
        return None

    return K_sorted, C_sorted, B, K_star


# -----------------------------
# Runner
# -----------------------------
def run_method(grid_n=400, min_points_each_side=3):
    """
    Two-sided BL implied density:
      - Builds a two-sided "call-equivalent" surface using caps+floors and swap parity
      - Applies BL second-derivative to that curve
      - Computes implied moments

    This is less "cap-only" and more comparable to BKM/lognormal, because it uses BOTH sides.
    """
    caps, floors, swaps = load_data(DATA_PATH)
    dates = get_available_dates(caps, floors)
    areas = sorted(pd.Series(caps["area"]).dropna().unique())

    results = []

    for date in dates:
        for area in areas:
            curve = build_two_sided_price_curve(
                caps, floors, swaps, date, area,
                min_points_each_side=min_points_each_side
            )
            if curve is None:
                continue

            strikes, prices, B, K_star = curve

            kmin, kmax = float(np.min(strikes)), float(np.max(strikes))
            if not np.isfinite(kmin) or not np.isfinite(kmax) or (kmax - kmin) <= 1e-10:
                continue

            grid = np.linspace(kmin, kmax, int(grid_n))

            spline = smooth_price_curve(strikes, prices, smooth_factor=None)
            pdf = implied_pdf_from_prices(spline, grid)
            if pdf is None:
                continue

            moments = compute_moments(grid, pdf)
            if moments is None:
                continue
            mean, var, skew, kurt = moments

            results.append({
                "date": str(date),
                "area": area,
                "instrument": "two_sided",
                "mean": mean,
                "variance": var,
                "skewness": skew,
                "kurtosis": kurt,
                "B": float(B),
                "K_star": float(K_star),
                "Kmin": float(kmin),
                "Kmax": float(kmax),
                "n_strikes": int(len(strikes)),
            })

    df = pd.DataFrame(results)
    out_path = Path(OUTPUT_PATH) / "moments_main.csv"

    if df.empty:
        print("No results produced — insufficient two-sided strike coverage.")
        df.to_csv(out_path, index=False)
        print("Saved empty file:", str(out_path))
        return

    df = df.sort_values(["area", "date"])
    df.to_csv(out_path, index=False)

    print("Saved:", str(out_path))
    print("Rows:", len(df))


if __name__ == "__main__":
    run_method(grid_n=400, min_points_each_side=3)
