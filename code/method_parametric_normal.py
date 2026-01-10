import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
from scipy.stats import norm

from config import DATA_PATH, OUTPUT_PATH
from utils.price_curves import load_data, get_available_dates, build_price_curve

# -----------------------------
# Quality gates (tuneable)
# -----------------------------
MONO_TOL = 1e-10
CONV_TOL = 1e-10

# If smooth_factor is None, we use a small data-scaled smoothing to reduce oscillations
SMOOTH_MULT = 1e-4


def is_monotone_decreasing(y, tol=MONO_TOL) -> bool:
    y = np.asarray(y, dtype=float)
    return bool(np.all(np.diff(y) <= tol))


def is_monotone_increasing(y, tol=MONO_TOL) -> bool:
    y = np.asarray(y, dtype=float)
    return bool(np.all(np.diff(y) >= -tol))


def is_convex(y, x, tol=CONV_TOL) -> bool:
    """
    Discrete convexity check:
      convex in strike => slopes are non-decreasing.
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


def choose_grid(strikes, n=400):
    strikes = np.asarray(strikes, dtype=float)
    kmin, kmax = float(np.min(strikes)), float(np.max(strikes))
    if not np.isfinite(kmin) or not np.isfinite(kmax) or (kmax - kmin) <= 1e-10:
        return None
    return np.linspace(kmin, kmax, int(n))


def smooth_price_curve(strikes, prices, smooth_factor=None):
    """
    Spline fit of prices across strikes.

    If smooth_factor is None, we use a small data-scaled smoothing level to reduce
    oscillations in the second derivative (important with sparse strikes).
    """
    strikes = np.asarray(strikes, dtype=float)
    prices = np.asarray(prices, dtype=float)

    if smooth_factor is None:
        v = float(np.nanvar(prices))
        s = float(SMOOTH_MULT * len(prices) * max(v, 1e-12))
    else:
        s = float(smooth_factor)

    return UnivariateSpline(strikes, prices, s=s)


def implied_pdf_from_prices(spline, grid):
    """
    BL-style density proxy: pdf(K) ∝ d²Price/dK² on the observed strike interval.
    Clip negative values (spline artifacts) and renormalize on the grid.
    """
    grid = np.asarray(grid, dtype=float)

    second_derivative = spline.derivative(n=2)(grid)
    pdf = np.asarray(second_derivative, dtype=float)
    pdf[~np.isfinite(pdf)] = 0.0

    pdf = np.maximum(pdf, 0.0)
    area = float(np.trapz(pdf, grid))
    if (not np.isfinite(area)) or area <= 0.0:
        return None

    return pdf / area


def fit_normal_to_pdf(grid, pdf):
    """
    Fit Normal(mu, sigma) to a target pdf on a grid by least squares.
    Uses moment initialization and bounded L-BFGS-B on (mu, log_sigma).
    """
    grid = np.asarray(grid, dtype=float)
    pdf = np.asarray(pdf, dtype=float)

    mu0 = float(np.trapz(grid * pdf, grid))
    var0 = float(np.trapz(((grid - mu0) ** 2) * pdf, grid))
    sigma0 = float(np.sqrt(max(var0, 1e-10)))

    gmin, gmax = float(grid.min()), float(grid.max())
    grange = max(gmax - gmin, 1e-6)

    def objective(theta):
        mu, log_sigma = theta
        sigma = float(np.exp(log_sigma))

        model_pdf = norm.pdf(grid, loc=mu, scale=sigma)

        # normalize on the truncated grid
        area = float(np.trapz(model_pdf, grid))
        if (not np.isfinite(area)) or area <= 0.0:
            return 1e12
        model_pdf = model_pdf / area

        err = pdf - model_pdf
        return float(np.mean(err * err))

    res = minimize(
        objective,
        x0=np.array([mu0, np.log(sigma0)]),
        method="L-BFGS-B",
        bounds=[
            (gmin, gmax),                      # mu within observed strike range
            (np.log(1e-6), np.log(2.0 * grange))  # sigma bounded away from 0
        ],
        options={"maxiter": 500},
    )

    if not res.success:
        return None

    mu_hat, log_sigma_hat = res.x
    sigma_hat = float(np.exp(log_sigma_hat))

    if not (np.isfinite(mu_hat) and np.isfinite(sigma_hat) and sigma_hat > 0.0):
        return None

    return float(mu_hat), float(sigma_hat)


def run_method(
    instrument="cap",
    min_points=4,
    grid_n=400,
    smooth_factor=None,
    edge_mass_threshold=0.7,
):
    """
    Parametric Normal (diagnostic benchmark):
      1) recover a BL-style density proxy from option prices (single instrument curve)
      2) fit a Normal distribution to that density on the observed strike interval

    Notes:
      - Strikes here are K = index-ratio strikes (1 + inflation).
      - With sparse strikes, the BL proxy can be boundary-dominated; we optionally filter those cases.
    """
    caps, floors, swaps = load_data(DATA_PATH)
    dates = get_available_dates(caps, floors)
    areas = sorted(pd.Series(caps["area"]).dropna().unique())

    results = []

    for date in dates:
        for area in areas:
            curve = build_price_curve(
                caps, floors, swaps, date, area,
                instrument=instrument,
                min_points=min_points,
            )
            if curve is None:
                continue

            strikes, prices = curve

            strikes = np.asarray(strikes, dtype=float)
            prices = np.asarray(prices, dtype=float)

            if len(strikes) < min_points:
                continue
            if not (np.isfinite(strikes).all() and np.isfinite(prices).all()):
                continue
            if np.nanstd(prices) < 1e-12:
                continue
            if np.nanmin(prices) < -1e-12:
                continue

            # Ensure increasing strikes for all downstream logic
            idx = np.argsort(strikes)
            strikes = strikes[idx]
            prices = prices[idx]

            # No-arbitrage shape checks (single-instrument curve):
            # Caps/calls: decreasing + convex in strike
            # Floors/puts: increasing + convex in strike
            if instrument == "cap":
                if not is_monotone_decreasing(prices):
                    continue
                if not is_convex(prices, strikes):
                    continue
            elif instrument == "floor":
                if not is_monotone_increasing(prices):
                    continue
                if not is_convex(prices, strikes):
                    continue

            grid = choose_grid(strikes, n=grid_n)
            if grid is None:
                continue

            spline = smooth_price_curve(strikes, prices, smooth_factor=smooth_factor)
            pdf = implied_pdf_from_prices(spline, grid)
            if pdf is None or (not np.isfinite(pdf).all()):
                continue

            # filter out densities dominated by boundary mass (common with sparse strikes)
            edge_mass = float(np.trapz(pdf[:10], grid[:10]) + np.trapz(pdf[-10:], grid[-10:]))
            if (not np.isfinite(edge_mass)) or edge_mass > float(edge_mass_threshold):
                continue

            fit = fit_normal_to_pdf(grid, pdf)
            if fit is None:
                continue

            mu_hat, sigma_hat = fit
            mean = mu_hat
            variance = sigma_hat ** 2

            # VERY loose K-space sanity caps (prevents plot poisoning without filtering everything)
            if not (0.5 < mean < 2.0):
                continue
            if not (0.0 <= variance < 1.0):
                continue

            results.append({
                "date": str(date),
                "area": area,
                "instrument": instrument,
                "mu_hat": mu_hat,
                "sigma_hat": sigma_hat,
                "mean": mean,
                "variance": variance,
                "skewness": 0.0,
                "kurtosis": 3.0,  # non-excess
            })

    df = pd.DataFrame(results)
    out_path = Path(OUTPUT_PATH) / "moments_parametric_normal.csv"

    if df.empty:
        print("No results produced — Normal fit returned no valid rows.")
        df.to_csv(out_path, index=False)
        print("Saved empty file:", str(out_path))
        return

    df = df.sort_values(["area", "date"])
    df.to_csv(out_path, index=False)

    print("Saved:", str(out_path))
    print("Rows:", len(df))


if __name__ == "__main__":
    run_method(instrument="cap")
