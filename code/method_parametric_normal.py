import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
from scipy.stats import norm

from config import DATA_PATH, OUTPUT_PATH
from utils.price_curves import load_data, get_available_dates, build_price_curve


def choose_grid(strikes, n=400):
    strikes = np.asarray(strikes, dtype=float)
    kmin, kmax = float(np.min(strikes)), float(np.max(strikes))
    if not np.isfinite(kmin) or not np.isfinite(kmax) or (kmax - kmin) <= 1e-10:
        return None
    return np.linspace(kmin, kmax, int(n))


def smooth_price_curve(strikes, prices, smooth_factor=None):
    """
    Smooth prices with a scale-aware smoothing factor.
    """
    strikes = np.asarray(strikes, dtype=float)
    prices = np.asarray(prices, dtype=float)

    if smooth_factor is None:
        # heuristic regularization proportional to dispersion
        smooth_factor = 1e-3 * len(strikes) * float(np.nanvar(prices))

    return UnivariateSpline(strikes, prices, s=float(smooth_factor))


def implied_pdf_from_prices(spline, grid):
    """
    BL-style density proxy: pdf ∝ d²Price/dK² on the observed strike interval.
    Negative mass is clipped (spline oscillations) and the pdf is renormalized.
    """
    grid = np.asarray(grid, dtype=float)

    second_derivative = spline.derivative(n=2)(grid)
    pdf = np.asarray(second_derivative, dtype=float)
    pdf[~np.isfinite(pdf)] = 0.0

    pdf = np.maximum(pdf, 0.0)
    area = np.trapezoid(pdf, grid)
    if area <= 0 or not np.isfinite(area):
        return None

    return pdf / area


def fit_normal_to_pdf(grid, pdf):
    """
    Fit Normal(mu, sigma) to a target pdf over a grid by least squares.
    Uses moment initialization + bounded L-BFGS-B.
    """
    grid = np.asarray(grid, dtype=float)
    pdf = np.asarray(pdf, dtype=float)

    mu0 = float(np.trapezoid(grid * pdf, grid))
    var0 = float(np.trapezoid(((grid - mu0) ** 2) * pdf, grid))
    sigma0 = float(np.sqrt(max(var0, 1e-10)))

    gmin, gmax = float(grid.min()), float(grid.max())
    grange = max(gmax - gmin, 1e-6)

    def objective(theta):
        mu, log_sigma = theta
        sigma = float(np.exp(log_sigma))

        model_pdf = norm.pdf(grid, loc=mu, scale=sigma)
        model_area = np.trapezoid(model_pdf, grid)
        if model_area <= 0 or not np.isfinite(model_area):
            return 1e12

        model_pdf /= model_area
        err = pdf - model_pdf
        return float(np.mean(err * err))

    res = minimize(
        objective,
        x0=np.array([mu0, np.log(sigma0)]),
        method="L-BFGS-B",
        bounds=[
            (gmin, gmax),                      # mu within strike range
            (np.log(1e-6), np.log(2.0*grange))  # sigma in a generous range
        ],
        options={"maxiter": 500}
    )

    if not res.success:
        # still can accept if parameters are finite; but safer to skip
        pass

    mu_hat, log_sigma_hat = res.x
    sigma_hat = float(np.exp(log_sigma_hat))

    if not (np.isfinite(mu_hat) and np.isfinite(sigma_hat) and sigma_hat > 0):
        return None

    return float(mu_hat), float(sigma_hat)


def run_method(instrument="cap", min_points=6, grid_n=400):
    """
    Parametric Normal: recover a BL-style density proxy from option prices,
    then fit a Normal distribution to that density.

    instrument:
        "cap"  (recommended) or "floor" for robustness checks
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
                min_points=min_points
            )
            if curve is None:
                continue

            strikes, prices = curve

            # extra guards (build_price_curve already does most cleaning)
            if len(strikes) < min_points:
                continue
            if not (np.isfinite(strikes).all() and np.isfinite(prices).all()):
                continue

            grid = choose_grid(strikes, n=grid_n)
            if grid is None:
                continue

            spline = smooth_price_curve(strikes, prices, smooth_factor=None)
            pdf = implied_pdf_from_prices(spline, grid)
            if pdf is None or (not np.isfinite(pdf).all()):
                continue

            fit = fit_normal_to_pdf(grid, pdf)
            if fit is None:
                continue

            mu_hat, sigma_hat = fit

            mean = mu_hat
            variance = sigma_hat ** 2

            # Optional sanity caps to prevent plot poisoning
            if not (-0.10 < mean < 0.20):
                continue
            if not (0.0 <= variance < 1.0):
                continue

            results.append({
                "date": date,
                "area": area,
                "instrument": instrument,
                "mu_hat": mu_hat,
                "sigma_hat": sigma_hat,
                "mean": mean,
                "variance": variance,
                "skewness": 0.0,
                "kurtosis": 3.0
            })

    df = pd.DataFrame(results).sort_values(["area", "date"])
    out_path = Path(OUTPUT_PATH) / "moments_parametric_normal.csv"
    df.to_csv(out_path, index=False)

    print("Saved:", str(out_path))
    print("Rows:", len(df))


if __name__ == "__main__":
    run_method(instrument="cap")
