import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import UnivariateSpline

from config import DATA_PATH, OUTPUT_PATH
from utils.price_curves import load_data, get_available_dates, build_price_curve


def smooth_price_curve(strikes, prices, smooth_factor=None):
    """
    Smooth the option price curve across strikes.
    If smooth_factor is None, choose a scale-aware default based on price dispersion.
    """
    strikes = np.asarray(strikes, dtype=float)
    prices = np.asarray(prices, dtype=float)

    if smooth_factor is None:
        # scale-aware heuristic regularization
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

    # enforce nonnegativity
    pdf = np.maximum(pdf, 0.0)

    area = np.trapezoid(pdf, grid)
    if area <= 0 or not np.isfinite(area):
        return None

    pdf /= area
    return pdf


def compute_moments(grid, pdf):
    """
    Compute mean, variance, skewness, kurtosis from a discrete pdf on a grid.
    Kurtosis here is non-excess kurtosis (Normal = 3).
    """
    grid = np.asarray(grid, dtype=float)
    pdf = np.asarray(pdf, dtype=float)

    mean = float(np.trapezoid(grid * pdf, grid))
    var = float(np.trapezoid(((grid - mean) ** 2) * pdf, grid))

    # Guard against near-zero variance
    if not np.isfinite(var) or var <= 1e-14:
        return None

    std = np.sqrt(var)

    skew = float(np.trapezoid((((grid - mean) / std) ** 3) * pdf, grid))
    kurt = float(np.trapezoid((((grid - mean) / std) ** 4) * pdf, grid))

    return mean, var, skew, kurt


def run_method(instrument="cap", grid_n=400):
    """
    Run BL implied density and implied moments.

    instrument:
        "cap"  -> use cap quotes only (recommended default)
        "floor"-> use floor quotes only (robustness check)
    """
    caps, floors, swaps = load_data(DATA_PATH)
    dates = get_available_dates(caps, floors)
    areas = sorted(pd.Series(caps["area"]).dropna().unique())

    results = []

    for date in dates:
        for area in areas:
            curve = build_price_curve(caps, floors, swaps, date, area, instrument=instrument)
            if curve is None:
                continue

            strikes, prices = curve

            # basic guards
            if len(strikes) < 6:
                continue
            if not (np.isfinite(strikes).all() and np.isfinite(prices).all()):
                continue

            # grid over observed strike range (avoid extrapolation)
            kmin, kmax = float(np.min(strikes)), float(np.max(strikes))
            if not np.isfinite(kmin) or not np.isfinite(kmax) or (kmax - kmin) <= 1e-10:
                continue

            grid = np.linspace(kmin, kmax, int(grid_n))

            spline = smooth_price_curve(strikes, prices, smooth_factor=None)
            pdf = implied_pdf_from_prices(spline, grid)
            if pdf is None:
                continue
            if not np.isfinite(pdf).all():
                continue

            moments = compute_moments(grid, pdf)
            if moments is None:
                continue
            mean, var, skew, kurt = moments

            results.append({
                "date": date,
                "area": area,
                "instrument": instrument,
                "mean": mean,
                "variance": var,
                "skewness": skew,
                "kurtosis": kurt
            })

    df = pd.DataFrame(results).sort_values(["area", "date"])
    out_path = Path(OUTPUT_PATH) / "moments_main.csv"
    df.to_csv(out_path, index=False)
    print("Saved:", str(out_path))
    print("Rows:", len(df))


if __name__ == "__main__":
    run_method(instrument="cap")
