import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

from config import DATA_PATH, OUTPUT_PATH
from utils.price_curves import load_data, get_available_dates, build_price_curve


def smooth_price_curve(strikes, prices, smooth_factor=1e-2):
    """
    Smooth price curve using spline
    """
    spline = UnivariateSpline(strikes, prices, s=smooth_factor)
    return spline


def implied_pdf_from_prices(spline, grid):
    """
    Breeden–Litzenberger: second derivative
    """
    second_derivative = spline.derivative(n=2)(grid)
    pdf = np.maximum(second_derivative, 0)
    pdf = pdf / np.trapz(pdf, grid)
    return pdf


def compute_moments(grid, pdf):
    """
    Compute mean, variance, skewness, kurtosis
    """
    mean = np.trapz(grid * pdf, grid)
    var = np.trapz((grid - mean) ** 2 * pdf, grid)
    std = np.sqrt(var)

    skew = np.trapz(((grid - mean) / std) ** 3 * pdf, grid)
    kurt = np.trapz(((grid - mean) / std) ** 4 * pdf, grid)

    return mean, var, skew, kurt


def run_method():
    caps, floors, swaps = load_data(DATA_PATH)
    dates = get_available_dates(caps, floors)

    results = []

    for date in dates:
        for area in caps["area"].unique():
            curve = build_price_curve(caps, floors, swaps, date, area)
            if curve is None:
                continue

            strikes, prices = curve
            
            # --- CLEAN STRIKES ---
            df_curve = pd.DataFrame({"k": strikes, "p": prices})
            df_curve = df_curve.groupby("k", as_index=False).mean()

            strikes = df_curve["k"].values
            prices = df_curve["p"].values

            # Require enough points
            if len(strikes) < 6:
                continue

            grid = np.linspace(strikes.min(), strikes.max(), 200)
            spline = smooth_price_curve(strikes, prices)
            pdf = implied_pdf_from_prices(spline, grid)
            
            # Skip degenerate PDFs
            if not np.isfinite(pdf).all() or np.trapezoid(pdf, grid) < 1e-6:
                continue

            mean, var, skew, kurt = compute_moments(grid, pdf)

            results.append({
                "date": date,
                "area": area,
                "mean": mean,
                "variance": var,
                "skewness": skew,
                "kurtosis": kurt
            })

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_PATH + "moments_main.csv", index=False)


if __name__ == "__main__":
    run_method()
