import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
from scipy.stats import norm

from config import DATA_PATH, OUTPUT_PATH
from utils.price_curves import load_data, get_available_dates, build_price_curve


def smooth_price_curve(strikes, prices, smooth_factor=1e-2):
    spline = UnivariateSpline(strikes, prices, s=smooth_factor)
    return spline


def implied_pdf_from_prices(spline, grid):
    second_derivative = spline.derivative(n=2)(grid)
    pdf = np.maximum(second_derivative, 0)
    area = np.trapezoid(pdf, grid)
    if area <= 0 or not np.isfinite(area):
        return None
    pdf = pdf / area
    return pdf


def fit_normal_to_pdf(grid, pdf):
    """
    Fit Normal(mu, sigma) to a target pdf on grid via least squares.
    Returns mu_hat, sigma_hat.
    """
    # initial guess from target moments
    mu0 = np.trapezoid(grid * pdf, grid)
    var0 = np.trapezoid((grid - mu0) ** 2 * pdf, grid)
    sigma0 = float(np.sqrt(max(var0, 1e-8)))

    def objective(theta):
        mu, log_sigma = theta
        sigma = np.exp(log_sigma)  # enforce sigma>0
        model_pdf = norm.pdf(grid, loc=mu, scale=sigma)
        # normalize numerically to be safe on truncated grid
        model_area = np.trapezoid(model_pdf, grid)
        if model_area <= 0 or not np.isfinite(model_area):
            return 1e9
        model_pdf = model_pdf / model_area
        return np.mean((pdf - model_pdf) ** 2)

    res = minimize(
        objective,
        x0=np.array([mu0, np.log(sigma0)]),
        method="Nelder-Mead",
        options={"maxiter": 2000}
    )

    mu_hat, log_sigma_hat = res.x
    sigma_hat = float(np.exp(log_sigma_hat))
    return float(mu_hat), sigma_hat


def run_method():
    caps, floors, swaps = load_data(DATA_PATH)
    dates = get_available_dates(caps, floors)
    areas = sorted(caps["area"].unique())

    results = []

    for date in dates:
        for area in areas:
            curve = build_price_curve(caps, floors, swaps, date, area)
            if curve is None:
                continue

            strikes, prices = curve

            # remove duplicate strikes (average)
            df_curve = pd.DataFrame({"k": strikes, "p": prices})
            df_curve = df_curve.groupby("k", as_index=False).mean()
            strikes = df_curve["k"].values
            prices = df_curve["p"].values

            if len(strikes) < 6:
                continue

            grid = np.linspace(strikes.min(), strikes.max(), 200)

            spline = smooth_price_curve(strikes, prices)
            pdf = implied_pdf_from_prices(spline, grid)
            if pdf is None or (not np.isfinite(pdf).all()):
                continue

            mu_hat, sigma_hat = fit_normal_to_pdf(grid, pdf)

            # Normal moments
            mean = mu_hat
            variance = sigma_hat ** 2
            skewness = 0.0
            kurtosis = 3.0  # non-excess kurtosis

            results.append({
                "date": date,
                "area": area,
                "mu_hat": mu_hat,
                "sigma_hat": sigma_hat,
                "mean": mean,
                "variance": variance,
                "skewness": skewness,
                "kurtosis": kurtosis
            })

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_PATH + "moments_parametric_normal.csv", index=False)
    print("Saved:", OUTPUT_PATH + "moments_parametric_normal.csv")
    print("Rows:", len(df))


if __name__ == "__main__":
    run_method()
