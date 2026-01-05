import numpy as np
import pandas as pd
from pathlib import Path

from config import DATA_PATH, OUTPUT_PATH
from utils.price_curves import load_data, get_available_dates, build_price_curve


def clean_strikes_for_lognormal(strikes: np.ndarray, min_points: int = 6):
    """
    Lognormal requires strictly positive strikes and non-degenerate log dispersion.
    Assumes strikes are already unit-normalized by build_price_curve().
    """
    strikes = np.asarray(strikes, dtype=float)
    strikes = strikes[np.isfinite(strikes)]
    strikes = strikes[strikes > 0]

    if strikes.size < min_points:
        return None

    # Prevent degenerate fit (all strikes essentially identical in log-space)
    if np.nanstd(np.log(strikes)) < 1e-10:
        return None

    return strikes


def fit_lognormal_from_strikes(strikes: np.ndarray):
    """
    Fit Lognormal parameters using log-strikes.
    NOTE: This is a strike-based proxy, not a price-calibrated fit.
    """
    y = np.log(strikes)
    mu_hat = float(np.mean(y))
    sigma_hat = float(np.std(y, ddof=1))  # sample std

    if not np.isfinite(mu_hat) or not np.isfinite(sigma_hat):
        return None

    sigma_hat = max(sigma_hat, 1e-8)
    return mu_hat, sigma_hat


def lognormal_moments(mu: float, sigma: float):
    """
    Lognormal mean/variance and standardized skewness / excess kurtosis.
    """
    s2 = sigma * sigma
    exp_s2 = np.exp(s2)

    mean = np.exp(mu + 0.5 * s2)
    var = (exp_s2 - 1.0) * np.exp(2.0 * mu + s2)

    skew = (exp_s2 + 2.0) * np.sqrt(exp_s2 - 1.0)
    kurt_excess = np.exp(4*s2) + 2*np.exp(3*s2) + 3*np.exp(2*s2) - 6.0

    return float(mean), float(var), float(skew), float(kurt_excess)


def run_method(instrument: str = "cap", min_points: int = 6):
    caps, floors, swaps = load_data(DATA_PATH)
    dates = get_available_dates(caps, floors)
    areas = sorted(pd.Series(caps["area"]).dropna().unique())

    results = []

    for date in dates:
        for area in areas:
            curve = build_price_curve(caps, floors, swaps, date, area, instrument=instrument, min_points=min_points)
            if curve is None:
                continue

            strikes, prices = curve  # prices unused in this proxy method
            strikes = clean_strikes_for_lognormal(strikes, min_points=min_points)
            if strikes is None:
                continue

            fit = fit_lognormal_from_strikes(strikes)
            if fit is None:
                continue
            mu_hat, sigma_hat = fit

            mean, var, skew, kurt_excess = lognormal_moments(mu_hat, sigma_hat)

            # Loose sanity caps (prevents one broken curve wrecking plots)
            if not (0.0 < mean < 1.0):
                continue
            if not (0.0 <= var < 1.0):
                continue

            results.append({
                "date": date,
                "area": area,
                "instrument": instrument,
                "mu_hat": mu_hat,
                "sigma_hat": sigma_hat,
                "mean": mean,
                "variance": var,
                "skewness": skew,
                "kurtosis_excess": kurt_excess
            })

    df = pd.DataFrame(results).sort_values(["area", "date"])
    out_path = Path(OUTPUT_PATH) / "moments_parametric_lognormal.csv"
    df.to_csv(out_path, index=False)
    print("Saved:", str(out_path))
    print("Rows:", len(df))


if __name__ == "__main__":
    run_method(instrument="cap")
