import numpy as np
import pandas as pd

from config import DATA_PATH, OUTPUT_PATH
from utils.price_curves import load_data, get_available_dates, build_price_curve

def fit_lognormal(strikes):
    strikes = strikes[strikes > 0]  # lognormal requires positivity
    y = np.log(strikes)

    mu_hat = y.mean()
    sigma_hat = y.std()

    return mu_hat, sigma_hat

def lognormal_moments(mu, sigma):
    mean = np.exp(mu + 0.5 * sigma**2)
    var = (np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2)
    skew = (np.exp(sigma**2) + 2) * np.sqrt(np.exp(sigma**2) - 1)
    kurt = np.exp(4*sigma**2) + 2*np.exp(3*sigma**2) + 3*np.exp(2*sigma**2) - 6

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

            strikes = strikes[strikes > 0]
            if len(strikes) < 6:
                continue

            mu_hat, sigma_hat = fit_lognormal(strikes)
            mean, var, skew, kurt = lognormal_moments(mu_hat, sigma_hat)

            results.append({
                "date": date,
                "area": area,
                "mu_hat": mu_hat,
                "sigma_hat": sigma_hat,
                "mean": mean,
                "variance": var,
                "skewness": skew,
                "kurtosis": kurt
            })

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_PATH + "moments_parametric_lognormal.csv", index=False)
