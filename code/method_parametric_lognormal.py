import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import norm

from config import DATA_PATH, OUTPUT_PATH
from utils.price_curves import load_data, get_available_dates, build_price_curve


# -----------------------------
# Swap helper
# -----------------------------
def get_swap_row(swaps: pd.DataFrame, date, area):
    s = swaps[(swaps["date"] == str(date)) & (swaps["area"] == area)]
    if s.empty:
        return None
    return s.iloc[0]


# -----------------------------
# Lognormal pricing (X = I_T / I_0)
# X ~ Lognormal(mu, sigma^2)
# PV = B * E[(X-K)^+] for caps (call-like)
# PV = B * E[(K-X)^+] for floors (put-like)
# -----------------------------
def lognormal_call_undisc(K, mu, sigma):
    K = np.asarray(K, dtype=float)
    d1 = (mu - np.log(K) + sigma**2) / sigma
    d2 = d1 - sigma
    EX = np.exp(mu + 0.5 * sigma**2)
    return EX * norm.cdf(d1) - K * norm.cdf(d2)


def lognormal_put_undisc(K, mu, sigma):
    K = np.asarray(K, dtype=float)
    d1 = (mu - np.log(K) + sigma**2) / sigma
    d2 = d1 - sigma
    EX = np.exp(mu + 0.5 * sigma**2)
    return K * norm.cdf(-d2) - EX * norm.cdf(-d1)


def lognormal_moments(mu: float, sigma: float):
    """
    Return mean, variance, skewness, kurtosis (NON-excess; Normal=3 convention).
    """
    s2 = sigma * sigma
    exp_s2 = np.exp(s2)

    mean = np.exp(mu + 0.5 * s2)
    var = (exp_s2 - 1.0) * np.exp(2.0 * mu + s2)
    skew = (exp_s2 + 2.0) * np.sqrt(exp_s2 - 1.0)

    kurt_excess = np.exp(4*s2) + 2*np.exp(3*s2) + 3*np.exp(2*s2) - 6.0
    kurt = kurt_excess + 3.0

    return float(mean), float(var), float(skew), float(kurt)


# -----------------------------
# Calibration (stabilized)
# -----------------------------
def calibrate_lognormal(
    K_calls, P_calls, K_puts, P_puts,
    B: float,
    K_star: float,
    sigma_max: float = 0.30,
    lambda_anchor: float = 2000.0,
):
    """
    Calibrate (mu, sigma) to cap/floor prices with stabilization:

      - Tight sigma bounds: sigma in [1e-6, sigma_max]
      - Anchor penalty to keep E[X] close to K_star:
            penalty = lambda_anchor * (E[X] - K_star)^2

    Returns (mu_hat, sigma_hat, rmse) or None.
    """
    K_calls = np.asarray(K_calls, dtype=float)
    P_calls = np.asarray(P_calls, dtype=float)
    K_puts = np.asarray(K_puts, dtype=float)
    P_puts = np.asarray(P_puts, dtype=float)

    if not (np.isfinite(B) and B > 0 and np.isfinite(K_star) and K_star > 0):
        return None
    if (K_calls.size and np.any(K_calls <= 0)) or (K_puts.size and np.any(K_puts <= 0)):
        return None
    if (K_calls.size + K_puts.size) < 4:
        return None

    # Initial guess anchored to K_star
    sigma0 = 0.05
    mu0 = np.log(float(K_star)) - 0.5 * sigma0**2

    # If we have enough strikes, use log-dispersion to initialize sigma
    allK = np.concatenate([K_calls, K_puts]) if (K_calls.size + K_puts.size) else np.array([K_star], dtype=float)
    allK = allK[np.isfinite(allK) & (allK > 0)]
    if allK.size >= 6:
        s_guess = float(np.nanstd(np.log(allK)))
        if np.isfinite(s_guess) and s_guess > 1e-6:
            sigma0 = float(np.clip(s_guess, 0.01, min(0.20, sigma_max)))
            mu0 = np.log(float(K_star)) - 0.5 * sigma0**2

    def loss(theta):
        mu, log_sigma = theta
        sigma = float(np.exp(log_sigma))

        errs = []

        if K_calls.size:
            model_calls = B * lognormal_call_undisc(K_calls, mu, sigma)
            errs.append(model_calls - P_calls)

        if K_puts.size:
            model_puts = B * lognormal_put_undisc(K_puts, mu, sigma)
            errs.append(model_puts - P_puts)

        if not errs:
            return 1e12

        err = np.concatenate(errs)
        mse = float(np.mean(err * err))

        # anchor mean to swap-implied forward K_star
        mean_model = float(np.exp(mu + 0.5 * sigma**2))
        penalty = float(lambda_anchor) * (mean_model - float(K_star))**2

        return mse + penalty

    res = minimize(
        loss,
        x0=np.array([mu0, np.log(sigma0)]),
        method="L-BFGS-B",
        bounds=[(None, None), (np.log(1e-6), np.log(float(sigma_max)))],
        options={"maxiter": 700},
    )

    if not res.success:
        return None

    mu_hat, log_sigma_hat = res.x
    sigma_hat = float(np.exp(log_sigma_hat))

    if not (np.isfinite(mu_hat) and np.isfinite(sigma_hat) and 0 < sigma_hat <= sigma_max):
        return None

    # RMSE diagnostic (PV units) on used instruments
    errs = []
    if K_calls.size:
        errs.append(B * lognormal_call_undisc(K_calls, mu_hat, sigma_hat) - P_calls)
    if K_puts.size:
        errs.append(B * lognormal_put_undisc(K_puts, mu_hat, sigma_hat) - P_puts)

    err = np.concatenate(errs) if errs else np.array([np.nan])
    rmse = float(np.sqrt(np.nanmean(err * err))) if np.isfinite(err).any() else np.nan

    return float(mu_hat), float(sigma_hat), rmse


# -----------------------------
# Runner
# -----------------------------
def run_method(
    min_points_each_side: int = 3,
    sigma_max: float = 0.30,
    lambda_anchor: float = 2000.0,
):
    """
    True implied Lognormal (stabilized calibration):
      - Uses swaps: B and K_star
      - Uses caps (calls) and floors (puts) on X = I_T/I_0
      - Calibrates (mu, sigma) to observed PVs across strikes
      - Stabilizes with sigma bounds and mean anchoring to K_star
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
            K_star = float(swap_row["K_star"])
            if not (np.isfinite(B) and B > 0 and np.isfinite(K_star) and K_star > 0):
                continue

            cap_curve = build_price_curve(caps, floors, swaps, date, area, instrument="cap", min_points=1)
            floor_curve = build_price_curve(caps, floors, swaps, date, area, instrument="floor", min_points=1)
            if cap_curve is None or floor_curve is None:
                continue

            Kc, Pc = cap_curve
            Kp, Pp = floor_curve

            # OTM sets around K_star (classic)
            calls_mask = (Kc >= K_star) & np.isfinite(Kc) & (Kc > 0) & np.isfinite(Pc) & (Pc >= -1e-12)
            puts_mask = (Kp <= K_star) & np.isfinite(Kp) & (Kp > 0) & np.isfinite(Pp) & (Pp >= -1e-12)

            K_calls, P_calls = Kc[calls_mask], Pc[calls_mask]
            K_puts, P_puts = Kp[puts_mask], Pp[puts_mask]

            if len(K_calls) < min_points_each_side or len(K_puts) < min_points_each_side:
                continue

            fit = calibrate_lognormal(
                K_calls, P_calls, K_puts, P_puts,
                B=B, K_star=K_star,
                sigma_max=sigma_max,
                lambda_anchor=lambda_anchor,
            )
            if fit is None:
                continue

            mu_hat, sigma_hat, rmse = fit
            mean, var, skew, kurt = lognormal_moments(mu_hat, sigma_hat)

            results.append({
                "date": str(date),
                "area": area,
                "mu_hat": mu_hat,
                "sigma_hat": sigma_hat,
                "mean": mean,
                "variance": var,
                "skewness": skew,
                "kurtosis": kurt,
                "rmse_price": rmse,
                "B": B,
                "K_star": K_star,
                "n_calls": int(len(K_calls)),
                "n_puts": int(len(K_puts)),
                "Kmin_put": float(np.min(K_puts)),
                "Kmax_call": float(np.max(K_calls)),
                "sigma_max": float(sigma_max),
                "lambda_anchor": float(lambda_anchor),
            })

    df = pd.DataFrame(results)
    out_path = Path(OUTPUT_PATH) / "moments_parametric_lognormal.csv"

    if df.empty:
        print("No results produced — lognormal calibration returned no valid rows.")
        df.to_csv(out_path, index=False)
        print("Saved empty file:", str(out_path))
        return

    df = df.sort_values(["area", "date"])
    df.to_csv(out_path, index=False)

    print("Saved:", str(out_path))
    print("Rows:", len(df))


if __name__ == "__main__":
    run_method(min_points_each_side=3, sigma_max=0.30, lambda_anchor=2000.0)
