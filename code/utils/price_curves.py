import numpy as np
import pandas as pd


def load_data(data_path):
    """
    Load caps, floors, and swaps data.
    """
    caps = pd.read_csv(data_path + "cleaned_caps_quotes_1y.csv")
    floors = pd.read_csv(data_path + "cleaned_floors_quotes_1y.csv")
    swaps = pd.read_csv(data_path + "cleaned_swaps_curves_1y.csv")
    return caps, floors, swaps


def get_available_dates(caps, floors):
    """
    Return sorted list of dates common to caps and floors.
    """
    dates_caps = set(caps["date"].unique())
    dates_floors = set(floors["date"].unique())
    return sorted(dates_caps.intersection(dates_floors))


def normalize_strikes(k: np.ndarray) -> np.ndarray:
    """
    Normalize strike units to decimals.
    Heuristic: if median strike > 1, interpret as bps-like (e.g., 300 -> 0.03).
    """
    k = np.asarray(k, dtype=float)
    k = k[np.isfinite(k)]
    if k.size == 0:
        return k

    if np.nanmedian(k) > 1.0:
        k = k / 10000.0

    return k


def build_price_curve(caps, floors, swaps, date, area, instrument="cap", min_points=6):
    """
    Build a *single-type* option price curve for one date and one area.

    instrument: "cap" or "floor"

    Returns:
        (k, price_per_1) as numpy arrays, sorted by k, with duplicated strikes averaged.
        Returns None if insufficient data.
    """
    if instrument not in {"cap", "floor"}:
        raise ValueError("instrument must be 'cap' or 'floor'")

    df = caps if instrument == "cap" else floors
    df_d = df[(df["date"] == date) & (df["area"] == area)]
    if df_d.empty:
        return None

    k = normalize_strikes(df_d["k"].to_numpy())
    p = np.asarray(df_d["price_per_1"].to_numpy(), dtype=float)

    mask = np.isfinite(k) & np.isfinite(p)
    k = k[mask]
    p = p[mask]

    if k.size < min_points:
        return None

    # Deduplicate strikes by averaging prices
    tmp = pd.DataFrame({"k": k, "p": p}).groupby("k", as_index=False).mean()
    tmp = tmp.sort_values("k")

    if len(tmp) < min_points:
        return None

    return tmp["k"].to_numpy(), tmp["p"].to_numpy()
