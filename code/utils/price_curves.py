import numpy as np
import pandas as pd


def load_data(data_path: str):
    """
    Load caps, floors, and swaps data (1Y).
    Dates are cast to string for consistent matching across scripts.
    """
    caps = pd.read_csv(data_path + "cleaned_caps_quotes_1y.csv")
    floors = pd.read_csv(data_path + "cleaned_floors_quotes_1y.csv")
    swaps = pd.read_csv(data_path + "cleaned_swaps_curves_1y.csv")
    PRICE_SCALE = 100.0

    for df in (caps, floors, swaps):
        if "date" in df.columns:
            df["date"] = df["date"].astype(str)


    for df in (caps, floors):
        if "price_per_1" in df.columns:
            df["price_per_1"] = pd.to_numeric(df["price_per_1"], errors="coerce") / PRICE_SCALE

    return caps, floors, swaps


def get_available_dates(caps: pd.DataFrame, floors: pd.DataFrame):
    """
    Return sorted list of dates common to caps and floors.
    """
    dates_caps = set(caps["date"].unique())
    dates_floors = set(floors["date"].unique())
    return sorted(dates_caps.intersection(dates_floors))


def build_price_curve(caps, floors, swaps, date, area, instrument="cap", min_points=6):
    """
    Build a single-instrument option price curve for one date and one area.

    Uses strike column 'K' (index ratio strike = 1 + inflation), not 'k' (inflation rate).

    Args:
        caps, floors, swaps: loaded dataframes (swaps kept for consistent interface)
        date: date key (string or convertible to string)
        area: area code (e.g., "EU")
        instrument: "cap" or "floor"
        min_points: minimum number of strikes required

    Returns:
        (K, price_per_1) as numpy arrays, sorted by K, with duplicate strikes averaged.
        Returns None if insufficient data.
    """
    if instrument not in {"cap", "floor"}:
        raise ValueError("instrument must be 'cap' or 'floor'")

    date = str(date)
    df = caps if instrument == "cap" else floors
    df_d = df[(df["date"] == date) & (df["area"] == area)]
    if df_d.empty:
        return None

    # Use index-ratio strikes
    K = np.asarray(df_d["K"], dtype=float)
    P = np.asarray(df_d["price_per_1"], dtype=float)

    mask = np.isfinite(K) & np.isfinite(P)
    K = K[mask]
    P = P[mask]

    if K.size < min_points:
        return None

    # Deduplicate strikes by averaging prices
    tmp = pd.DataFrame({"K": K, "P": P}).groupby("K", as_index=False).mean()
    tmp = tmp.sort_values("K")

    if len(tmp) < min_points:
        return None

    return tmp["K"].to_numpy(), tmp["P"].to_numpy()
