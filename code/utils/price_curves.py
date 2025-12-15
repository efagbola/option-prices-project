import pandas as pd

def load_data(data_path):
    """
    Load caps, floors, and swaps data
    """
    caps = pd.read_csv(data_path + "cleaned_caps_quotes_1y.csv")
    floors = pd.read_csv(data_path + "cleaned_floors_quotes_1y.csv")
    swaps = pd.read_csv(data_path + "cleaned_swaps_curves_1y.csv")
    return caps, floors, swaps


def get_available_dates(caps, floors):
    """
    Return sorted list of dates common to caps and floors
    """
    dates_caps = set(caps["date"].unique())
    dates_floors = set(floors["date"].unique())
    return sorted(dates_caps.intersection(dates_floors))


def build_price_curve(caps, floors, swaps, date, area):
    """
    Build option price curve for one date and one area
    Returns strikes (k) and prices per unit notional
    """
    caps_d = caps[(caps["date"] == date) & (caps["area"] == area)]
    floors_d = floors[(floors["date"] == date) & (floors["area"] == area)]

    if caps_d.empty or floors_d.empty:
        return None

    prices = pd.concat([
        caps_d[["k", "price_per_1"]],
        floors_d[["k", "price_per_1"]]
    ])

    prices = prices.sort_values("k")

    return prices["k"].values, prices["price_per_1"].values

