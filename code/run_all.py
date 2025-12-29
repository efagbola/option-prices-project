from config import DATA_PATH
from utils.price_curves import load_data, get_available_dates, build_price_curve

caps, floors, swaps = load_data(DATA_PATH)

dates = get_available_dates(caps, floors)
print("Number of dates:", len(dates))

example_date = dates[0]
example_area = caps["area"].unique()[0]

curve = build_price_curve(caps, floors, swaps, example_date, example_area)

print("Example date:", example_date)
print("Example area:", example_area)
print("Strikes:", curve[0])
print("Prices:", curve[1])

from method_main import run_method as run_main
from method_parametric_normal import run_method as run_param_normal

if __name__ == "__main__":
    run_main()
    run_param_normal()
