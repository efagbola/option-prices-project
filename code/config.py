# config.py
import os
from pathlib import Path

# Base directory = .../option-prices-project
BASE_DIR = Path(__file__).resolve().parent.parent  # code/.. -> project root
CODE_DIR = BASE_DIR / "code"

# -----------------------
# Data location (local)
# -----------------------
# Expect cleaned CSVs to live in: code/Data/
#   - cleaned_caps_quotes_1y.csv
#   - cleaned_floors_quotes_1y.csv
#   - cleaned_swaps_curves_1y.csv
DATA_DIR = CODE_DIR / "Data"
DATA_PATH = str(DATA_DIR) + os.sep

# -----------------------
# Output location (local)
# -----------------------
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = str(OUTPUT_DIR) + os.sep
