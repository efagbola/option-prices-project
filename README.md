## Repository Structure and File Description

This repository implements and compares multiple methods to extract option-implied moments from option price data. Each script corresponds to one methodological approach or a supporting task.

### Configuration and Utilities
- `config.py`  
  Defines data and output paths used consistently across all scripts.

- `utils/price_curves.py`  
  Helper functions to load raw option data and construct option price curves across strikes for each date and area.

### Core Method Implementations
- `method_main.py`  
  Nonparametric implied risk-neutral density method based on Breeden–Litzenberger (1978).  
  Option prices are smoothed across strikes, differentiated to recover the implied density, and moments are computed via numerical integration.  
  Output: `moments_main.csv`.

- `method_parametric_normal.py`  
  Parametric approach assuming a Normal implied distribution.  
  Distribution parameters are estimated per date, and moments are computed analytically.  
  Output: `moments_parametric_normal.csv`.

- `method_parametric_lognormal.py`  
  Parametric approach assuming a Lognormal implied distribution.  
  Parameters are estimated per date in log-space, with analytical formulas used for implied moments.  
  Output: `moments_parametric_lognormal.csv`.

- `method_direct_moments_bkm.py`  
  Direct moment extraction method inspired by Bakshi–Kapadia–Madan (BKM).  
  Moments are computed directly from option prices without explicitly recovering the implied density.  
  Output: direct moments CSV (see script for exact filename).

### Comparison and Execution
- `plot_comparison.py`  
  Loads moment estimates from all methods and produces time-series plots for mean and variance.  
  Used to assess stability and systematic differences across methods.

- `run_all.py`  
  Master script to run all estimation methods sequentially and generate outputs in a single execution.

### Output
- `output/`  
  Contains CSV files with estimated moments and generated comparison plots.

### Notes
- All estimations are cross-sectional (per date).
- The focus is on identification and stability of implied moments, not forecasting.
