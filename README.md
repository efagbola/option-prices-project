## Repository Structure and File Description

This project estimates and compares option-implied moments of the inflation distribution using cap and floor prices.  
The focus is on methodology, replication, and comparison across different approaches.

The supervisor inspects:
1. Code files
2. Output CSV files
3. Plots and results

---

## Data

The dataset consists of:
- Inflation caps
- Inflation floors
- Inflation swaps (used for discounting and forward means)

Data are used to construct cross-sectional option price curves at each date.

All results are computed **date by date**.

---

## Methods Implemented

### 1. Nonparametric Implied Density (Breeden–Litzenberger)

- Smooth option price curve across strikes using splines
- Recover implied risk-neutral density via second derivative
- Compute moments via numerical integration

**Output**
- `moments_main.csv`

---

### 2. Parametric Normal Distribution

- Assume normally distributed inflation
- Estimate mean and variance from strikes
- Compute moments analytically

**Output**
- `moments_parametric_normal.csv`

---

### 3. Parametric Lognormal Distribution

- Assume lognormal distribution for inflation
- Estimate parameters in log-space
- Compute analytical moments

**Output**
- `moments_parametric_lognormal.csv`

---

### 4. Direct Moment Extraction (BKM-style)

- No explicit density recovery
- Use static replication of payoff moments
- Moments obtained directly from option prices

**Output**
- `moments_direct_bkm.csv`

---

## Comparison

Methods are compared using:
- Time-series plots of implied **mean**
- Time-series plots of implied **variance**
- Level comparison (time-series averages)
- Volatility comparison (standard deviation over time)
- Diagnostic gap analysis across methods

---

## Repository Structure

```text
option-prices-project/
│
├── code/
│   ├── config.py
│   ├── run_all.py
│   ├── plot_comparison.py
│   ├── method_main.py
│   ├── method_parametric_normal.py
│   ├── method_parametric_lognormal.py
│   ├── method_direct_bkm.py
│   └── utils/
│       └── price_curves.py
│
├── output/
│   ├── moments_main.csv
│   ├── moments_parametric_normal.csv
│   ├── moments_parametric_lognormal.csv
│   ├── moments_direct_bkm.csv
│   ├── mean_comparison_EU.png
│   └── variance_comparison_EU.png
│
└── README.md

