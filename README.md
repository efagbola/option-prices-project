# Option-Implied Inflation Moments

This repository implements and compares alternative methodologies to extract **option-implied moments of the inflation distribution** from cap and floor prices.

The project is structured to allow the supervisor to directly inspect:
1. The implemented **methodologies (code)**
2. The resulting **moment estimates (CSV outputs)**
3. The **comparative analysis (plots)**

The emphasis is on **methodological replication and comparison**, not forecasting.

---

## Data Description

The data consist of:
- Inflation caps
- Inflation floors
- Inflation swaps

Caps and floors are used to infer information about the risk-neutral distribution of future inflation.  
Swaps provide:
- Discount factors
- A proxy for the forward (risk-neutral) mean inflation rate

All computations are performed **cross-sectionally by date**.

---

## Methodological Framework

The project implements four classes of methods commonly used in the literature to recover distributional information from option prices.

### 1. Nonparametric Implied Density (Breeden–Litzenberger)

This approach recovers the full risk-neutral density implied by option prices.

Steps:
- Construct a price curve across strikes
- Smooth prices using spline interpolation
- Apply the Breeden–Litzenberger identity (second derivative)
- Compute moments via numerical integration

This method is flexible but sensitive to smoothing and data quality.

**Output**
- `moments_main.csv`

---

### 2. Parametric Normal Approximation

This approach assumes the inflation distribution is Normal.

Steps:
- Estimate mean and variance parameters
- Compute implied moments analytically

This method is simple and stable but imposes strong distributional assumptions.

**Output**
- `moments_parametric_normal.csv`

---

### 3. Parametric Lognormal Approximation

This approach assumes a Lognormal inflation distribution.

Steps:
- Estimate parameters in log-space
- Recover analytical expressions for moments

This method allows for skewness but restricts the shape of the distribution.

**Output**
- `moments_parametric_lognormal.csv`

---

### 4. Direct Moment Extraction (BKM-style)

This approach follows the logic of Bakshi, Kapadia, and Madan (2003).

Steps:
- Use static replication formulas
- Recover raw moments directly from option prices
- Convert raw moments into central moments

This method avoids explicit density estimation.

**Output**
- `moments_direct_bkm.csv`

---

## Comparison Strategy

Methods are compared along several dimensions:

- Time-series evolution of implied **mean**
- Time-series evolution of implied **variance**
- Level comparison (time-series averages)
- Volatility comparison (standard deviation over time)
- Diagnostic gap analysis between methods

Comparison plots are generated for a common set of dates.

---

## Repository Structure

```text
option-prices-project/
│
├── code/
│   ├── config.py                     # paths and configuration
│   ├── run_all.py                    # runs all estimation methods
│   ├── plot_comparison.py            # produces comparison plots
│   ├── method_main.py                # nonparametric BL method
│   ├── method_parametric_normal.py   # parametric normal method
│   ├── method_parametric_lognormal.py# parametric lognormal method
│   ├── method_direct_bkm.py           # direct BKM-style method
│   └── utils/
│       └── price_curves.py            # data loading and curve construction
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
