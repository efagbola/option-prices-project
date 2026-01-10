import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from config import OUTPUT_PATH, DATA_PATH  # <-- make sure DATA_PATH exists in config

# =======================
# User settings
# =======================
AREA = "EU"          # "EU" or "US"
SHOW_PLOTS = True    # set False if running headless

out_dir = Path(OUTPUT_PATH)
out_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------
def merge_on_intersection(df_left, df_right, on=("date", "area", "n"), suffixes=("_L", "_R")):
    """
    Inner join to enforce same-sample comparisons.
    """
    left_cols = list(on) + [c for c in df_left.columns if c not in on]
    right_cols = list(on) + [c for c in df_right.columns if c not in on]
    return df_left[left_cols].merge(df_right[right_cols], on=list(on), how="inner", suffixes=suffixes)


def set_robust_ylim(ax, series_list, pad_frac=0.10, qlow=0.01, qhigh=0.99):
    s = pd.concat([pd.to_numeric(x, errors="coerce") for x in series_list if x is not None], axis=0).dropna()
    if s.empty:
        return
    lo = float(s.quantile(qlow))
    hi = float(s.quantile(qhigh))
    span = max(hi - lo, 1e-12)
    ax.set_ylim(lo - pad_frac * span, hi + pad_frac * span)


def prep(df, mean_name, var_name):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[df["area"] == AREA].copy()
    df = df[["date", "area", "n", "mean", "variance"]].dropna(subset=["date"])
    return df.rename(columns={"mean": mean_name, "variance": var_name})


# =======================
# Load outputs
# =======================
df_bl   = pd.read_csv(out_dir / "moments_main.csv")
df_norm = pd.read_csv(out_dir / "moments_parametric_normal.csv")
df_logn = pd.read_csv(out_dir / "moments_parametric_lognormal.csv")
df_bkm  = pd.read_csv(out_dir / "moments_direct_bkm.csv")

# Load K_star (swap-implied mean) for diagnostics
swaps = pd.read_csv(Path(DATA_PATH) / "cleaned_swaps_curves_1y.csv")
swaps["date"] = pd.to_datetime(swaps["date"], errors="coerce")
swaps = swaps.dropna(subset=["date"])
swaps = swaps[swaps["area"] == AREA].copy()
swaps_base = swaps[["date", "area", "n", "K_star"]].copy()

# =======================
# Filter + standardize columns
# =======================
df_bl   = prep(df_bl,   "mean_bl",   "var_bl")
df_norm = prep(df_norm, "mean_norm", "var_norm")
df_logn = prep(df_logn, "mean_logn", "var_logn")
df_bkm  = prep(df_bkm,  "mean_bkm",  "var_bkm")

# =======================
# Align on intersection sample (BL ∩ BKM ∩ Lognormal ∩ Normal)
# =======================
df = merge_on_intersection(df_bkm, df_bl, on=("date", "area", "n"))
df = merge_on_intersection(df, df_logn, on=("date", "area", "n"))
df = merge_on_intersection(df, df_norm, on=("date", "area", "n"))
df = df.merge(swaps_base, on=["date", "area", "n"], how="left")
df = df.sort_values("date")

print("Coverage (rows):")
print(f"  BL:   {len(df_bl)}")
print(f"  BKM:  {len(df_bkm)}")
print(f"  LOGN: {len(df_logn)}")
print(f"  NORM: {len(df_norm)}")
print(f"  Intersection (all four): {len(df)}\n")

if df.empty:
    raise RuntimeError("Intersection sample is empty — cannot plot.")

# =======================
# MEAN PLOT
# =======================
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(df["date"], df["mean_bl"],   label="Nonparametric (BL)", linewidth=2, zorder=2)
ax.plot(df["date"], df["mean_logn"], label="Parametric Lognormal", linewidth=3, zorder=2)
ax.plot(df["date"], df["mean_norm"], label="Parametric Normal", linewidth=2, alpha=0.6, zorder=1)

ax.plot(
    df["date"], df["mean_bkm"],
    label="Direct (BKM)",
    linewidth=4,
    linestyle="--",
    marker="o",
    markersize=5,
    markeredgewidth=1.5,
    markeredgecolor="black",
    zorder=10,
)

ax.set_title(f"Implied Mean Inflation Index ({AREA}) — intersection sample")
ax.grid(True, alpha=0.25)
ax.legend()

set_robust_ylim(ax, [df["mean_bkm"], df["mean_bl"], df["mean_logn"], df["mean_norm"]])

fig.tight_layout()
fig.savefig(out_dir / f"mean_comparison_{AREA}.png", dpi=200)

if SHOW_PLOTS:
    plt.show()
else:
    plt.close(fig)

# =======================
# VARIANCE PLOT (log scale)
# =======================
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(df["date"], df["var_bl"],   label="Nonparametric (BL)", linewidth=2, zorder=2)
ax.plot(df["date"], df["var_logn"], label="Parametric Lognormal", linewidth=3, zorder=2)
ax.plot(df["date"], df["var_norm"], label="Parametric Normal", linewidth=2, alpha=0.6, zorder=1)

ax.plot(
    df["date"], df["var_bkm"],
    label="Direct (BKM)",
    linewidth=4,
    linestyle="--",
    marker="o",
    markersize=4,
    markeredgewidth=1.2,
    markeredgecolor="black",
    zorder=10,
)

ax.set_yscale("log")
ax.set_title(f"Implied Variance ({AREA}) — log scale — intersection sample")
ax.grid(True, alpha=0.25)
ax.legend()

vars_all = pd.concat([df["var_bkm"], df["var_bl"], df["var_logn"], df["var_norm"]], axis=0)
vars_all = pd.to_numeric(vars_all, errors="coerce").dropna()
vars_all = vars_all[vars_all > 0]
if not vars_all.empty:
    lo = float(vars_all.quantile(0.01))
    hi = float(vars_all.quantile(0.99))
    ax.set_ylim(lo * 0.8, hi * 1.2)

fig.tight_layout()
fig.savefig(out_dir / f"variance_comparison_{AREA}.png", dpi=200)

if SHOW_PLOTS:
    plt.show()
else:
    plt.close(fig)

# =======================
# DIFFERENCE PLOTS vs BKM
# =======================
fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(df["date"], 10000 * (df["mean_bl"]   - df["mean_bkm"]), label="BL − BKM")
ax.plot(df["date"], 10000 * (df["mean_logn"] - df["mean_bkm"]), label="Lognormal − BKM")
ax.plot(df["date"], 10000 * (df["mean_norm"] - df["mean_bkm"]), label="Normal − BKM")

ax.axhline(0.0, linewidth=1)
ax.set_title(f"Mean Differences vs BKM ({AREA}) — intersection sample")
ax.set_ylabel("Difference (bps)")
ax.grid(True, alpha=0.25)
ax.legend()

fig.tight_layout()
fig.savefig(out_dir / f"mean_diff_vs_bkm_{AREA}.png", dpi=200)

if SHOW_PLOTS:
    plt.show()
else:
    plt.close(fig)

# =======================
# MEAN GAP vs K_star (diagnostic)
# =======================
if "K_star" in df.columns and df["K_star"].notna().any():
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(df["date"], df["mean_bl"]   - df["K_star"], label="BL mean − K*")
    ax.plot(df["date"], df["mean_bkm"]  - df["K_star"], label="BKM mean − K*")
    ax.plot(df["date"], df["mean_logn"] - df["K_star"], label="Lognormal mean − K*")
    ax.plot(df["date"], df["mean_norm"] - df["K_star"], label="Normal mean − K*")

    ax.axhline(0.0, linewidth=1)
    ax.set_title(f"Mean Gap vs Swap-Implied K* ({AREA}) — intersection sample")
    ax.set_ylabel("Gap (gross units)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / f"mean_gap_vs_kstar_{AREA}.png", dpi=200)

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

# =======================
# NUMERICAL COMPARISON
# =======================
print("\n=== LEVEL COMPARISON (time-series averages) ===")
print("Mean level BL:     ", df["mean_bl"].mean())
print("Mean level Normal: ", df["mean_norm"].mean())
print("Mean level Lognorm:", df["mean_logn"].mean())
print("Mean level BKM:    ", df["mean_bkm"].mean())

print("\nVariance level BL:      ", df["var_bl"].mean())
print("Variance level Normal:  ", df["var_norm"].mean())
print("Variance level Lognorm: ", df["var_logn"].mean())
print("Variance level BKM:     ", df["var_bkm"].mean())

print("\n=== VOLATILITY COMPARISON (std over time) ===")
print("Std(mean) BL:      ", df["mean_bl"].std())
print("Std(mean) Normal:  ", df["mean_norm"].std())
print("Std(mean) Lognorm: ", df["mean_logn"].std())
print("Std(mean) BKM:     ", df["mean_bkm"].std())

print("\nStd(variance) BL:      ", df["var_bl"].std())
print("Std(variance) Normal:  ", df["var_norm"].std())
print("Std(variance) Lognorm: ", df["var_logn"].std())
print("Std(variance) BKM:     ", df["var_bkm"].std())

# =======================
# DISAGREEMENT DIAGNOSTICS (BL vs BKM)
# =======================
df["gap_mean_bl_bkm"] = (df["mean_bl"] - df["mean_bkm"]).abs()
df["gap_var_bl_bkm"] = (df["var_bl"] - df["var_bkm"]).abs()

print("\n=== Largest disagreements (BL vs BKM) ===")
print("\nTop mean gaps:")
print(
    df.sort_values("gap_mean_bl_bkm", ascending=False)[["date", "gap_mean_bl_bkm"]]
      .head(10)
      .to_string(index=False)
)

print("\nTop variance gaps:")
print(
    df.sort_values("gap_var_bl_bkm", ascending=False)[["date", "gap_var_bl_bkm"]]
      .head(10)
      .to_string(index=False)
)
