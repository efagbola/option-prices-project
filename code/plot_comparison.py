import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from config import OUTPUT_PATH

# =======================
# User settings
# =======================
AREA = "EU"          # "EU" or "US"
SHOW_PLOTS = True    # set False if running headless

out_dir = Path(OUTPUT_PATH)
out_dir.mkdir(parents=True, exist_ok=True)

# =======================
# Load outputs
# =======================
df_bl   = pd.read_csv(out_dir / "moments_main.csv")
df_norm = pd.read_csv(out_dir / "moments_parametric_normal.csv")
df_logn = pd.read_csv(out_dir / "moments_parametric_lognormal.csv")
df_bkm  = pd.read_csv(out_dir / "moments_direct_bkm.csv")

# =======================
# Date formatting
# =======================
for d in (df_bl, df_norm, df_logn, df_bkm):
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"])

# =======================
# Filter by area + keep columns
# =======================
def prep(df, mean_name, var_name):
    df = df[df["area"] == AREA].copy()
    df = df[["date", "mean", "variance"]].dropna(subset=["date"])
    df = df.rename(columns={"mean": mean_name, "variance": var_name})
    return df

df_bl   = prep(df_bl,   "mean_bl",   "var_bl")
df_norm = prep(df_norm, "mean_norm", "var_norm")
df_logn = prep(df_logn, "mean_logn", "var_logn")
df_bkm  = prep(df_bkm,  "mean_bkm",  "var_bkm")

# =======================
# Align on BKM dates (reference)
# =======================
df = df_bkm.merge(df_bl, on="date", how="left")
df = df.merge(df_norm, on="date", how="left")
df = df.merge(df_logn, on="date", how="left")
df = df.sort_values("date")

print(f"Dates with BKM available: {len(df)}\n")
print("Non-missing observations by method:")
print(df.notna().sum())

if df.empty:
    raise RuntimeError("No BKM dates available — cannot plot.")

# =======================
# Helper: robust y-limits
# =======================
def set_robust_ylim(ax, series_list, pad_frac=0.10, qlow=0.01, qhigh=0.99):
    s = pd.concat([pd.to_numeric(x, errors="coerce") for x in series_list], axis=0).dropna()
    if s.empty:
        return
    lo = float(s.quantile(qlow))
    hi = float(s.quantile(qhigh))
    span = max(hi - lo, 1e-12)
    ax.set_ylim(lo - pad_frac * span, hi + pad_frac * span)

# =======================
# MEAN PLOT (make BKM visible)
# =======================
fig, ax = plt.subplots(figsize=(10, 5))

# Plot other methods first
if "mean_bl" in df.columns:
    ax.plot(df["date"], df["mean_bl"], label="Nonparametric (BL)", linewidth=2, zorder=2)

if "mean_logn" in df.columns:
    ax.plot(df["date"], df["mean_logn"], label="Parametric Lognormal", linewidth=3, zorder=2)

if "mean_norm" in df.columns:
    ax.plot(df["date"], df["mean_norm"], label="Parametric Normal", linewidth=2, alpha=0.6, zorder=1)

# Plot BKM last + strong styling
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

ax.set_title(f"Implied Mean Inflation Index ({AREA})")
ax.grid(True, alpha=0.25)
ax.legend()

set_robust_ylim(ax, [df.get("mean_bkm"), df.get("mean_bl"), df.get("mean_logn"), df.get("mean_norm")])

fig.tight_layout()
fig.savefig(out_dir / f"mean_comparison_{AREA}.png", dpi=200)

if SHOW_PLOTS:
    plt.show()
else:
    plt.close(fig)

# =======================
# VARIANCE PLOT (log scale + robust limits)
# =======================
fig, ax = plt.subplots(figsize=(10, 5))

if "var_bl" in df.columns:
    ax.plot(df["date"], df["var_bl"], label="Nonparametric (BL)", linewidth=2, zorder=2)

if "var_logn" in df.columns:
    ax.plot(df["date"], df["var_logn"], label="Parametric Lognormal", linewidth=3, zorder=2)

if "var_norm" in df.columns:
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
ax.set_title(f"Implied Variance ({AREA}) — log scale")
ax.grid(True, alpha=0.25)
ax.legend()

# Robust y-limits for log scale (avoid zeros)
vars_all = pd.concat([
    pd.to_numeric(df.get("var_bkm"), errors="coerce"),
    pd.to_numeric(df.get("var_bl"), errors="coerce"),
    pd.to_numeric(df.get("var_logn"), errors="coerce"),
    pd.to_numeric(df.get("var_norm"), errors="coerce"),
], axis=0).dropna()

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
# DIFFERENCE PLOTS vs BKM (makes overlap obvious)
# =======================
fig, ax = plt.subplots(figsize=(10, 4))

if "mean_bl" in df.columns:
    ax.plot(df["date"], 10000 * (df["mean_bl"] - df["mean_bkm"]), label="BL − BKM")
if "mean_logn" in df.columns:
    ax.plot(df["date"], 10000 * (df["mean_logn"] - df["mean_bkm"]), label="Lognormal − BKM")
if "mean_norm" in df.columns:
    ax.plot(df["date"], 10000 * (df["mean_norm"] - df["mean_bkm"]), label="Normal − BKM")

ax.axhline(0.0, linewidth=1)
ax.set_title(f"Mean Differences vs BKM ({AREA})")
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
