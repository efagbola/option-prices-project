import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from config import OUTPUT_PATH

# =======================
# User settings
# =======================
AREA = "EU"
SHOW_PLOTS = True

out_dir = Path(OUTPUT_PATH)

# =======================
# Load outputs
# =======================
df_np   = pd.read_csv(out_dir / "moments_main.csv")
df_norm = pd.read_csv(out_dir / "moments_parametric_normal.csv")
df_logn = pd.read_csv(out_dir / "moments_parametric_lognormal.csv")
df_bkm  = pd.read_csv(out_dir / "moments_direct_bkm.csv")

# =======================
# Date formatting
# =======================
for df in [df_np, df_norm, df_logn, df_bkm]:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

# =======================
# Filter area + select columns
# =======================
df_np   = df_np[df_np["area"] == AREA][["date", "mean", "variance"]]
df_norm = df_norm[df_norm["area"] == AREA][["date", "mean", "variance"]]
df_logn = df_logn[df_logn["area"] == AREA][["date", "mean", "variance"]]
df_bkm  = df_bkm[df_bkm["area"] == AREA][["date", "mean", "variance"]]

# Rename columns
df_np   = df_np.rename(columns={"mean": "mean_np",   "variance": "var_np"})
df_norm = df_norm.rename(columns={"mean": "mean_norm","variance": "var_norm"})
df_logn = df_logn.rename(columns={"mean": "mean_logn","variance": "var_logn"})
df_bkm  = df_bkm.rename(columns={"mean": "mean_bkm", "variance": "var_bkm"})

# =======================
# Align on BKM dates (reference)
# =======================
df = df_bkm.merge(df_np,   on="date", how="left")
df = df.merge(df_norm, on="date", how="left")
df = df.merge(df_logn, on="date", how="left")
df = df.sort_values("date")

print(f"Dates with BKM available: {len(df)}")
print("\nNon-missing observations by method:")
print(df.notna().sum())

if df.empty:
    raise RuntimeError("No overlapping dates for BKM — cannot plot.")

# =======================
# MEAN PLOT
# =======================
plt.figure(figsize=(10, 5))

plt.plot(df["date"], df["mean_bkm"], label="Direct (BKM)", linewidth=2)

if "mean_np" in df:
    plt.plot(df["date"], df["mean_np"], label="Nonparametric (BL)", linewidth=2)

if "mean_logn" in df:
    plt.plot(df["date"], df["mean_logn"], label="Parametric Lognormal", linewidth=2)

if "mean_norm" in df:
    plt.plot(df["date"], df["mean_norm"], label="Parametric Normal", alpha=0.5)

plt.title(f"Implied Mean Inflation Index ({AREA})")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / f"mean_comparison_{AREA}.png", dpi=200)

if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

# =======================
# VARIANCE PLOT (log scale)
# =======================
plt.figure(figsize=(10, 5))

plt.plot(df["date"], df["var_bkm"], label="Direct (BKM)", linewidth=2)

if "var_np" in df:
    plt.plot(df["date"], df["var_np"], label="Nonparametric (BL)", linewidth=2)

if "var_logn" in df:
    plt.plot(df["date"], df["var_logn"], label="Parametric Lognormal", linewidth=2)

if "var_norm" in df:
    plt.plot(df["date"], df["var_norm"], label="Parametric Normal", alpha=0.5)

plt.yscale("log")
plt.title(f"Implied Variance ({AREA}) — log scale")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / f"variance_comparison_{AREA}.png", dpi=200)

if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

# =======================
# NUMERICAL COMPARISON
# =======================
print("\n=== LEVEL COMPARISON (time-series averages) ===")
print("Mean level BL:     ", df["mean_np"].mean())
print("Mean level Normal: ", df["mean_norm"].mean())
print("Mean level Lognorm:", df["mean_logn"].mean())
print("Mean level BKM:    ", df["mean_bkm"].mean())

print("\nVariance level BL:     ", df["var_np"].mean())
print("Variance level Normal:", df["var_norm"].mean())
print("Variance level Lognorm:", df["var_logn"].mean())
print("Variance level BKM:    ", df["var_bkm"].mean())

print("\n=== VOLATILITY COMPARISON (std over time) ===")
print("Std(mean) BL:     ", df["mean_np"].std())
print("Std(mean) Normal: ", df["mean_norm"].std())
print("Std(mean) Lognorm:", df["mean_logn"].std())
print("Std(mean) BKM:    ", df["mean_bkm"].std())

print("\nStd(variance) BL:     ", df["var_np"].std())
print("Std(variance) Normal:", df["var_norm"].std())
print("Std(variance) Lognorm:", df["var_logn"].std())
print("Std(variance) BKM:    ", df["var_bkm"].std())

# =======================
# DISAGREEMENT DIAGNOSTICS
# =======================
df["gap_mean_np_bkm"] = (df["mean_np"] - df["mean_bkm"]).abs()
df["gap_var_np_bkm"]  = (df["var_np"]  - df["var_bkm"]).abs()

print("\n=== Largest disagreements (BL vs BKM) ===")

print("\nTop mean gaps:")
print(
    df.sort_values("gap_mean_np_bkm", ascending=False)
      [["date", "gap_mean_np_bkm"]]
      .head(10)
      .to_string(index=False)
)

print("\nTop variance gaps:")
print(
    df.sort_values("gap_var_np_bkm", ascending=False)
      [["date", "gap_var_np_bkm"]]
      .head(10)
      .to_string(index=False)
)
