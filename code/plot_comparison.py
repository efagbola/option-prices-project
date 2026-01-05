import pandas as pd
import matplotlib.pyplot as plt

from config import OUTPUT_PATH

AREA = "EU"

# ---------- Load outputs ----------
df_np = pd.read_csv(OUTPUT_PATH + "moments_main.csv")
df_norm = pd.read_csv(OUTPUT_PATH + "moments_parametric_normal.csv")
df_logn = pd.read_csv(OUTPUT_PATH + "moments_parametric_lognormal.csv")
df_bkm = pd.read_csv(OUTPUT_PATH + "moments_direct_bkm.csv")

# ---------- Ensure dates are datetime ----------
for df in [df_np, df_norm, df_logn, df_bkm]:
    df["date"] = pd.to_datetime(df["date"])

# ---------- Filter to one area and keep needed columns ----------
df_np = df_np[df_np["area"] == AREA][["date", "mean", "variance"]].copy()
df_norm = df_norm[df_norm["area"] == AREA][["date", "mean", "variance"]].copy()
df_logn = df_logn[df_logn["area"] == AREA][["date", "mean", "variance"]].copy()
df_bkm = df_bkm[df_bkm["area"] == AREA][["date", "mean", "variance"]].copy()

# ---------- Rename columns by method ----------
df_np = df_np.rename(columns={"mean": "mean_np", "variance": "var_np"})
df_norm = df_norm.rename(columns={"mean": "mean_norm", "variance": "var_norm"})
df_logn = df_logn.rename(columns={"mean": "mean_logn", "variance": "var_logn"})
df_bkm = df_bkm.rename(columns={"mean": "mean_bkm", "variance": "var_bkm"})

# ---------- Align on common dates (important) ----------
df = df_np.merge(df_norm, on="date", how="inner") \
          .merge(df_logn, on="date", how="inner") \
          .merge(df_bkm, on="date", how="inner")

df = df.sort_values("date")

# ---------- PLOTS ----------
# MEAN
plt.figure()
plt.plot(df["date"], df["mean_np"], label="Nonparametric (BL)", linewidth=2)
plt.plot(df["date"], df["mean_norm"], label="Parametric Normal", linewidth=2)
plt.plot(df["date"], df["mean_logn"], label="Parametric Lognormal", linewidth=2)
plt.plot(df["date"], df["mean_bkm"], label="Direct (BKM)", linewidth=2, linestyle="--")
plt.title(f"Implied Mean Over Time ({AREA})")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PATH + f"mean_comparison_{AREA}.png", dpi=200)
plt.show()

# VARIANCE
plt.figure()
plt.plot(df["date"], df["var_np"], label="Nonparametric (BL)", linewidth=2)
plt.plot(df["date"], df["var_norm"], label="Parametric Normal", linewidth=2)
plt.plot(df["date"], df["var_logn"], label="Parametric Lognormal", linewidth=2)
plt.plot(df["date"], df["var_bkm"], label="Direct (BKM)", linewidth=2, linestyle="--")
plt.title(f"Implied Variance Over Time ({AREA})")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PATH + f"variance_comparison_{AREA}.png", dpi=200)
plt.show()

# ---------- LEVELS (required) ----------
print("\n=== LEVEL COMPARISON (time-series averages) ===")
print("Mean level NP:", df["mean_np"].mean())
print("Mean level Normal:", df["mean_norm"].mean())
print("Mean level Lognormal:", df["mean_logn"].mean())
print("Mean level BKM:", df["mean_bkm"].mean())

print("Variance level NP:", df["var_np"].mean())
print("Variance level Normal:", df["var_norm"].mean())
print("Variance level Lognormal:", df["var_logn"].mean())
print("Variance level BKM:", df["var_bkm"].mean())

# ---------- VOLATILITY (required) ----------
print("\n=== VOLATILITY COMPARISON (std over time) ===")
print("Std(mean) NP:", df["mean_np"].std())
print("Std(mean) Normal:", df["mean_norm"].std())
print("Std(mean) Lognormal:", df["mean_logn"].std())
print("Std(mean) BKM:", df["mean_bkm"].std())

print("Std(variance) NP:", df["var_np"].std())
print("Std(variance) Normal:", df["var_norm"].std())
print("Std(variance) Lognormal:", df["var_logn"].std())
print("Std(variance) BKM:", df["var_bkm"].std())

# ---------- DIAGNOSIS: disagreements ----------
df["gap_mean_np_bkm"] = (df["mean_np"] - df["mean_bkm"]).abs()
df["gap_var_np_bkm"] = (df["var_np"] - df["var_bkm"]).abs()
df["gap_mean_np_logn"] = (df["mean_np"] - df["mean_logn"]).abs()
df["gap_var_np_logn"] = (df["var_np"] - df["var_logn"]).abs()

print("\n=== DIAGNOSIS: largest disagreements ===")

print("\nTop mean gaps (NP vs BKM):")
print(df.sort_values("gap_mean_np_bkm", ascending=False)[["date", "gap_mean_np_bkm"]].head(10).to_string(index=False))

print("\nTop variance gaps (NP vs BKM):")
print(df.sort_values("gap_var_np_bkm", ascending=False)[["date", "gap_var_np_bkm"]].head(10).to_string(index=False))

print("\nTop mean gaps (NP vs Lognormal):")
print(df.sort_values("gap_mean_np_logn", ascending=False)[["date", "gap_mean_np_logn"]].head(10).to_string(index=False))

print("\nTop variance gaps (NP vs Lognormal):")
print(df.sort_values("gap_var_np_logn", ascending=False)[["date", "gap_var_np_logn"]].head(10).to_string(index=False))

print(f"\nAligned dates used (all four methods): {len(df)}")
