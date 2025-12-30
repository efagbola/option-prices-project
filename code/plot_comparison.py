import pandas as pd
import matplotlib.pyplot as plt

from config import OUTPUT_PATH


# Load outputs
df_np = pd.read_csv(OUTPUT_PATH + "moments_main.csv")
df_par = pd.read_csv(OUTPUT_PATH + "moments_parametric_normal.csv")
df_bkm = pd.read_csv(OUTPUT_PATH + "moments_direct_bkm.csv")

# Ensure dates are datetime
df_np["date"] = pd.to_datetime(df_np["date"])
df_par["date"] = pd.to_datetime(df_par["date"])
df_bkm["date"] = pd.to_datetime(df_bkm["date"])

# Keep one area (EU)
df_np = df_np[df_np["area"] == "EU"]
df_par = df_par[df_par["area"] == "EU"]
df_bkm = df_bkm[df_bkm["area"] == "EU"]

# Sort by date
df_np = df_np.sort_values("date")
df_par = df_par.sort_values("date")
df_bkm = df_bkm.sort_values("date")

# --- MEAN ---
plt.figure()
plt.plot(df_np["date"], df_np["mean"], label="Nonparametric (BL)", linewidth=2)
plt.plot(df_par["date"], df_par["mean"], label="Parametric Normal", linewidth=2)
plt.plot(df_bkm["date"], df_bkm["mean"], label="Direct (BKM)", linewidth=2, linestyle="--")
plt.title("Implied Mean Over Time (EU)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PATH + "mean_comparison_EU.png", dpi=200)
plt.show()

# --- VARIANCE ---
plt.figure()
plt.plot(df_np["date"], df_np["variance"], label="Nonparametric (BL)", linewidth=2)
plt.plot(df_par["date"], df_par["variance"], label="Parametric Normal", linewidth=2)
plt.plot(df_bkm["date"], df_bkm["variance"], label="Direct (BKM)", linewidth=2, linestyle="--")
plt.title("Implied Variance Over Time (EU)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PATH + "variance_comparison_EU.png", dpi=200)
plt.show()

print("Std(mean) nonparametric:", df_np["mean"].std())
print("Std(mean) parametric:", df_par["mean"].std())

print("Std(variance) nonparametric:", df_np["variance"].std())
print("Std(variance) parametric:", df_par["variance"].std())

print("Std(mean) direct BKM:", df_bkm["mean"].std())
print("Std(variance) direct BKM:", df_bkm["variance"].std())

