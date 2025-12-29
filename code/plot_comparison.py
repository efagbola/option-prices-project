import pandas as pd
import matplotlib.pyplot as plt

from config import OUTPUT_PATH

# Load outputs
df_np = pd.read_csv(OUTPUT_PATH + "moments_main.csv")
df_par = pd.read_csv(OUTPUT_PATH + "moments_parametric_normal.csv")

# Ensure dates are datetime
df_np["date"] = pd.to_datetime(df_np["date"])
df_par["date"] = pd.to_datetime(df_par["date"])

# Keep one area (EU)
df_np = df_np[df_np["area"] == "EU"]
df_par = df_par[df_par["area"] == "EU"]

# Sort by date
df_np = df_np.sort_values("date")
df_par = df_par.sort_values("date")

# --- MEAN ---
plt.figure()
plt.plot(df_np["date"], df_np["mean"], label="Nonparametric", linewidth=2)
plt.plot(df_par["date"], df_par["mean"], label="Parametric Normal", linewidth=2)
plt.title("Implied Mean Over Time (EU)")
plt.legend()
plt.tight_layout()
plt.show()

# --- VARIANCE ---
plt.figure()
plt.plot(df_np["date"], df_np["variance"], label="Nonparametric", linewidth=2)
plt.plot(df_par["date"], df_par["variance"], label="Parametric Normal", linewidth=2)
plt.title("Implied Variance Over Time (EU)")
plt.legend()
plt.tight_layout()
plt.show()

print("Std(mean) nonparametric:", df_np["mean"].std())
print("Std(mean) parametric:", df_par["mean"].std())

print("Std(variance) nonparametric:", df_np["variance"].std())
print("Std(variance) parametric:", df_par["variance"].std())
