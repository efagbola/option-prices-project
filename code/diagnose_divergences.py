import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import UnivariateSpline

from config import DATA_PATH, OUTPUT_PATH
from utils.price_curves import load_data, build_price_curve


# =======================
# User settings
# =======================
AREA = "EU"              # "EU" or "US"
TOP_K = 15               # number of worst divergence dates to inspect
GRID_N = 400
EDGE_N = 10              # edge points for edge-mass diagnostic
SHOW_PLOTS = False       # set True if you want windows to pop up


out_dir = Path(OUTPUT_PATH)
out_dir.mkdir(parents=True, exist_ok=True)


# -----------------------------
# BL helpers (same spirit as method_main.py)
# -----------------------------
def choose_grid(strikes, n=400):
    strikes = np.asarray(strikes, dtype=float)
    kmin, kmax = float(np.min(strikes)), float(np.max(strikes))
    if not np.isfinite(kmin) or not np.isfinite(kmax) or (kmax - kmin) <= 1e-10:
        return None
    return np.linspace(kmin, kmax, int(n))


def implied_pdf_from_prices(spline, grid):
    grid = np.asarray(grid, dtype=float)
    second = spline.derivative(n=2)(grid)
    pdf = np.asarray(second, dtype=float)
    pdf[~np.isfinite(pdf)] = 0.0
    pdf = np.maximum(pdf, 0.0)
    area = float(np.trapz(pdf, grid))
    if (not np.isfinite(area)) or area <= 0.0:
        return None
    return pdf / area


def bl_edge_mass(grid, pdf, edge_n=10):
    grid = np.asarray(grid, dtype=float)
    pdf = np.asarray(pdf, dtype=float)
    m = float(np.trapz(pdf[:edge_n], grid[:edge_n]) + np.trapz(pdf[-edge_n:], grid[-edge_n:]))
    return m


def build_call_equivalent_curve(caps, floors, swaps, date, area):
    """
    Build call-equivalent prices across strikes using:
      - OTM caps for K >= K*
      - OTM floors converted to calls via parity: C(K) = P(K) + B*(K* - K), for K <= K*
    """
    # We request min_points=1 here; we’ll filter later.
    cap_curve = build_price_curve(caps, floors, swaps, date, area, instrument="cap", min_points=1)
    floor_curve = build_price_curve(caps, floors, swaps, date, area, instrument="floor", min_points=1)
    if cap_curve is None or floor_curve is None:
        return None

    Kc, Pc = cap_curve
    Kp, Pp = floor_curve

    # Need B and K_star from swaps
    srow = swaps[(swaps["date"] == str(date)) & (swaps["area"] == area)]
    if srow.empty:
        return None
    B = float(srow["B"].iloc[0])
    K_star = float(srow["K_star"].iloc[0])
    if not (np.isfinite(B) and B > 0 and np.isfinite(K_star) and K_star > 0):
        return None

    calls_mask = (Kc >= K_star) & np.isfinite(Pc) & (Pc >= -1e-12)
    puts_mask  = (Kp <= K_star) & np.isfinite(Pp) & (Pp >= -1e-12)

    K_calls, C_calls = Kc[calls_mask], Pc[calls_mask]
    K_puts,  P_puts  = Kp[puts_mask],  Pp[puts_mask]
    if len(K_calls) == 0 or len(K_puts) == 0:
        return None

    C_from_puts = P_puts + B * (K_star - K_puts)

    K_all = np.concatenate([K_puts, K_calls])
    C_all = np.concatenate([C_from_puts, C_calls])

    # sort & average duplicates
    order = np.argsort(K_all)
    K_all, C_all = K_all[order], C_all[order]

    df = pd.DataFrame({"K": K_all, "C": C_all}).groupby("K", as_index=False).mean()
    K_all = df["K"].to_numpy(dtype=float)
    C_all = df["C"].to_numpy(dtype=float)

    ok = np.isfinite(K_all) & np.isfinite(C_all)
    K_all, C_all = K_all[ok], C_all[ok]
    if len(K_all) < 4:
        return None

    return K_all, C_all, B, K_star


def parity_residual_on_overlap(caps, floors, swaps, date, area, tol=1e-10):
    """
    On strikes that exist in both cap and floor quotes, check:
      C(K) - P(K) - B*(K* - K)  ≈ 0
    Return median absolute residual; NaN if no overlap.
    """
    cap_curve = build_price_curve(caps, floors, swaps, date, area, instrument="cap", min_points=1)
    floor_curve = build_price_curve(caps, floors, swaps, date, area, instrument="floor", min_points=1)
    if cap_curve is None or floor_curve is None:
        return np.nan

    Kc, C = cap_curve
    Kp, P = floor_curve

    srow = swaps[(swaps["date"] == str(date)) & (swaps["area"] == area)]
    if srow.empty:
        return np.nan
    B = float(srow["B"].iloc[0])
    K_star = float(srow["K_star"].iloc[0])

    dc = pd.DataFrame({"K": np.asarray(Kc, float), "C": np.asarray(C, float)})
    dp = pd.DataFrame({"K": np.asarray(Kp, float), "P": np.asarray(P, float)})

    # merge on strike with rounding to avoid float mismatch
    dc["K_r"] = dc["K"].round(6)
    dp["K_r"] = dp["K"].round(6)
    over = dc.merge(dp, on="K_r", how="inner", suffixes=("_c", "_p"))
    if over.empty:
        return np.nan

    K = over["K_c"].to_numpy(float)
    resid = over["C"].to_numpy(float) - over["P"].to_numpy(float) - B * (K_star - K)
    resid = resid[np.isfinite(resid)]
    if resid.size == 0:
        return np.nan
    return float(np.median(np.abs(resid)))


# =======================
# Load data + method outputs
# =======================
caps, floors, swaps = load_data(DATA_PATH)
caps["date"] = caps["date"].astype(str)
floors["date"] = floors["date"].astype(str)
swaps["date"] = swaps["date"].astype(str)

df_bl   = pd.read_csv(out_dir / "moments_main.csv")
df_bkm  = pd.read_csv(out_dir / "moments_direct_bkm.csv")
df_logn = pd.read_csv(out_dir / "moments_parametric_lognormal.csv")
df_norm = pd.read_csv(out_dir / "moments_parametric_normal.csv")

for d in (df_bl, df_bkm, df_logn, df_norm):
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d.dropna(subset=["date"], inplace=True)

# Filter AREA
df_bl   = df_bl[df_bl["area"] == AREA].copy()
df_bkm  = df_bkm[df_bkm["area"] == AREA].copy()
df_logn = df_logn[df_logn["area"] == AREA].copy()
df_norm = df_norm[df_norm["area"] == AREA].copy()

# Ensure n exists
for d in (df_bl, df_bkm, df_logn, df_norm):
    if "n" not in d.columns:
        d["n"] = 1

# Intersection sample
key = ["date", "area", "n"]
df = df_bkm[key + ["mean", "variance", "Kmin_used_put", "Kmax_used_call", "n_puts", "n_calls"]].rename(
    columns={"mean": "mean_bkm", "variance": "var_bkm"}
)
df = df.merge(
    df_bl[key + ["mean", "variance", "K_star", "Kmin", "Kmax", "n_strikes"]].rename(
        columns={"mean": "mean_bl", "variance": "var_bl"}
    ),
    on=key, how="inner"
)
df = df.merge(
    df_logn[key + ["mean", "variance", "sigma_hat", "sigma_max", "rmse_price", "Kmin_put", "Kmax_call"]].rename(
        columns={"mean": "mean_logn", "variance": "var_logn"}
    ),
    on=key, how="inner"
)
df = df.merge(
    df_norm[key + ["mean", "variance"]].rename(columns={"mean": "mean_norm", "variance": "var_norm"}),
    on=key, how="inner"
).sort_values("date")

print(f"Intersection sample rows (AREA={AREA}): {len(df)}")
if df.empty:
    raise RuntimeError("Intersection sample is empty — cannot run diagnostics.")

# =======================
# Divergence metrics (variance)
# =======================
eps = 1e-18
df["ratio_bl_bkm"]   = (df["var_bl"]   + eps) / (df["var_bkm"] + eps)
df["ratio_logn_bkm"] = (df["var_logn"] + eps) / (df["var_bkm"] + eps)
df["ratio_norm_bkm"] = (df["var_norm"] + eps) / (df["var_bkm"] + eps)

df["logbps_bl_bkm"]   = 10000.0 * np.log(df["ratio_bl_bkm"])
df["logbps_logn_bkm"] = 10000.0 * np.log(df["ratio_logn_bkm"])
df["logbps_norm_bkm"] = 10000.0 * np.log(df["ratio_norm_bkm"])

# Choose “worst” dates by BL vs BKM (you can change to lognormal if preferred)
df["abs_logbps_bl_bkm"] = df["logbps_bl_bkm"].abs()
worst = df.sort_values("abs_logbps_bl_bkm", ascending=False).head(TOP_K).copy()

# =======================
# Diagnostics on worst dates
# =======================
rows = []
for _, r in worst.iterrows():
    date = r["date"].strftime("%Y-%m-%d")
    area = r["area"]

    # K* position within observed strike window (use BL Kmin/Kmax)
    K_star = float(r["K_star"])
    Kmin = float(r["Kmin"])
    Kmax = float(r["Kmax"])
    span = max(Kmax - Kmin, 1e-12)
    kstar_pos = (K_star - Kmin) / span

    # Parity residual
    med_parity_abs = parity_residual_on_overlap(caps, floors, swaps, date, area)

    # BL edge mass (rebuild BL pdf)
    edge_mass = np.nan
    curve = build_call_equivalent_curve(caps, floors, swaps, date, area)
    if curve is not None:
        K_all, C_all, B, K_star2 = curve
        grid = choose_grid(K_all, n=GRID_N)
        if grid is not None:
            spline = UnivariateSpline(K_all, C_all, s=0.0)
            pdf = implied_pdf_from_prices(spline, grid)
            if pdf is not None and np.isfinite(pdf).all():
                edge_mass = bl_edge_mass(grid, pdf, edge_n=EDGE_N)

    # Lognormal “at-bound” flag
    sigma_hat = float(r["sigma_hat"])
    sigma_max = float(r["sigma_max"])
    sigma_at_bound = (np.isfinite(sigma_hat) and np.isfinite(sigma_max) and sigma_max > 0 and sigma_hat >= 0.95 * sigma_max)

    rows.append({
        "date": date,
        "area": area,
        "var_bkm": float(r["var_bkm"]),
        "var_bl": float(r["var_bl"]),
        "var_logn": float(r["var_logn"]),
        "var_norm": float(r["var_norm"]),
        "ratio_bl_bkm": float(r["ratio_bl_bkm"]),
        "ratio_logn_bkm": float(r["ratio_logn_bkm"]),
        "ratio_norm_bkm": float(r["ratio_norm_bkm"]),
        "logbps_bl_bkm": float(r["logbps_bl_bkm"]),
        "logbps_logn_bkm": float(r["logbps_logn_bkm"]),
        "logbps_norm_bkm": float(r["logbps_norm_bkm"]),
        "K_star": K_star,
        "Kmin": Kmin,
        "Kmax": Kmax,
        "kstar_pos": float(kstar_pos),
        "n_strikes": int(r["n_strikes"]),
        "edge_mass_bl": float(edge_mass) if np.isfinite(edge_mass) else np.nan,
        "parity_med_abs": float(med_parity_abs) if np.isfinite(med_parity_abs) else np.nan,
        "Kmin_used_put_bkm": float(r["Kmin_used_put"]),
        "Kmax_used_call_bkm": float(r["Kmax_used_call"]),
        "n_puts_bkm": int(r["n_puts"]),
        "n_calls_bkm": int(r["n_calls"]),
        "Kmin_put_logn": float(r["Kmin_put"]),
        "Kmax_call_logn": float(r["Kmax_call"]),
        "rmse_price_logn": float(r["rmse_price"]),
        "sigma_hat_logn": sigma_hat,
        "sigma_max_logn": sigma_max,
        "sigma_at_bound_logn": bool(sigma_at_bound),
    })

diag = pd.DataFrame(rows).sort_values("logbps_bl_bkm", key=lambda s: s.abs(), ascending=False)
out_csv = out_dir / f"diagnostics_divergence_{AREA}.csv"
diag.to_csv(out_csv, index=False)
print("Saved diagnostics:", out_csv)

# =======================
# Plot 1: variance ratios vs BKM (time series)
# =======================
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df["date"], df["ratio_bl_bkm"], label="BL / BKM")
ax.plot(df["date"], df["ratio_logn_bkm"], label="Lognormal / BKM")
ax.plot(df["date"], df["ratio_norm_bkm"], label="Normal / BKM")
ax.axhline(1.0, linewidth=1)
ax.set_title(f"Variance ratios vs BKM ({AREA}) — intersection sample")
ax.set_ylabel("Variance ratio")
ax.grid(True, alpha=0.25)
ax.legend()
fig.tight_layout()
p1 = out_dir / f"variance_ratio_vs_bkm_{AREA}.png"
fig.savefig(p1, dpi=200)
print("Saved:", p1)
if SHOW_PLOTS:
    plt.show()
else:
    plt.close(fig)

# =======================
# Plot 2: edge mass vs variance gap (worst dates only)
# =======================
fig, ax = plt.subplots(figsize=(7, 4))
x = diag["edge_mass_bl"]
y = diag["logbps_bl_bkm"]
ax.scatter(x, y)
ax.axhline(0.0, linewidth=1)
ax.set_title(f"BL edge mass vs log-variance gap ({AREA}) — worst {TOP_K} dates")
ax.set_xlabel("BL edge mass (first+last grid slices)")
ax.set_ylabel("10000 * log(var_BL / var_BKM)")
ax.grid(True, alpha=0.25)
fig.tight_layout()
p2 = out_dir / f"variance_gap_vs_edge_mass_{AREA}.png"
fig.savefig(p2, dpi=200)
print("Saved:", p2)
if SHOW_PLOTS:
    plt.show()
else:
    plt.close(fig)

print("\nTop divergence dates (BL vs BKM) summary:")
print(diag[["date", "ratio_bl_bkm", "logbps_bl_bkm", "edge_mass_bl", "kstar_pos", "parity_med_abs", "sigma_at_bound_logn"]].head(10).to_string(index=False))
