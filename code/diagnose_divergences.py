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
    raw = np.asarray(second, dtype=float)
    raw[~np.isfinite(raw)] = 0.0

    neg_mass = float(np.trapz(np.maximum(-raw, 0.0), grid))
    pos_mass = float(np.trapz(np.maximum(raw, 0.0), grid))
    neg_share = (neg_mass / max(neg_mass + pos_mass, 1e-18)) if np.isfinite(neg_mass + pos_mass) else np.nan

    pdf = np.maximum(raw, 0.0)
    area = float(np.trapz(pdf, grid))
    if (not np.isfinite(area)) or area <= 0.0:
        return None, np.nan

    return pdf / area, neg_share



def bl_edge_mass(grid, pdf, edge_n=10):
    grid = np.asarray(grid, dtype=float)
    pdf = np.asarray(pdf, dtype=float)
    m = float(np.trapz(pdf[:edge_n], grid[:edge_n]) + np.trapz(pdf[-edge_n:], grid[-edge_n:]))
    return m

def moments_from_pdf(grid, pdf):
    """Compute mean and variance from a pdf defined on grid (assumed normalized)."""
    grid = np.asarray(grid, float)
    pdf = np.asarray(pdf, float)
    m = float(np.trapz(grid * pdf, grid))
    v = float(np.trapz(((grid - m) ** 2) * pdf, grid))
    return m, v


def restrict_pdf_to_interval(grid, pdf, a, b):
    """
    Restrict a pdf to [a,b] and renormalize.
    Returns (grid_sub, pdf_sub) or (None, None) if invalid.
    """
    grid = np.asarray(grid, float)
    pdf = np.asarray(pdf, float)

    if not (np.isfinite(a) and np.isfinite(b) and b > a):
        return None, None

    mask = (grid >= a) & (grid <= b)
    if mask.sum() < 10:
        return None, None

    g2 = grid[mask]
    p2 = pdf[mask].copy()

    area = float(np.trapz(p2, g2))
    if (not np.isfinite(area)) or area <= 0:
        return None, None

    return g2, p2 / area


def curve_sanity_metrics(K, C):
    """
    Basic no-arbitrage shape checks for call prices as function of strike:
      - monotone decreasing: dC/dK <= 0
      - convex: second differences >= 0
    Returns a dict with counts and magnitudes.
    """
    K = np.asarray(K, float)
    C = np.asarray(C, float)

    # first differences
    dK = np.diff(K)
    dC = np.diff(C)
    slope = dC / np.maximum(dK, 1e-12)

    # monotonicity violations: slope > 0
    mono_viol = slope[slope > 0]
    mono_count = int(mono_viol.size)
    mono_max = float(mono_viol.max()) if mono_count else 0.0

    # convexity via second differences on C(K)
    # Use discrete second derivative: (C[i+1] - 2C[i] + C[i-1]) / (avg dK)^2
    ddC = C[2:] - 2 * C[1:-1] + C[:-2]
    convex_viol = ddC[ddC < 0]
    convex_count = int(convex_viol.size)
    convex_min = float(convex_viol.min()) if convex_count else 0.0  # most negative

    return {
        "mono_viol_count": mono_count,
        "mono_viol_max_slope": mono_max,
        "convex_viol_count": convex_count,
        "convex_viol_min_ddC": convex_min,
    }


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
            pdf = None
            neg_share = np.nan
            curve_metrics = {}

            if curve is not None:
                K_all, C_all, B, K_star2 = curve

                # shape sanity
                curve_metrics = curve_sanity_metrics(K_all, C_all)

                grid = choose_grid(K_all, n=GRID_N)
                if grid is not None:
                    spline = UnivariateSpline(K_all, C_all, s=0.0)
                    pdf, neg_share = implied_pdf_from_prices(spline, grid)
                    if pdf is not None and np.isfinite(pdf).all():
                        edge_mass = bl_edge_mass(grid, pdf, edge_n=EDGE_N)

                        # Common-support BL variance: restrict BL pdf to BKM-used support
                        a = float(r["Kmin_used_put"])
                        b = float(r["Kmax_used_call"])
                        g2, p2 = restrict_pdf_to_interval(grid, pdf, a, b)
                        if g2 is not None:
                            _, var_bl_common = moments_from_pdf(g2, p2)
                        else:
                            var_bl_common = np.nan

                        # Also compute BL variance on its own observed strike range (sanity)
                        g3, p3 = restrict_pdf_to_interval(grid, pdf, float(r["Kmin"]), float(r["Kmax"]))
                        if g3 is not None:
                            _, var_bl_own = moments_from_pdf(g3, p3)
                        else:
                            var_bl_own = np.nan
                    else:
                        var_bl_common = np.nan
                        var_bl_own = np.nan
            else:
                var_bl_common = np.nan
                var_bl_own = np.nan

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
        "neg_share_bl": float(neg_share) if np.isfinite(neg_share) else np.nan,
        "var_bl_common_support": float(var_bl_common) if np.isfinite(var_bl_common) else np.nan,
        "var_bl_own_support": float(var_bl_own) if np.isfinite(var_bl_own) else np.nan,
        "mono_viol_count": int(curve_metrics.get("mono_viol_count", 0)),
        "mono_viol_max_slope": float(curve_metrics.get("mono_viol_max_slope", 0.0)),
        "convex_viol_count": int(curve_metrics.get("convex_viol_count", 0)),
        "convex_viol_min_ddC": float(curve_metrics.get("convex_viol_min_ddC", 0.0)),
    })

diag = pd.DataFrame(rows).sort_values("logbps_bl_bkm", key=lambda s: s.abs(), ascending=False)
out_csv = out_dir / f"diagnostics_divergence_{AREA}.csv"
diag.to_csv(out_csv, index=False)
print("Saved diagnostics:", out_csv)

if "var_bl_common_support" in diag.columns:
    fig, ax = plt.subplots(figsize=(7, 4))

    x = (diag["var_bl_common_support"] + eps) / (diag["var_bkm"] + eps)
    y = (diag["var_bl"] + eps) / (diag["var_bkm"] + eps)

    ax.scatter(x, y)
    ax.plot([x.min(), x.max()], [x.min(), x.max()])  # 45-degree line
    ax.set_title(f"BL/BKM variance ratio: full vs common-support ({AREA}) — worst {TOP_K}")
    ax.set_xlabel("Common-support BL var / BKM var")
    ax.set_ylabel("Full BL var / BKM var")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    p3 = out_dir / f"variance_ratio_full_vs_common_{AREA}.png"
    fig.savefig(p3, dpi=200)
    print("Saved:", p3)

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


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
