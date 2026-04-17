# ============================================================
##DATA CONTINUOUS-TREATMENT
# SEVERITY-CONDITIONED SHAP MODERATION ANALYSIS
# ============================================================
# Data-path neutral: set BASE_DIR below to your working directory.
# Required input: dml_data_{year}.npz files (see preprocessing script)
# ============================================================

import os
import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# USER CONFIGURATION – EDIT THIS SECTION
# ============================================================
BASE_DIR = "./data"                     # Directory containing input .npz files
OUTPUT_DIR = "./results/shap_moderation" # Where outputs will be saved
YEARS = [2000, 2010, 2020]              # Benchmark years
RANDOM_STATE = 42
EPS = 1e-6
N_JOBS = -1

# Model hyperparameters
SURROGATE_N_ESTIMATORS = 150
SURROGATE_MAX_DEPTH = 12

# Severity conditioning
SEVERITY_FEATURE = "EP"
SEVERITY_PERCENTILE = 67
HIGH_T_PERCENTILE = 75
LOW_T_PERCENTILE = 25
TERRACE_MIN = 1.0

MAX_MATCHED_SAMPLES = 10000

# ============================================================
# Setup
# ============================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD DATA + MODEL
# ============================================================
def load_data_and_model(year):
    npz_path = os.path.join(BASE_DIR, f"dml_data_{year}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing file: {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)

    X = data["X"].astype(np.float32)
    T = data["T"].astype(np.float32)
    feature_names = data["feature_names"].tolist()

    model = joblib.load(os.path.join(BASE_DIR, f"CausalForestDML_{year}.pkl"))
    return X, T, feature_names, model


# ============================================================
# MATCHED SEVERITY PAIRS
# ============================================================
def build_full_matched_severity_pairs(
    X,
    T,
    feature_names,
    severity_feature=SEVERITY_FEATURE,
    terrace_min=TERRACE_MIN,
):
    """
    Full-data matched comparison:
    severe high-intensity vs low-intensity terraces
    within terraced pixels only.
    """
    sev_idx = feature_names.index(severity_feature)
    severity = X[:, sev_idx]

    severe_thr = np.percentile(severity, SEVERITY_PERCENTILE)
    severe_mask = severity >= severe_thr

    terraced_mask = T > terrace_min
    analysis_mask = severe_mask & terraced_mask

    T_analysis = T[analysis_mask]
    if len(T_analysis) == 0:
        raise ValueError("No severe terraced pixels found.")

    high_thr = np.percentile(T_analysis, HIGH_T_PERCENTILE)
    low_thr = np.percentile(T_analysis, LOW_T_PERCENTILE)

    high_mask = analysis_mask & (T >= high_thr)
    low_mask = analysis_mask & (T <= low_thr)

    high_idx = np.where(high_mask)[0]
    low_idx = np.where(low_mask)[0]

    if len(high_idx) == 0 or len(low_idx) == 0:
        raise ValueError("No valid severe high-vs-low terraced groups.")

    print(f"Severe threshold (EP): {severe_thr:.4f}")
    print(f"Terraced domain: T > {terrace_min}")
    print(f"Low terrace threshold: {low_thr:.4f}")
    print(f"High terrace threshold: {high_thr:.4f}")
    print(f"Low severe terraced: {len(low_idx):,}")
    print(f"High severe terraced: {len(high_idx):,}")

    scaler = StandardScaler()
    X_high = scaler.fit_transform(X[high_idx])
    X_low = scaler.transform(X[low_idx])

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(X_low)

    _, idx_match = nn.kneighbors(X_high)
    matched_low_idx = low_idx[idx_match.flatten()]

    print(f"Matched low-intensity severe terraces: {len(matched_low_idx):,}")
    print(f"Unique matched controls: {len(np.unique(matched_low_idx)):,}")

    return high_idx, matched_low_idx


# ============================================================
# SURROGATE SHAP (WITH R² AND RMSE)
# ============================================================
def compute_surrogate_shap(model, X_pair):
    """
    Explain DML CATE surface using RF surrogate SHAP.
    Returns shap_values, surrogate, r2, rmse.
    """
    cate_preds = model.effect(X_pair)

    surrogate = RandomForestRegressor(
        n_estimators=SURROGATE_N_ESTIMATORS,
        max_depth=SURROGATE_MAX_DEPTH,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

    surrogate.fit(X_pair, cate_preds)
    pred_sur = surrogate.predict(X_pair)
    r2 = r2_score(cate_preds, pred_sur)
    rmse = np.sqrt(mean_squared_error(cate_preds, pred_sur))

    explainer = shap.TreeExplainer(
        surrogate,
        data=X_pair,
        feature_perturbation="interventional",
    )

    shap_values = explainer.shap_values(X_pair)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)
    return shap_values, surrogate, r2, rmse


# ============================================================
# PLOT (NO ERROR BARS)
# ============================================================
def plot_bar(diff_df, year, title_suffix=""):
    plot_df = diff_df.sort_values("abs", ascending=False).head(10).copy()
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)
    plot_df["label"] = plot_df["feature"]

    y = np.arange(len(plot_df))
    values = plot_df["Δϕ_norm"].values
    colors = ["#2ca02c" if x > 0 else "#d62728" for x in values]

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.barh(y, values, color=colors, edgecolor="none", alpha=0.9)
    ax.axvline(0, linestyle="--", color="gray", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"].values)
    ax.set_title(f"Severity-conditioned SHAP moderation ({year}{title_suffix})")
    ax.set_xlabel("Normalized Δϕ (High vs Low Terrace Intensity)")
    ax.set_ylabel("Feature")

    plt.tight_layout()
    save_name = f"severity_conditioned_bar_{year}{str(title_suffix).replace(' ', '_').replace('-', '')}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, save_name), dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# MAIN LOOP
# ============================================================
all_results = []
all_diff_dfs = []
metrics_list = []

for year in YEARS:
    print(f"\n{'='*70}")
    print(f"YEAR {year}")
    print(f"{'='*70}")

    X, T, feature_names, model = load_data_and_model(year)
    print(f"T range: {T.min():.4f} to {T.max():.4f}")

    # 1) Full-data matching
    high_idx, low_idx = build_full_matched_severity_pairs(X, T, feature_names)

    # 2) Build matched sample
    pair_idx = np.concatenate([high_idx, low_idx])
    if len(pair_idx) > MAX_MATCHED_SAMPLES:
        rng = np.random.RandomState(RANDOM_STATE)
        pair_idx = rng.choice(pair_idx, MAX_MATCHED_SAMPLES, replace=False)

    X_pair = X[pair_idx]
    T_pair = T[pair_idx]

    # 3) SHAP on matched sample (returns r2 and rmse)
    shap_values, surrogate, r2, rmse = compute_surrogate_shap(model, X_pair)

    # Store surrogate metrics
    metrics_list.append({"year": year, "surrogate_r2": r2, "surrogate_rmse": rmse})
    print(f"  Surrogate RF R²: {r2:.4f}, RMSE: {rmse:.4f}")

    # Use consistent thresholds from full severe terraced domain
    sev_idx = feature_names.index(SEVERITY_FEATURE)
    severity = X[:, sev_idx]
    severe_thr = np.percentile(severity, SEVERITY_PERCENTILE)
    severe_mask = severity >= severe_thr
    terraced_mask = T > TERRACE_MIN
    analysis_mask = severe_mask & terraced_mask
    T_analysis = T[analysis_mask]
    high_thr = np.percentile(T_analysis, HIGH_T_PERCENTILE)
    low_thr = np.percentile(T_analysis, LOW_T_PERCENTILE)

    mask_high_in_pair = T_pair >= high_thr
    mask_low_in_pair = T_pair <= low_thr

    # 4) Point estimates only (no bootstrap)
    high_vals = shap_values[mask_high_in_pair]
    low_vals = shap_values[mask_low_in_pair]
    mean_abs_shap = np.abs(shap_values).mean(axis=0) + EPS

    delta_phi = high_vals.mean(axis=0) - low_vals.mean(axis=0)
    delta_phi_norm = delta_phi / mean_abs_shap

    diff_df = pd.DataFrame({
        "feature": feature_names,
        "Δϕ": delta_phi,
        "Δϕ_norm": delta_phi_norm,
        "abs": np.abs(delta_phi_norm),
    }).sort_values("abs", ascending=False)

    diff_df.to_csv(os.path.join(OUTPUT_DIR, f"severity_conditioned_delta_phi_{year}.csv"), index=False)
    plot_bar(diff_df, year)

    all_results.append({
        "year": year,
        "surrogate_r2": r2,
        "surrogate_rmse": rmse,
        "top_feature": diff_df.iloc[0]["feature"],
        "top_delta_norm": diff_df.iloc[0]["Δϕ_norm"],
    })
    all_diff_dfs.append(diff_df)

    gc.collect()


# ============================================================
# SAVE METRICS CSV (R² and RMSE per year)
# ============================================================
metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "surrogate_metrics.csv"), index=False)
print(f"\n✅ Saved surrogate metrics to {OUTPUT_DIR}/surrogate_metrics.csv")
print(metrics_df.to_string(index=False))

# ============================================================
# MEAN ANALYSIS ACROSS YEARS
# ============================================================
if len(all_diff_dfs) > 1:
    print(f"\n{'='*70}")
    print(f"MEAN ANALYSIS ACROSS YEARS {YEARS}")
    print(f"{'='*70}")

    combined_diff_dfs = pd.concat(all_diff_dfs, ignore_index=True)
    averaged_diff_df = combined_diff_dfs.groupby("feature").agg({
        "Δϕ": "mean",
        "Δϕ_norm": "mean",
        "abs": "mean",
    }).reset_index()

    # Exclude baseline if present
    averaged_diff_df = averaged_diff_df[averaged_diff_df["feature"] != "LUC_baseline"]
    averaged_diff_df = averaged_diff_df.sort_values("abs", ascending=False)

    averaged_diff_df.to_csv(
        os.path.join(OUTPUT_DIR, f"severity_conditioned_delta_phi_averaged_{'_'.join(map(str, YEARS))}.csv"),
        index=False,
    )

    plot_bar(
        averaged_diff_df,
        "Average",
        title_suffix=f" ({YEARS[0]}-{YEARS[-1]})\nMean across years",
    )

    print(f"\n✅ Saved averaged moderation results to {OUTPUT_DIR}/severity_conditioned_delta_phi_averaged_{'_'.join(map(str, YEARS))}.csv")


# ============================================================
# FINAL SUMMARY
# ============================================================
summary_df = pd.DataFrame(all_results)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary_results.csv"), index=False)

print("\nFINAL SUMMARY")
print(summary_df)
print(f"\nSaved to: {OUTPUT_DIR}")
