# =====================================================
# TIME-VARYING DATA PREPARATION (per-year pre-treatment)
# For DML causal analysis
# Inputs:
#   - LEREI-X rasters (outcome + lag)
#   - Climate, soil, terrain, land cover, population, treatment
# Outputs:
#   - dml_data_{year}.npz bundles for causal inference
#   - per-year imputer/scaler objects
#   - panel CSV for diagnostics
# =====================================================

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import xy
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

# =====================================================
# USER CONFIGURATION
# =====================================================
DATA_DIR = "./data"                  # Set to your working directory
OUTPUT_DIR = "./results/causal_data" # DML bundles and metadata
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# REFERENCE GRID
# =====================================================
ref_path = os.path.join(DATA_DIR, "LEREI_Results", "LEREI_X_1990.tif")
with rasterio.open(ref_path) as src:
    ref_shape = src.shape
    ref_transform = src.transform
    ref_crs = src.crs
    ref_bounds = src.bounds
    ref_nodata = src.nodata

ref_meta = {
    "shape": ref_shape,
    "crs": ref_crs,
    "transform": ref_transform,
    "bounds": ref_bounds,
    "nodata": ref_nodata
}
joblib.dump(ref_meta, os.path.join(OUTPUT_DIR, "reference_metadata.pkl"))
print("Reference metadata saved.")

# =====================================================
# HELPER: load and align any raster to reference grid
# =====================================================
def load_and_align(path, band=1, required=True, resample=Resampling.bilinear):
    """
    Load a raster band and align it to the reference grid.
    Handles nodata explicitly and reprojects when needed.
    """
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"Missing file: {path}")
        return None

    with rasterio.open(path) as src:
        # Read directly if already on the same grid
        if src.shape == ref_shape and src.crs == ref_crs and src.transform == ref_transform:
            data = src.read(band).astype(np.float32)
            if src.nodata is not None:
                data = np.where(data == src.nodata, np.nan, data)
            return data

        # Otherwise reproject to reference grid
        data = np.full(ref_shape, np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, band),
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            dst_nodata=np.nan,
            resampling=resample
        )
        return data

def load_ndvi_pair(path):
    """
    Load NDVI mean and trend bands and align them if needed.
    Assumes band 1 = mean, band 2 = trend.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing NDVI file: {path}")

    with rasterio.open(path) as src:
        ndvi_mean = src.read(1).astype(np.float32)
        ndvi_trend = src.read(2).astype(np.float32)

        if src.nodata is not None:
            ndvi_mean = np.where(ndvi_mean == src.nodata, np.nan, ndvi_mean)
            ndvi_trend = np.where(ndvi_trend == src.nodata, np.nan, ndvi_trend)

        if src.shape == ref_shape and src.crs == ref_crs and src.transform == ref_transform:
            return ndvi_mean, ndvi_trend

        mean_aligned = np.full(ref_shape, np.nan, dtype=np.float32)
        trend_aligned = np.full(ref_shape, np.nan, dtype=np.float32)

        reproject(
            source=ndvi_mean,
            destination=mean_aligned,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear
        )
        reproject(
            source=ndvi_trend,
            destination=trend_aligned,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear
        )
        return mean_aligned, trend_aligned

def compute_lulc_trends(luc_start, luc_end, mask):
    """
    Return fractional land-cover change indicators for:
    Grass, Farm, Forest, Urban.
    """
    class_names = {1: "Grass", 2: "Farm", 3: "Forest", 7: "Urban"}
    trends = {}

    for val, name in class_names.items():
        start_frac = (luc_start == val).astype(np.float32)
        end_frac = (luc_end == val).astype(np.float32)
        change = (end_frac - start_frac) * mask.astype(np.float32)
        trends[f"Δ{name}"] = change

    return trends

# =====================================================
# STATIC RASTERS
# =====================================================
print("Loading static rasters...")
STATIC = {
    "DEM": os.path.join(DATA_DIR, "DEM.tif"),
    "SOC": os.path.join(DATA_DIR, "GEE_Exports", "SOC_original.tif"),
    "SoilType": os.path.join(DATA_DIR, "Soil_Data", "Soil_Type_Enhanced.tif")
}

static_data = {}
for name, path in STATIC.items():
    static_data[name] = load_and_align(
        path,
        required=True,
        resample=Resampling.nearest if name == "SoilType" else Resampling.bilinear
    )

dem = static_data["DEM"]
soc = static_data["SOC"]
soil_type = static_data["SoilType"]

# =====================================================
# YEAR-SPECIFIC CONFIGURATION
# Lag logic:
#   2000 uses 1990 lag
#   2010 uses 2000 lag
#   2020 uses 2010 lag
# =====================================================
YEARS = {
    2000: {
        "LEREI_outcome": os.path.join(DATA_DIR, "LEREI_Results", "LEREI_X_2000.tif"),
        "LEREI_lag":     os.path.join(DATA_DIR, "LEREI_Results", "LEREI_X_1990.tif"),
        "AI":            os.path.join(DATA_DIR, "GEE_ClimateExports", "mean_ai_1985_2000.tif"),
        "LUC_baseline":  os.path.join(DATA_DIR, "Imported", "lulc_1985.tif"),
        "LUC_start":     os.path.join(DATA_DIR, "Imported", "lulc_1985.tif"),
        "LUC_end":       os.path.join(DATA_DIR, "Imported", "lulc_1995.tif"),
        "NDVI":          os.path.join(DATA_DIR, "GEE_Exports", "ndvi_pre2000.tif"),
        "PopDens":       os.path.join(DATA_DIR, "GEE_Exports", "WorldPop_1990_1km.tif"),
        "EP":            os.path.join(DATA_DIR, "GEE_Exports", "EP_pretreated_z_1990.tif"),
        "treatment_raw": os.path.join(DATA_DIR, "GEE_Exports", "T_2000.tif")
    },
    2010: {
        "LEREI_outcome": os.path.join(DATA_DIR, "LEREI_Results", "LEREI_X_2010.tif"),
        "LEREI_lag":     os.path.join(DATA_DIR, "LEREI_Results", "LEREI_X_2000.tif"),
        "AI":            os.path.join(DATA_DIR, "GEE_ClimateExports", "mean_ai_2000_2010.tif"),
        "LUC_baseline":  os.path.join(DATA_DIR, "Imported", "lulc_2000.tif"),
        "LUC_start":     os.path.join(DATA_DIR, "Imported", "lulc_1995.tif"),
        "LUC_end":       os.path.join(DATA_DIR, "Imported", "lulc_2005.tif"),
        "NDVI":          os.path.join(DATA_DIR, "GEE_Exports", "ndvi_pre2010.tif"),
        "PopDens":       os.path.join(DATA_DIR, "GEE_Exports", "WorldPop_2000_1km.tif"),
        "EP":            os.path.join(DATA_DIR, "GEE_Exports", "EP_pretreated_z_2000.tif"),
        "treatment_raw": os.path.join(DATA_DIR, "GEE_Exports", "T_2010.tif")
    },
    2020: {
        "LEREI_outcome": os.path.join(DATA_DIR, "LEREI_Results", "LEREI_X_2020.tif"),
        "LEREI_lag":     os.path.join(DATA_DIR, "LEREI_Results", "LEREI_X_2010.tif"),
        "AI":            os.path.join(DATA_DIR, "GEE_ClimateExports", "mean_ai_2010_2020.tif"),
        "LUC_baseline":  os.path.join(DATA_DIR, "Imported", "lulc_2010.tif"),
        "LUC_start":     os.path.join(DATA_DIR, "Imported", "lulc_2005.tif"),
        "LUC_end":       os.path.join(DATA_DIR, "Imported", "lulc_2015.tif"),
        "NDVI":          os.path.join(DATA_DIR, "GEE_Exports", "ndvi_pre2020.tif"),
        "PopDens":       os.path.join(DATA_DIR, "GEE_Exports", "WorldPop_2010_1km.tif"),
        "EP":            os.path.join(DATA_DIR, "GEE_Exports", "EP_pretreated_z_2010.tif"),
        "treatment_raw": os.path.join(DATA_DIR, "GEE_Exports", "T_2020.tif")
    }
}

# =====================================================
# PROCESS EACH YEAR
# =====================================================
all_dfs = []
per_year_masks = {}

for year, paths in YEARS.items():
    print(f"\n{'='*50}\nProcessing year {year}\n{'='*50}")

    # Outcome and lag
    y = load_and_align(paths["LEREI_outcome"], required=True)
    y_lag = load_and_align(paths["LEREI_lag"], required=True)
    expected_lag_year = {2000: 1990, 2010: 2000, 2020: 2010}[year]
    print(f"  Outcome year: {year}, lag year: {expected_lag_year}")

    # Climate, cover, treatment
    ai = load_and_align(paths["AI"], required=True)
    luc_baseline = load_and_align(paths["LUC_baseline"], required=True, resample=Resampling.nearest)
    luc_start = load_and_align(paths["LUC_start"], required=True, resample=Resampling.nearest)
    luc_end = load_and_align(paths["LUC_end"], required=True, resample=Resampling.nearest)
    ndvi_mean, ndvi_trend = load_ndvi_pair(paths["NDVI"])
    pop = load_and_align(paths["PopDens"], required=True)
    ep = load_and_align(paths["EP"], required=True)
    treat_raw = load_and_align(paths["treatment_raw"], required=True)

    # Valid mask
    mask = (
        ~np.isnan(y) &
        ~np.isnan(y_lag) &
        ~np.isnan(ai) &
        ~np.isnan(luc_baseline) &
        ~np.isnan(luc_start) &
        ~np.isnan(luc_end) &
        ~np.isnan(ndvi_mean) &
        ~np.isnan(ndvi_trend) &
        ~np.isnan(pop) &
        ~np.isnan(ep) &
        ~np.isnan(treat_raw) &
        ~np.isnan(dem) &
        ~np.isnan(soc) &
        ~np.isnan(soil_type)
    )

    if mask.sum() == 0:
        print(f"  No valid pixels for {year}, skipping.")
        continue

    print(f"  Valid pixels: {mask.sum():,}")

    # Save mask
    per_year_masks[year] = mask
    np.save(os.path.join(OUTPUT_DIR, f"valid_mask_{year}.npy"), mask)

    # Coordinates
    rows, cols = np.where(mask)
    xs, ys = xy(ref_transform, rows, cols)
    xs = np.array(xs)
    ys = np.array(ys)

    # LULC trends
    lulc_trends = compute_lulc_trends(luc_start, luc_end, mask)

    # Year table
    df = pd.DataFrame({
        "x": xs,
        "y": ys,
        "year": year,
        "LEREI": y[mask],
        "LEREI_lag": y_lag[mask],
        "AI": ai[mask],
        "NDVI": ndvi_mean[mask],
        "NDVI_trend": ndvi_trend[mask],
        "PopDens": pop[mask],
        "EP": ep[mask],
        "treatment": treat_raw[mask],  # raw treatment, no normalization
        "DEM": dem[mask],
        "SOC": soc[mask],
        "SoilType": soil_type[mask].astype(np.int16)
    })

    for col, arr in lulc_trends.items():
        df[col] = arr[mask]

    # Land-cover dummies only (avoid redundancy with numeric baseline class)
    for val, name in {1: "Grassland", 2: "Farmland", 3: "Forests", 7: "Urban"}.items():
        df[f"LUC_{name}"] = (luc_baseline[mask] == val).astype(np.int8)

    all_dfs.append(df)

# =====================================================
# COMBINE AND SAVE FULL PANEL
# =====================================================
combined = pd.concat(all_dfs, ignore_index=True)
combined.to_csv(os.path.join(OUTPUT_DIR, "causal_data_timevarying_panel.csv"), index=False)
print(f"\nSaved combined panel: {os.path.join(OUTPUT_DIR, 'causal_data_timevarying_panel.csv')}")

# =====================================================
# DML-READY OUTPUTS PER YEAR
# =====================================================
print("\n" + "="*60)
print("Generating DML-ready bundles per year...")
print("="*60)

for year in YEARS:
    df_year = combined[combined["year"] == year].copy()
    if df_year.empty:
        continue

    Y = df_year["LEREI"].values.astype(np.float32)
    T = df_year["treatment"].values.astype(np.float32)
    x_coords = df_year["x"].values.astype(np.float32)
    y_coords = df_year["y"].values.astype(np.float32)

    # Exclude identifiers, outcome, treatment
    drop_cols = {"x", "y", "year", "LEREI", "treatment"}

    # Candidate covariates
    candidate_X_cols = [c for c in df_year.columns if c not in drop_cols]

    # Remove all-NaN and constant columns
    X_cols = []
    for c in candidate_X_cols:
        series = df_year[c]
        if series.isnull().all():
            continue
        if series.nunique(dropna=False) <= 1:
            continue
        X_cols.append(c)

    X = df_year[X_cols].values.astype(np.float32)

    # Impute and scale
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed).astype(np.float32)

    # Save NPZ bundle
    np.savez(
        os.path.join(OUTPUT_DIR, f"dml_data_{year}.npz"),
        Y=Y,
        T=T,
        X=X_scaled,
        x=x_coords,
        y=y_coords,
        feature_names=np.array(X_cols, dtype=object)
    )

    # Save preprocessing objects
    joblib.dump(imputer, os.path.join(OUTPUT_DIR, f"imputer_{year}.pkl"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, f"scaler_{year}.pkl"))

    # Save feature list
    with open(os.path.join(OUTPUT_DIR, f"features_{year}.txt"), "w") as f:
        f.write("\n".join(X_cols))

    print(f"Year {year}: {len(Y):,} observations, {X_scaled.shape[1]} covariates")
    print(f"  Saved dml_data_{year}.npz")

print(f"\n✅ DML-ready data saved to: {OUTPUT_DIR}")
print("Each dml_data_<year>.npz contains: Y, T, X, x, y, feature_names")
