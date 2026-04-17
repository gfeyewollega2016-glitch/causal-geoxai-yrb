// ============================================================================
// TERRACE-AWARE RUSLE FOR CAUSAL TREATMENT PREPARATION
// Study area: Yellow River Basin (YRB)
// Resolution: 1 km
// Units: Soil erosion reduction (t .ha-1. yr-1)
//
// Data sources:
// - CHIRPS precipitation
// - ERA5-Land rainfall frequency
// - Landsat NDVI composites
// - OpenLandMap soil datasets
// - SRTM DEM
//
// Note:
// Users must provide:
// (1) ROI asset
// (2) Terrace fraction maps (per year)
//
// ============================================================================


// USER INPUT: define study region
var roi = ee.FeatureCollection('users/yourusername/YRB_boundary');
var region = roi.geometry();
Map.centerObject(region, 6);


var TARGET_SCALE = 1000;      // 1km resolution
var CRS = 'EPSG:4326';
var ETA = 0.5;                // terrace attenuation strength
var LS_MAX = 20;
var years = [2000, 2010, 2020];//add as needed 

// ============================================================================
// USER: REPLACE THESE ASSET PATHS WITH YOUR OWN
// ============================================================================
var terracePaths = {
  2000: 'users/yourusername/terrace_2000',   // ← REPLACE
  2010: 'users/yourusername/terrace_2010',   // ← REPLACE
  2020: 'users/yourusername/terrace_2020'    // ← REPLACE
  //------
};

// ============================================================================
// HELPERS
// ============================================================================

/**
 * Converts an image to 1km resolution using bilinear resampling.
 * @param {ee.Image} img - Input image
 * @param {string} method - Resampling method ('bilinear' or 'bicubic')
 * @returns {ee.Image} Reprojected image
 */
function to1km(img, method) {
  method = method || 'bilinear';
  return img
    .resample(method)
    .reproject({crs: CRS, scale: TARGET_SCALE})
    .clip(region);
}

/**
 * Converts categorical image to 1km resolution WITHOUT resampling.
 * Uses majority rule or nearest neighbor reprojection to preserve class integrity.
 * @param {ee.Image} img - Categorical input image
 * @returns {ee.Image} Reprojected image
 */
function to1kmCategorical(img) {
  return img
    .reproject({crs: CRS, scale: TARGET_SCALE})
    .clip(region);
}

/**
 * Loads terrace fractional coverage for a given year.
 * @param {number} year - Benchmark year
 * @returns {ee.Image} Terrace fraction (0-1)
 */
function loadTerraceFrac(year) {
  var path = terracePaths[year];
  if (!path) {
    print('Error: No terrace data for year ' + year);
    return ee.Image.constant(0).rename('terrace_frac');
  }
  return to1km(ee.Image(path).select(0).rename('terrace_frac'), 'bilinear').clamp(0, 1);
}

function yearStart(year) {
  return ee.Date.fromYMD(year, 1, 1);
}

function yearEnd(year) {
  return ee.Date.fromYMD(year + 1, 1, 1);
}

// ============================================================================
// STATIC TERRAIN + SOIL (constant across years)
// ============================================================================

var dem = to1km(ee.Image('USGS/SRTMGL1_003').rename('DEM'));
var slopeRad = ee.Terrain.slope(dem).multiply(Math.PI / 180);

var L = ee.Image.constant(1000);
var m = 0.4;
var n = 1.3;

var LSbase = L.divide(22.13).pow(m)
  .multiply(slopeRad.sin().divide(0.0896).pow(n))
  .rename('LS')
  .clamp(0.001, LS_MAX);

var soc = to1km(
  ee.Image('OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02')
    .select('b0')
    .divide(10)
    .rename('SOC_pct')
);

var texture = to1kmCategorical(
  ee.Image('OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02')
    .select('b0')
    .rename('texture_class')
);

var textureWeight = texture.expression(
  "tex == 10 || tex == 11 ? 0.45 : " +
  "tex == 9 ? 0.35 : " +
  "tex == 7 || tex == 8 ? 0.25 : " +
  "tex <= 4 ? 0.18 : 0.22",
  {tex: texture}
).rename('texture_weight');

var socStats = soc.reduceRegion({
  reducer: ee.Reducer.percentile([2, 98]),
  geometry: region,
  scale: TARGET_SCALE,
  bestEffort: true,
  maxPixels: 1e13
});

var socP2 = ee.Number(socStats.get('SOC_pct_p2'));
var socP98 = ee.Number(socStats.get('SOC_pct_p98'));

var socNorm = soc.subtract(socP2)
  .divide(socP98.subtract(socP2))
  .clamp(0, 1);

var K = textureWeight
  .multiply(ee.Image.constant(1).subtract(socNorm.multiply(0.5)))
  .rename('K_factor')
  .clamp(0.05, 0.6);

// ============================================================================
// CLIMATE FUNCTIONS (year-specific)
// ============================================================================

function annualPrecipCHIRPS(year) {
  return to1km(
    ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
      .filterBounds(region)
      .filterDate(yearStart(year), yearEnd(year))
      .sum()
      .rename('P_annual_mm')
  );
}

function annualRainDaysERA5(year) {
  return to1km(
    ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
      .filterBounds(region)
      .filterDate(yearStart(year), yearEnd(year))
      .select('total_precipitation_sum')
      .map(function(img) {
        return img.gt(0.01);
      })
      .sum()
      .rename('rain_days')
  );
}

function annualNDVI(year) {
  return to1km(
    ee.ImageCollection('LANDSAT/COMPOSITES/C02/T1_L2_ANNUAL_NDVI') // NOTE: If unavailable, users may compute NDVI from Landsat SR collections
      .filterBounds(region)
      .filterDate(yearStart(year), yearEnd(year))
      .select('NDVI')
      .mean()
      .rename('NDVI')
  );
}

// ============================================================================
// MAIN LOOP: PROCESS EACH BENCHMARK YEAR
// ============================================================================

var summaryList = [];
var visYear = 2020;
var visReduction;

years.forEach(function(year) {
  print('Processing year: ' + year);
  
  var terraceFrac = loadTerraceFrac(year);
  var precip = annualPrecipCHIRPS(year);
  var rainDays = annualRainDaysERA5(year);
  var ndvi = annualNDVI(year);

  // R factor (rainfall erosivity)
  //empirical rainfall erosivity approximation adapted for dryland_semi-arid regions
  var R = precip
    .multiply(rainDays.divide(365).add(0.5))
    .multiply(0.35)
    .rename('R_factor');

  // C factor (cover-management)
  // NDVI-based cover factor following exponential vegetation–erosion relationship
  var ndviClamped = ndvi.clamp(-0.99, 0.99);
  var C = ndviClamped
    .divide(ee.Image.constant(1).subtract(ndviClamped).add(1e-6))  // added epsilon
    .multiply(-2)
    .exp()
    .clamp(0, 1)
    .rename('C_factor')
    .where(ndvi.lte(0), 1);

  // Terrace modifies slope length (LS) and support practice (P)
  //Exponential decay models nonlinear reduction of slope length and runoff connectivity due to terracing
  var terraceEffect = terraceFrac.multiply(-ETA).exp();
  var LS_with = LSbase.multiply(terraceEffect);
  var P_with = terraceEffect.rename('P_factor');

  // Erosion with and without terracing
  var E_no = R.multiply(K).multiply(LSbase).multiply(C).rename('E_noTerr');
  var E_with = R.multiply(K).multiply(LS_with).multiply(C).multiply(P_with).rename('E_withTerr');

  // Erosion reduction (treatment variable)
  var reduction = E_no.subtract(E_with)
    .max(0)
    .rename('erosion_reduction');

  var validMask = dem.gt(0);
  reduction = reduction.updateMask(validMask).unmask(0);

  // Save for visualization
  if (year === visYear) {
    visReduction = reduction;
  }

  // Export raster
  Export.image.toDrive({
    image: reduction.toFloat(),
    description: 'Terrace_Erosion_Reduction_1km_' + year,
    folder: 'YRB_Erosion_1km',
    fileNamePrefix: 'YRB_terrace_erosion_reduction_1km_' + year,
    scale: TARGET_SCALE,
    region: region,
    crs: CRS,
    maxPixels: 1e13
  });

  // Summary statistics over terraced pixels only (fraction > 1%)
  var terrMask = terraceFrac.gt(0.01);
  var terrReduction = reduction.updateMask(terrMask);

  var countDict = terrReduction.reduceRegion({
    reducer: ee.Reducer.count(),
    geometry: region,
    scale: TARGET_SCALE,
    bestEffort: true,
    maxPixels: 1e13
  });

  var pixelCount = ee.Number(ee.Dictionary(countDict).values().get(0));

  function safeSingleStat(img, reducer) {
    var d = img.reduceRegion({
      reducer: reducer,
      geometry: region,
      scale: TARGET_SCALE,
      bestEffort: true,
      maxPixels: 1e13
    });
    return ee.Number(ee.Dictionary(d).values().get(0));
  }

  var stats = ee.Dictionary(ee.Algorithms.If(
    pixelCount.gt(0),
    ee.Dictionary({
      mean: safeSingleStat(terrReduction, ee.Reducer.mean()),
      stdDev: safeSingleStat(terrReduction, ee.Reducer.stdDev()),
      p50: safeSingleStat(terrReduction, ee.Reducer.percentile([50])),
      p90: safeSingleStat(terrReduction, ee.Reducer.percentile([90])),
      p95: safeSingleStat(terrReduction, ee.Reducer.percentile([95])),
      max: safeSingleStat(terrReduction, ee.Reducer.max())
    }),
    ee.Dictionary({
      mean: null,
      stdDev: null,
      p50: null,
      p90: null,
      p95: null,
      max: null
    })
  ));

  var feature = ee.Feature(null, {
    year: year,
    mean_reduction: stats.get('mean'),
    std_reduction: stats.get('stdDev'),
    median_reduction: stats.get('p50'),
    p90_reduction: stats.get('p90'),
    p95_reduction: stats.get('p95'),
    max_reduction: stats.get('max')
  });

  summaryList.push(feature);
  print('Year ' + year + ' reduction summary', feature);
});

// ============================================================================
// EXPORT SUMMARY TABLE
// ============================================================================

var summaryFC = ee.FeatureCollection(summaryList);
print('All-year terrace erosion reduction summary', summaryFC);

Export.table.toDrive({
  collection: summaryFC,
  description: 'Terrace_RUSLE_1km_CHIRPS_ERA5_Summary',
  folder: 'YRB_Erosion_1km',
  fileFormat: 'CSV'
});

// ============================================================================
// VISUALIZATION
// ============================================================================

var reductionVis = {
  min: 0,
  max: 50,
  palette: ['white', 'yellow', 'orange', 'red', 'darkred']
};

var terraceVis = {
  min: 0,
  max: 1,
  palette: ['white', 'cyan', 'blue']
};

Map.addLayer(loadTerraceFrac(visYear), terraceVis, 'Terrace fraction ' + visYear);
Map.addLayer(visReduction, reductionVis, 'Erosion reduction ' + visYear);
Map.addLayer(LSbase, {min: 0, max: 20, palette: ['white', 'green', 'brown']}, 'LS factor');

print('📘 Reproducibility note:');
print('- Ensure ROI and terrace datasets are provided');
print('- Outputs are in t ha^-1 yr^-1');
print('- Script designed for 1 km resolution analysis');



