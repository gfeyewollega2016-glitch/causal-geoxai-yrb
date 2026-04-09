// ============================================================================
// TERRACE-AWARE RUSLE FOR CAUSAL TREATMENT PREPARATION 
// Yellow River Basin | Years: 2000, 2010, 2020
// Climate harmonized with CHIRPS + ERA5-Land
// Output: terrace-induced erosion reduction (continuous treatment)
// ============================================================================

var region = (roi.geometry !== undefined) ? roi.geometry() : roi;
Map.centerObject(region, 6);

var TARGET_SCALE = 1000;   // unified causal-analysis grid
var CRS = 'EPSG:4326';
var ETA = 0.5;
var LS_MAX = 20;
var years = [2000, 2010, 2020];

var terracePaths = {
  2000: '2000_terraces_data',
  2010: '2010_terraces_data',
  2020: '2020_terraces_data'
};

// ----------------------------------------------------------------------------
// HELPERS
// ----------------------------------------------------------------------------
function to1km(img, method) {
  method = method || 'bilinear';
  return img
    .resample(method)
    .reproject({crs: CRS, scale: TARGET_SCALE})
    .clip(region);
}

function loadTerraceFrac(year) {
  return to1km(
    ee.Image(terracePaths[year])
      .select(0)
      .rename('terrace_frac'),
    'bilinear'
  ).clamp(0, 1);
}

// ----------------------------------------------------------------------------
// STATIC TERRAIN + SOIL (1 km)
// ----------------------------------------------------------------------------
var dem = to1km(ee.Image('USGS/SRTMGL1_003').rename('DEM'));
var slopeRad = ee.Terrain.slope(dem).multiply(Math.PI / 180);

var L = ee.Image.constant(1000); // aligned with 1 km grid
var m = 0.4;
var n = 1.3;

var LSbase = L.divide(22.13).pow(m)
  .multiply(slopeRad.sin().divide(0.0896).pow(n))
  .rename('LS')
  .clamp(0.001, LS_MAX);

// SOC + texture-based K factor
var soc = to1km(
  ee.Image('OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02')
    .select('b0')
    .divide(10)
    .rename('SOC_pct')
);

var texture = to1km(
  ee.Image('OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02')
    .select('b0')
    .rename('texture_class'),
  'nearest'
);

var textureWeight = texture.expression(
  "tex==10 || tex==11 ? 0.45 : " +
  "tex==9 ? 0.35 : " +
  "tex==7 || tex==8 ? 0.25 : " +
  "tex<=4 ? 0.18 : 0.22",
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

// ----------------------------------------------------------------------------
// CLIMATE (CHIRPS + ERA5) ALL HARMONIZED TO 1 km
// ----------------------------------------------------------------------------
function annualPrecipCHIRPS(year) {
  var pr = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
    .filterBounds(region)
    .filterDate(year + '-01-01', year + '-12-31')
    .sum()
    .rename('P_annual_mm');

  return to1km(pr, 'bilinear');
}

function annualRainIntensityERA5(year) {
  var rainDays = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
    .filterBounds(region)
    .filterDate(year + '-01-01', year + '-12-31')
    .select('total_precipitation_sum')
    .map(function(img) {
      return img.gt(0.01);
    })
    .sum()
    .rename('rain_days');

  return to1km(rainDays, 'bilinear');
}

function annualNDVI(year) {
  var ndvi = ee.ImageCollection('LANDSAT/COMPOSITES/C02/T1_L2_ANNUAL_NDVI')
    .filterBounds(region)
    .filterDate(year + '-01-01', year + '-12-31')
    .select('NDVI')
    .mean()
    .rename('NDVI');

  return to1km(ndvi, 'bilinear');
}

// ----------------------------------------------------------------------------
// MAIN LOOP
// ----------------------------------------------------------------------------
var summaryList = [];

years.forEach(function(year) {
  var terraceFrac = loadTerraceFrac(year);
  var precip = annualPrecipCHIRPS(year);
  var rainDays = annualRainIntensityERA5(year);
  var ndvi = annualNDVI(year);

  // improved rainfall erosivity proxy
  var R = precip
    .multiply(rainDays.divide(365).add(0.5))
    .multiply(0.35)
    .rename('R_factor');

  // NDVI-derived C factor
  var ndviClamped = ndvi.clamp(-0.99, 0.99);
  var C = ndviClamped
    .divide(ee.Image.constant(1).subtract(ndviClamped))
    .multiply(-2)
    .exp()
    .clamp(0, 1)
    .rename('C_factor')
    .where(ndvi.lte(0), 1);

  // terrace effect
  var terraceEffect = terraceFrac.multiply(-ETA).exp();
  var LS_with = LSbase.multiply(terraceEffect);
  var P_with = terraceEffect;

  var E_no = R.multiply(K).multiply(LSbase).multiply(C).rename('E_noTerr');
  var E_with = R.multiply(K).multiply(LS_with).multiply(C).multiply(P_with).rename('E_withTerr');

  var reduction = E_no.subtract(E_with)
    .max(0)
    .rename('erosion_reduction');

  var validMask = dem.gt(0).and(terraceFrac.gte(0));
  reduction = reduction.updateMask(validMask).unmask(0);

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

  var terrMask = terraceFrac.gt(0.01);
  var stats = reduction.updateMask(terrMask).reduceRegion({
    reducer: ee.Reducer.mean()
      .combine(ee.Reducer.stdDev(), '', true)
      .combine(ee.Reducer.percentile([50, 90, 95]), '', true),
    geometry: region,
    scale: TARGET_SCALE,
    bestEffort: true,
    maxPixels: 1e13
  });

  summaryList.push(ee.Feature(null, {
    year: year,
    mean_reduction: stats.get('mean'),
    std_reduction: stats.get('stdDev'),
    median_reduction: stats.get('p50'),
    p90_reduction: stats.get('p90'),
    p95_reduction: stats.get('p95')
  }));
});

Export.table.toDrive({
  collection: ee.FeatureCollection(summaryList),
  description: 'Terrace_RUSLE_1km_CHIRPS_ERA5_Summary',
  folder: 'YRB_Erosion_1km',
  fileFormat: 'CSV'
});

print('✅ 1 km terrace-aware RUSLE exports started.');
print('All predictors, climate, and treatment layers are harmonized to 1 km.');
