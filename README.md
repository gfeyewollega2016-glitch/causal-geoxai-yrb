# causal-geoxai-yrb
Code and Colab notebooks: Causal GeoXAI framework to assess terracing impacts on landscape resilience
# Causal GeoXAI: Terracing and Landscape Resilience

## Overview
This repository contains the complete code and workflows for the manuscript:

**Does terracing build landscape resilience? A causal GeoXAI assessment using Earth observations in China’s Yellow River Basin**  
*Submitted to Remote Sensing of Environment*

## Repository Structure
-Code: Colab notebooks and Python scripts for data preprocessing, LEREI-X construction, causal inference (Debiased ML), and conditioned SHAP analysis
#✨Part1
    ✔01_data_loading_and_preprocess.ipynb - data loading, reprojection, sampling, spatial block creation
    ✔02_ML_traina&_evaluation.ipynb - Training and Spatial cross-validation 
    ✔03_SHAP_nalaysis and visualization.ipynb - Models explainability
#✨Part2
    ✔01_Causal_Effect_estimation - 
    ✔
-data: Instructions to obtain the geospatial data (most raw data are from public sources)

## Requirements
- Python 3.9+
- Install dependencies: *pip install -r requirements.txt*

## How to Reproduce
1. Download the harmonised dataset from the provided public sources and apply all data preprocessing and preparation within the desired ROI
2. Run *Colab notebook.ipynb /python code/run_all.py* (or execute scripts in order: 01 → 02 → 03 → 04==).
3. Results will be written to "outputs".

## Citation
If you use this code, please cite the associated paper (once published) and this repository.

## License
MIT (see LICENSE file)
