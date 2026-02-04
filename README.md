# Ye_Bitner2025-temperature-traits-brachiopods
Reproducible R workflow for Ye &amp; Bitner (2025) on temperature–trait relationships in brachiopods

# Ye & Bitner (2025) — R code for temperature–trait analyses in living brachiopods

This repository contains the analysis workflow (R) supporting:

**Ye, F. & Bitner, M. A. (2025)**  
*Exploring the association between temperature and multiple ecomorphological traits of biocalcifiers (Brachiopoda)*  
Palaeogeography, Palaeoclimatology, Palaeoecology, 667, 112883.  
DOI: 10.1016/j.palaeo.2025.112883

## Repository structure
- `analysis/` scripts to reproduce analyses and figures
- `R/functions/` reusable helper functions
- `data/` data notes + (optional) small example data
- `output/` generated figures/tables

## Data availability
The raw occurrence/trait/environment datasets are obtained from:  
- Zenodo

## Reproducibility
1. Open R in the project root
2. Restore packages:
```r
renv::restore()
