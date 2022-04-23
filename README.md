# Assessing Amazon Rainforest Regrowth with GEDI and ICESat-2 Data

This repository contains code necessary to reproduce the results published in [Milenković et al. 2022](https://doi.org/10.1016/j.srs.2022.100051) 

## Overview

#### Calibration of GEDI forest heights
- Calculate airborne LiDAR statistics per GEDI footprint: 
    - [GEDI_L2A_airborneLiDAR_footprintQuery.py](GEDI_L2A_airborneLiDAR_footprintQuery.py)
- Derive the GEDI calibration models and plot figures (Section 4.1): 
    - [GEDI_L2A_calibrationModels.py](GEDI_L2A_calibrationModels.py)

#### Calibration of ICESat-2 forest heights

- Calculate airborne LiDAR statistics per ICESat-2 ATL08 segment: 
    - [ATL08_airborneLiDAR_segmentQuery.py](ATL08_airborneLiDAR_segmentQuery.py)
- Derive the ICESat-2 calibration models and plot figures (Section 4.2): 
    - [ATL08_calibrationModels.py](ATL08_calibrationModels.py)

#### Forest Regrowth Assessment
- GEDI linear models fitted in 20-year regrowth period (Section 4.3.1): 
    - [GEDI_L2A_modeling_20years.py](GEDI_L2A_modeling_20years.py)
    - [GEDI_L2A_directAnalysis.py](GEDI_L2A_directAnalysis.py)
- ICESat-2 linear models fitted in 20-year regrowth period  (Section 4.3.2):
    - [ATL08_directAnalysis.py](ATL08_directAnalysis.py)
- GEDI and ICESat-2 non-linear models fitted in 33-year regrowth period (Section 4.5): 
    - [GEDI_ATL08_modeling_33years.py](GEDI_ATL08_modeling_33years.py)
    
#### Other Scripts

- Assign the forest age class and border pixel flag to GEDI shots and ATL08 segments:
    - [GEDI_L2A_forAge_pointQuery.py](GEDI_L2A_forAge_pointQuery.py)
    - [ATL08_forAge_pointQuery.py](ATL08_forAge_pointQuery.py)
- Figure with the distributions of calibrated GEDI ALL and ICESat-2 ALL forest heights (Section 4.3):
    - [GEDI_ATL08_regrowthFigure_33years.py](GEDI_ATL08_regrowthFigure_33years.py)
    
 
## Usage 
It will be updated soon ...

## Citation
If you find the code useful in your work, please consider citing the paper and data below:
- Milutin Milenković, Johannes Reiche, John Armston, Amy Neuenschwander, Wanda De Keersmaecker, Martin Herold, Jan Verbesselt, Assessing amazon rainforest regrowth with GEDI and ICESat-2 data, Science of Remote Sensing, 2022, 100051, ISSN 2666-0172, https://doi.org/10.1016/j.srs.2022.100051.
- Zenodo DOI of the data will be added soon ...



