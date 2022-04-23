# Assessing Amazon Rainforest Regrowth with GEDI and ICESat-2 Data

## Description
This repository contains code necessary to reproduce the results published in [Melenković et al. 2022](https://doi.org/10.1016/j.srs.2022.100051) 

It will be updated soon ...

## Usage
The workflow involves a calibration step and a regrowth assessment step.

#### Calibration of GEDI forest heights
- Calculate airborne LiDAR statistics per GEDI footprint: [GEDI_L2A_airborneLiDAR_footprintQuery.py](GEDI_L2A_airborneLiDAR_footprintQuery.py)
- Derive the GEDI calibration models and plot figures (Section 4.1 in the paper): [GEDI_L2A_calibrationModels.py](GEDI_L2A_calibrationModels.py)

#### Calibration of ICESat-2 forest heights

- Calculate airborne LiDAR statistics per ICESat-2 ATL08 segment: [ATL08_airborneLiDAR_segmentQuery.py](ATL08_airborneLiDAR_segmentQuery.py)
- Derive the ICESat-2 calibration models and plot figures (Section 4.2 in the paper) [ATL08_calibrationModels.py](ATL08_calibrationModels.py)

#### Regrowth Assessment

It will be updated soon ...

## Citation
If you find the code useful in your work, please consider citing the paper and data below:
- Milutin Milenković, Johannes Reiche, John Armston, Amy Neuenschwander, Wanda De Keersmaecker, Martin Herold, Jan Verbesselt, Assessing amazon rainforest regrowth with GEDI and ICESat-2 data, Science of Remote Sensing, 2022, 100051, ISSN 2666-0172, https://doi.org/10.1016/j.srs.2022.100051.
- Zenodo DOI of the data will be added soon ...



