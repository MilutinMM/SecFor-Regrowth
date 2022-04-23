# Author:       Milutin Milenkovic
# Copyright:
# Licence:
# ----------------------------------------------------------------------------------------------------------------------
# -- Short Description --
"""
This script assigns the forest age class and border pixel flag to GEDI shots.
"""
# -------- Input --------
# (1) GEDI shots over secondary forest in the Rondônia state, Brazil.
# (2) a forest age raster: Silva Junior et al. 2020, https://doi.org/10.6084/m9.figshare.12622025
# (3) a black-and-withe version of the forest age raster with border pixels removed
# -------- Output -------
# (1) The input GEDI shots with forest age attribute and border pixel flag
# ----------------------------------------------------------------------------------------------------------------------

import geopandas as gpd
import pandas as pd
import numpy as np
from rasterstats import point_query

# ------------------
# Forest Age rasters:
# ------------------
forestAge_file = r'/mnt/ssd/milutin/Para_upScaling/StandAgeMapBiomas/svbr-rondonia-2018.tif'
#
out_eroded_forestAge_file = r'/mnt/ssd/milutin/Para_upScaling/StandAgeMapBiomas/svbr-rondonia-2018_bw_eroded.tif'
# -----------------------------
# gedi csv file with all data
# -----------------------------
gediL2A_file = r'/mnt/raid/milutin/upScaling/Rondonia/GEDI/Rondonia_L2A_v002/output/gedi_L2A_Rondonia_all_sens_a2.csv'

# ##############################################
# data reading and processing
# ##############################################
# read the GEDI points
gedi_df = pd.read_csv(gediL2A_file)
# ------------------------
# convert to geopandas df
# ------------------------
gedi_gdf = gpd.GeoDataFrame(gedi_df, geometry=gpd.points_from_xy(gedi_df.Longitude, gedi_df.Latitude))
gedi_gdf.set_crs(epsg=4326, inplace=True)
# ------------------------------------
# calculate zonal statistics
# ------------------------------------
forestAge_list = point_query(vectors=gedi_gdf['geometry'], raster=forestAge_file, interpolate='nearest')
gedi_gdf['forestAge'] = np.asarray(forestAge_list)
# ------------------------------------
# remove points with stand age == 0
# ------------------------------------
gedi_gdf = gedi_gdf[gedi_gdf['forestAge'] != 0]
# ------------------------------------
# remove points with stand age == None
# ------------------------------------
gedi_gdf.dropna(subset=['forestAge'], inplace=True)
# ----------------------------------------------
# get pixel value from the eroded bw forest age
# ----------------------------------------------
forestAge_list = point_query(vectors=gedi_gdf['geometry'], raster=out_eroded_forestAge_file, interpolate='nearest')
gedi_gdf['eroded_age'] = np.asarray(forestAge_list)
# ------------------------------------
# save:
# ------------------------------------
outFilePath = r'/mnt/raid/milutin/upScaling/Rondonia/GEDI/Rondonia_L2A_v002/output/directAnalysis/gedi_L2A_gdf_sens_a2.json'
with open(outFilePath, 'w') as f:
    f.write(gedi_gdf.to_json())