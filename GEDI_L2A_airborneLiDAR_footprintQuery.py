# Author:       Milutin Milenkovic
# Copyright:
# Licence:
# ----------------------------------------------------------------------------------------------------------------------
# -- Short Description --
"""
This script calculates airborne LiDAR forest height statistics (90th percentile) per each GEDI footprint intersecting
the two calibration sites.
"""
# -------- Input --------
# (1) airborne LiDAR forest heights as 1m raster in the tiff format per each calibration site
# (2) GEDI shots with variables stored in the csv files
# -------- Output -------
# (1) a single geodataframe (geopandas) including GEDI shots from both calibration sites.
# The geo-data-frame contains GEDI variables as well as airborne LiDAR statistics (90th percentile)
# ----------------------------------------------------------------------------------------------------------------------

import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats

# ------------------
# CHM raster:
# ------------------
chm_file_Para = '/mnt/ssd/milutin/Para_upScaling/Validation_GEDI_ICESat2/ALS_CHMs/Para_all_strips_norZ_q99.tif'
chm_file_MG = '/mnt/ssd/milutin/Para_upScaling/Validation_GEDI_ICESat2/ALS_CHMs/MG_all_strips_norZ_q99.tif'
# -----------------------------
# gedi csv file with all data
# -----------------------------
# shots with sensitives from the algorithm setting group 2
gediL2A_file_Para = r'/mnt/ssd/milutin/Para_upScaling/Validation_GEDI_ICESat2/GEDI_Para_v002_L2A/output_sens_a2/gedi_L2A_Para_all_sens_a2.csv'
gediL2A_file_MG = r'/mnt/ssd/milutin/Para_upScaling/Validation_GEDI_ICESat2/GEDI_MG_v002_L2A/output_sens_a2/gedi_L2A_MG_all_sens_a2.csv'

# shots with the default sensitives:
#gediL2A_file_Para = r'/mnt/ssd/milutin/Para_upScaling/Validation_GEDI_ICESat2/GEDI_Para_v002_L2A_allTime/output/gedi_L2A_allTime_Para_all.csv'
#gediL2A_file_MG = r'/mnt/ssd/milutin/Para_upScaling/Validation_GEDI_ICESat2/GEDI_MG_v002_L2A_allTime/output/gedi_L2A_allTime_MG_all.csv'
# ##############################################
# data reading and processing
# ##############################################
# read the GEDI points
gedi_df_Para = pd.read_csv(gediL2A_file_Para)
gedi_df_MG = pd.read_csv(gediL2A_file_MG)
# ------------------------
# convert to geopandas df
# ------------------------
gedi_gdf_Para = gpd.GeoDataFrame(gedi_df_Para, geometry=gpd.points_from_xy(gedi_df_Para.Longitude, gedi_df_Para.Latitude))
gedi_gdf_Para.set_crs(epsg=4326, inplace=True)
#
gedi_gdf_MG = gpd.GeoDataFrame(gedi_df_MG, geometry=gpd.points_from_xy(gedi_df_MG.Longitude, gedi_df_MG.Latitude))
gedi_gdf_MG.set_crs(epsg=4326, inplace=True)
# ------------------------
# transform to the UTM 21S
# ------------------------
gedi_gdf_Para.to_crs(epsg=31981, inplace=True)
#
gedi_gdf_MG.to_crs(epsg=31981, inplace=True)
# ------------------------
# buffer all the points:
# ------------------------
gedi_gdf_Para['geometry'] = gedi_gdf_Para['geometry'].buffer(12.5)
#
gedi_gdf_MG['geometry'] = gedi_gdf_MG['geometry'].buffer(12.5)
# ------------------------------------
# calculate zonal statistics
# ------------------------------------
gedi_gdf_Para['als_h'] = pd.DataFrame(zonal_stats(vectors=gedi_gdf_Para['geometry'],
                                                         raster=chm_file_Para,
                                                         stats='percentile_90'))['percentile_90']
#
gedi_gdf_MG['als_h'] = pd.DataFrame(zonal_stats(vectors=gedi_gdf_MG['geometry'],
                                                         raster=chm_file_MG,
                                                         stats='percentile_90'))['percentile_90']


# #########################################
# pre-processing related to ALS
# #########################################
# ------------------------------------
# remove points with ALS height == None
# ------------------------------------
gedi_gdf_Para.dropna(subset=['als_h'], inplace=True)
#
gedi_gdf_MG.dropna(subset=['als_h'], inplace=True)
# ------------------------------------
# remove points with ALS height == 0
# ------------------------------------
gedi_gdf_Para = gedi_gdf_Para[gedi_gdf_Para['als_h'] != 0]
#
gedi_gdf_MG = gedi_gdf_MG[gedi_gdf_MG['als_h'] != 0]
# #########################################
# pre-processing related to GEDI
# #########################################
# ---------------------------------------------------------
# filter out points with sensitivity outside 0, 1 interval
# ---------------------------------------------------------
gedi_gdf_Para = gedi_gdf_Para[(gedi_gdf_Para['sensitivity'] >= 0) & (gedi_gdf_Para['sensitivity'] <= 1)]
#
gedi_gdf_MG = gedi_gdf_MG[(gedi_gdf_MG['sensitivity'] >= 0) & (gedi_gdf_MG['sensitivity'] <= 1)]
# #########################################
# merge to gdf-s
# #########################################
# add the site column
gedi_gdf_Para['Site'] = 'Para'
gedi_gdf_MG['Site'] = 'MG'
# reset index
gedi_gdf_Para.reset_index(inplace=True)
gedi_gdf_MG.reset_index(inplace=True)
# merge
gedi_gdf = gedi_gdf_Para.append(gedi_gdf_MG, ignore_index=True)
# #########################################
# save
# #########################################
#outFilePath = r'/mnt/ssd/milutin/Para_upScaling/Validation_GEDI_ICESat2/GEDI_Para_v002_L2A_allTime/output/gedi_L2A_allTime_gdf_Para_MG.json'
outFilePath = r'/mnt/ssd/milutin/Para_upScaling/Validation_GEDI_ICESat2/GEDI_Para_v002_L2A/output_sens_a2/gedi_L2A_gdf_Para_MG_sens_a2.json'
with open(outFilePath, 'w') as f:
    f.write(gedi_gdf.to_json())

