# Author:       Milutin Milenkovic
# Copyright:
# Licence:
# ----------------------------------------------------------------------------------------------------------------------
# -- Short Description --
"""
This script calculates airborne LiDAR forest height statistics (90th percentile) per each ICESat-2 ATL08 segment
intersecting the two calibration sites
"""
# -------- Input --------
# (1) airborne LiDAR forest heights as 1m raster in the tiff format per each calibration site
# (2) ICESat-2 ATL08 segments of the intersecting orbits stored in the h5 files
# -------- Output -------
# (1) a single geodataframe (geopandas) including ATL08 segments from both calibration sites.
# The geo-data-frame contains ATL08 variables as well as airborne LiDAR statistics (90th percentile)
# ----------------------------------------------------------------------------------------------------------------------

import sys
import glob
import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats
from atl08_to_dataFrame import atl08_to_dataFrame
import shapely
# ---------------
sys.path.append('/home/milutin/Downloads/install/PhoREAL_v3.24/PhoREAL/source_code/')
# ---------------
from icesatIO import (calculateangle, calculategrounddirection)

# ------------------
# CHM raster:
# ------------------
chm_file_Para = '/mnt/ssd/milutin/Para_upScaling/Validation_GEDI_ICESat2/ALS_CHMs/Para_all_strips_norZ_q99.tif'
chm_file_MG = '/mnt/ssd/milutin/Para_upScaling/Validation_GEDI_ICESat2/ALS_CHMs/MG_all_strips_norZ_q99.tif'

# -----------------------------
# ATL08 files
# -----------------------------
atl08FilePaths_MG = glob.glob(r'/mnt/ssd/milutin/Para_upScaling/Validation_GEDI_ICESat2/ICESat2_MG/ATL08_*.h5')
atl08FilePaths_Para = glob.glob(r'/mnt/ssd/milutin/Para_upScaling/Validation_GEDI_ICESat2/ICESat2_Para/ATL08_*.h5')

# #######################################
# process all ATL08 files
# #######################################
# ------------------------
# read all atl08 fies and store in single df
# ------------------------
atl08_df_all_Para = pd.DataFrame()
for atl08FilePath in atl08FilePaths_Para:
    aux_df = atl08_to_dataFrame(atl08FilePath)
    atl08_df_all_Para = atl08_df_all_Para.append(aux_df, ignore_index=True)
#
atl08_df_all_MG = pd.DataFrame()
for atl08FilePath in atl08FilePaths_MG:
    aux_df = atl08_to_dataFrame(atl08FilePath)
    atl08_df_all_MG = atl08_df_all_MG.append(aux_df, ignore_index=True)
# ------------------------
# convert to geopandas df
# ------------------------
atl08_gdf_Para = gpd.GeoDataFrame(atl08_df_all_Para, geometry=gpd.points_from_xy(atl08_df_all_Para.lon, atl08_df_all_Para.lat))
atl08_gdf_Para.set_crs(epsg=4326, inplace=True)
#
atl08_gdf_MG = gpd.GeoDataFrame(atl08_df_all_MG, geometry=gpd.points_from_xy(atl08_df_all_MG.lon, atl08_df_all_MG.lat))
atl08_gdf_MG.set_crs(epsg=4326, inplace=True)
# ------------------------
# transform to the UTM 21S
# ------------------------
atl08_gdf_Para.to_crs(epsg=31981, inplace=True)
#
atl08_gdf_MG.to_crs(epsg=31981, inplace=True)

# ------------------------
# buffer all the points:
# ------------------------
# make a copy of data
atl08_gdf_Para_buf = atl08_gdf_Para.copy(deep=True)
atl08_gdf_MG_buf = atl08_gdf_MG.copy(deep=True)
# make a rectangular buffer around the points (footprint radius):
atl08_gdf_Para_buf['geometry'] = atl08_gdf_Para_buf['geometry'].buffer(5.5, cap_style=3)
atl08_gdf_MG_buf['geometry'] = atl08_gdf_MG_buf['geometry'].buffer(5.5, cap_style=3)
# scale y-axis to get rectangles (ATL segment is 11x100m)
atl08_gdf_Para_buf['geometry'] = atl08_gdf_Para_buf['geometry'].scale(1, 50/5.5, 1)
atl08_gdf_MG_buf['geometry'] = atl08_gdf_MG_buf['geometry'].scale(1, 50/5.5, 1)
#
# atl08_gdf_MG_buf.plot(color='None', edgecolor='red')
# atl08_gdf_Para_buf.head(2000).plot(color='None', edgecolor='red')

# ------------------------------
# calculate the rotation angles:
# -------------------------------
my_orbits = atl08_gdf_MG_buf.orbit.unique()
#
for my_orbit in my_orbits:
    orbit_gdf = atl08_gdf_MG_buf.loc[atl08_gdf_MG_buf.orbit == my_orbit]
    my_tracks = orbit_gdf.beamNum.unique()
    for my_track in my_tracks:
        track_gdf = orbit_gdf.loc[orbit_gdf.beamNum == my_track]
        my_xx = track_gdf.geometry.centroid.x.to_list()
        my_yy = track_gdf.geometry.centroid.y.to_list()
        #
        angle_deg = calculategrounddirection(my_xx, my_yy)
        #
        atl08_gdf_MG_buf.loc[track_gdf.index, 'myAzimuth'] = 90-angle_deg
# ----------------------------
my_orbits = atl08_gdf_Para_buf.orbit.unique()
#
for my_orbit in my_orbits:
    orbit_gdf = atl08_gdf_Para_buf.loc[atl08_gdf_Para_buf.orbit == my_orbit]
    my_tracks = orbit_gdf.beamNum.unique()
    for my_track in my_tracks:
        track_gdf = orbit_gdf.loc[orbit_gdf.beamNum == my_track]
        my_xx = track_gdf.geometry.centroid.x.to_list()
        my_yy = track_gdf.geometry.centroid.y.to_list()
        #
        angle_deg = calculategrounddirection(my_xx, my_yy)
        #
        atl08_gdf_Para_buf.loc[track_gdf.index, 'myAzimuth'] = 90-angle_deg
# -------------------------------
# rotate
# -------------------------------
for index, row in atl08_gdf_Para_buf.iterrows():
    rotated = shapely.affinity.rotate(row['geometry'], -1*row['myAzimuth'], use_radians=False, origin='centroid')
    atl08_gdf_Para_buf.loc[index, 'geometry'] = rotated
#
for index, row in atl08_gdf_MG_buf.iterrows():
    rotated = shapely.affinity.rotate(row['geometry'], -1*row['myAzimuth'], use_radians=False, origin='centroid')
    atl08_gdf_MG_buf.loc[index, 'geometry'] = rotated

# atl08_gdf_Para_buf.head(2000).plot(color='None', edgecolor='red')
# atl08_gdf_MG_buf.head(2000).plot(color='None', edgecolor='red')
# ------------------------------------
# calculate zonal statistics
# ------------------------------------
atl08_gdf_Para_buf['als_h'] = pd.DataFrame(zonal_stats(vectors=atl08_gdf_Para_buf['geometry'],
                                                       raster=chm_file_Para,
                                                       stats='percentile_90'))['percentile_90']
#
atl08_gdf_MG_buf['als_h'] = pd.DataFrame(zonal_stats(vectors=atl08_gdf_MG_buf['geometry'],
                                                     raster=chm_file_MG,
                                                     stats='percentile_90'))['percentile_90']

# #########################################
# pre-processing related to ALS
# #########################################
# ------------------------------------
# remove points with ALS height == None
# ------------------------------------
atl08_gdf_Para_buf.dropna(subset=['als_h'], inplace=True)
#
atl08_gdf_MG_buf.dropna(subset=['als_h'], inplace=True)
# ------------------------------------
# remove points with ALS height == 0
# ------------------------------------
atl08_gdf_Para_buf = atl08_gdf_Para_buf[atl08_gdf_Para_buf['als_h'] != 0]
#
atl08_gdf_MG_buf = atl08_gdf_MG_buf[atl08_gdf_MG_buf['als_h'] != 0]
# #########################################
# merge to gdf-s
# #########################################
# add the site column
atl08_gdf_Para_buf['Site'] = 'Para'
atl08_gdf_MG_buf['Site'] = 'MG'
# reset index
atl08_gdf_Para_buf.reset_index(inplace=True)
atl08_gdf_MG_buf.reset_index(inplace=True)
# merge
atl08_gdf = atl08_gdf_Para_buf.append(atl08_gdf_MG_buf, ignore_index=True)
# #########################################
# save
# #########################################
outFilePath = r'/mnt/ssd/milutin/Para_upScaling/Validation_GEDI_ICESat2/ICESat2_Para/ATL08_validation/ATL08_gdf_Para_MG.json'
with open(outFilePath, 'w') as f:
    f.write(atl08_gdf.to_json())

