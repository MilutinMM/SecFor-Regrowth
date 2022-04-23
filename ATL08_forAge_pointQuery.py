# Author:       Milutin Milenkovic
# Copyright:
# Licence:
# ----------------------------------------------------------------------------------------------------------------------
# -- Short Description --
"""
This script assigns the forest age class and border pixel flag to ATL08 segments.
"""
# -------- Input --------
# (1) ATL08 segments over secondary forest in the Rondônia state, Brazil.
# (2) a forest age raster: Silva Junior et al. 2020, https://doi.org/10.6084/m9.figshare.12622025
# (3) a black-and-withe version of the forest age raster with border pixels removed
# -------- Output -------
# (1) The input ATL08 segments with forest age attribute and border pixel flag
# ----------------------------------------------------------------------------------------------------------------------

import glob
import numpy as np
import pandas as pd
import geopandas as gpd
from atl08_to_dataFrame import atl08_to_dataFrame
from rasterstats import point_query

# ------------------
# Forest Age raster:
# ------------------
forestAge_file = r'/mnt/ssd/milutin/Para_upScaling/StandAgeMapBiomas/svbr-rondonia-2018.tif'
# eroded forest age raster (to account for geolocation/border errors)
out_eroded_forestAge_file = r'/mnt/ssd/milutin/Para_upScaling/StandAgeMapBiomas/svbr-rondonia-2018_bw_eroded.tif'
# ------------------
# process all ATL08 files
# ------------------
atl08FilePaths = glob.glob(r'/mnt/raid/milutin/upScaling/Rondonia/ICESat2/ATL08_v003/ATL08_*.h5')

# ---------------------------------------------
# merge all ATL08 files into single data frame
# ---------------------------------------------
atl08_df_all = pd.DataFrame()
for atl08FilePath in atl08FilePaths:
    aux_df = atl08_to_dataFrame(atl08FilePath)
    atl08_df_all = atl08_df_all.append(aux_df, ignore_index=True)

# ------------------------
# convert to geopandas df
# ------------------------
atl08_gdf = gpd.GeoDataFrame(atl08_df_all, geometry=gpd.points_from_xy(atl08_df_all.lon, atl08_df_all.lat))
atl08_gdf.set_crs(epsg=4326, inplace=True)

# -------------------------------------------------
# drop rows with erroneously large hCanopy values:
# -------------------------------------------------
atl08_gdf = atl08_gdf[atl08_gdf.hCanopy <= 1e20]
# ------------------------------------
# calculate zonal statistics
# ------------------------------------
forestAge_list = point_query(vectors=atl08_gdf['geometry'], raster=forestAge_file, interpolate='nearest')
atl08_gdf['forestAge'] = np.asarray(forestAge_list)
# ------------------------------------
# remove points with stand age == 0
# ------------------------------------
atl08_gdf = atl08_gdf[atl08_gdf['forestAge'] != 0]
# ------------------------------------
# remove points with stand age == None
# ------------------------------------
atl08_gdf = atl08_gdf[~atl08_gdf['forestAge'].isna()]
# ----------------------------------------------
# get pixel value from the eroded bw forest age
# ----------------------------------------------
forestAge_list = point_query(vectors=atl08_gdf['geometry'], raster=out_eroded_forestAge_file, interpolate='nearest')
atl08_gdf['eroded_age'] = np.asarray(forestAge_list)
# ------------------------------------
# save the ATL08 data frame on disk
# ------------------------------------
outFilePath = r'/mnt/raid/milutin/upScaling/Rondonia/ICESat2/ATL08_v003/directAnalysis/ATL08_gdf.json'
with open(outFilePath, 'w') as f:
    f.write(atl08_gdf.to_json())