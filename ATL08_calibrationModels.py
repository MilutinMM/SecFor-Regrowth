# Author:       Milutin Milenkovic
# Copyright:
# Licence:
# ----------------------------------------------------------------------------------------------------------------------
# -- Short Description --
"""
This script derives calibration models (linear fits) for different ICESat-2 ATL08 segment subgroups
"""
# -------- Input --------
# (1) a single geodataframe (geopandas) including ATL08 segments from both calibration sites.
# -------- Output -------
# (1) xlsx files with calibration models and statistics reported in Section 4.2 (Milenkovic et al. 2022)
# (2) Figures from the Section 4.2 (Milenkovic et al. 2022)
# ----------------------------------------------------------------------------------------------------------------------

import os
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib

# #########################################
# Input
# #########################################
# file path to the geodataframe with ATL08 segment variables and airborne LiDAR forest heights per segment
inFilePath = r'/mnt/ssd/milutin/Para_upScaling/Validation_GEDI_ICESat2/ICESat2_Para/ATL08_validation/ATL08_gdf_Para_MG.json'
#
atl08_gdf = gpd.read_file(inFilePath)

# #########################################################################################################
# exclude segments corresponding to deforestation between airborne LiDAR and GEDI acquisitions (2018-2019)
# #########################################################################################################
deForPoly_gdf = gpd.read_file(r'/mnt/ssd/milutin/Para_upScaling/Validation_GEDI_ICESat2/ALS_CHMs/Deforestation_2018_2019/deforested_poligons_2018_2019.shp')
# set the crs:
atl08_gdf.set_crs(epsg=31981, inplace=True, allow_override=True)
# exclude gedi shots
atl08_gdf = atl08_gdf[~atl08_gdf.centroid.geometry.map(lambda x: deForPoly_gdf.geometry.contains(x)[0])]

# ########################################################
# final pre-processing and parameter setting
# ########################################################
# set the height column to analyse
height_col = 'hCanopy'
#
# drop rows with large heights:
atl08_gdf = atl08_gdf[atl08_gdf.hCanopy <= 1e20]
# filter out height equal to zero and heights above 75m
atl08_df = atl08_gdf[atl08_gdf[height_col] != 0]
atl08_df = atl08_gdf[atl08_gdf[height_col] <= 75]
# calculate the GEDI - ALS difference column
atl08_gdf['Diff'] = atl08_gdf[height_col] - atl08_gdf['als_h']
# convert back to data frame
atl08_df = pd.DataFrame(atl08_gdf)
# --
# apply canopy photons filter
#atl08_df = atl08_df[atl08_df.caPhoNum < 140]
# #################################
# ATL filtering
# #################################
# select strong and weak points
atl08_S = atl08_df[atl08_df.beamStreng == 1]
atl08_W = atl08_df[atl08_df.beamStreng == 0]
# select night and day points
atl08_SN = atl08_S[atl08_S.nightFlag == 1]
atl08_SD = atl08_S[atl08_S.nightFlag == 0]
# --
# apply canopy photons filter
atl08_SN = atl08_SN[atl08_SN.caPhoNum < 140]
# --
atl08_WN = atl08_W[atl08_W.nightFlag == 1]
atl08_WD = atl08_W[atl08_W.nightFlag == 0]
# select day-night in strong+weak points
atl08_SWN = atl08_SN.append(atl08_WN)
atl08_SWD = atl08_SD.append(atl08_WD)
# remake S and All (!!! because of SN photon filter !!!):
atl08_S = atl08_SN.append(atl08_SD)
atl08_All = atl08_S.append(atl08_W)
# ##############################################################################################
# make lists for iterations
# ##############################################################################################
# make a list of gdf-s:
gdf_list = [atl08_SN, atl08_SD, atl08_S,
            atl08_WN, atl08_WD, atl08_W,
            atl08_SWN, atl08_SWD, atl08_All]
df_names = ["SN", "SD", "SND",
            "WN", "WD", "WDN",
            "SWN", "SWD", "ALL"]

# ###################################################################################
# get some statistics
# ###################################################################################
# initiate empty lists to store stats:
my_mae = []
my_mean_e = []
my_rmse = []
my_samples = []
my_orbits = []
#
my_slope = []
my_intercept = []
my_r2 = []
# iterate and get statistics:
for my_gdf in gdf_list:
    # ------------------------------------
    # get max absolute error:
    my_mae.append(my_gdf['Diff'].abs().max())
    # get mean error:
    my_mean_e.append(my_gdf['Diff'].mean())
    # get RMSE:
    my_rmse.append((my_gdf['Diff']**2).mean()**0.5)
    #
    my_samples.append(my_gdf.shape[0])
    # combine orbit and cycle:
    my_gdf['orbit_cycle'] = 100*my_gdf.orbit + my_gdf.cycle
    #
    my_orbits.append(my_gdf['orbit_cycle'].unique().shape[0])
    # ------------------------------------
    # fit the regression line:
    if my_gdf.shape[0] > 2:
        xx = my_gdf['als_h'].values.reshape((-1, 1))
        yy = my_gdf[height_col].values
        #
        model1 = LinearRegression()
        model1.fit(xx, yy)
        #
        my_slope.append(model1.coef_[0])
        #
        my_intercept.append(model1.intercept_)
        #
        my_r2.append(model1.score(xx, yy))
    else:
        my_slope.append(np.nan)
        my_intercept.append(np.nan)
        my_r2.append(np.nan)

my_stats_df = pd.DataFrame(list(zip(df_names, my_samples, my_orbits, my_mae, my_mean_e, my_rmse, my_slope, my_intercept, my_r2)),
                           columns=['Filt_name', 'Pts_num', 'Orb_num', 'MaxAbsE', 'MeanE', 'RMSE', 'Slope', 'Intercept', 'R2'])


# save the statistics:
out_htmlName = height_col + '_stats_hDiff_Para_MG.html'
#out_htmlName = height_col + '_stats_hDiff_Para_MG_canopyPhoton_lt140.html'
my_stats_df.to_html(os.path.join(os.path.dirname(inFilePath), out_htmlName))
# write to a excel file:
my_stats_df.to_excel(os.path.join(os.path.dirname(inFilePath), out_htmlName[:-4] + 'xlsx'), index_label=False)


# ###################################################################################
# plot histogram of diff values and scatter of atl08 and als heights
# ###################################################################################
#plotting_gdf_list = [atl08_df, atl08_S, atl08_SN, atl08_SD]
plotting_gdf_list = [atl08_All, atl08_S, atl08_SN, atl08_SD]
plotting_df_names = ["ALL", "SND", "SN", "SD"]

fig1, ax1 = plt.subplots(2, 4, figsize=(12, 6))
for myIndx in np.arange(len(plotting_gdf_list)):
    # select the df
    my_atl08_df = plotting_gdf_list[myIndx]
    # ---------------------------------------------------------
    # plot scatter plot of all differences
    # ---------------------------------------------------------
    # get the regression line:
    xx = my_atl08_df['als_h'].values.reshape((-1, 1))
    yy = my_atl08_df[height_col].values
    #
    model1 = LinearRegression()
    model1.fit(xx, yy)
    #
    aux_x = np.arange(0, 76, 1)
    aux_y = model1.intercept_ + aux_x * model1.coef_[0]
    # plot
    my_atl08_df.plot.scatter(x='als_h', y=height_col, s=4, color='grey', ax=ax1[0, myIndx])
    ax1[0, myIndx].plot(aux_x, aux_y, 'r-', label="y={0:.2f}x+{1:.2f}".format(model1.coef_[0], model1.intercept_), lw=0.75)
    ax1[0, myIndx].plot(aux_x, aux_x, 'k--', label="y=x", lw=1.5)
    ax1[0, myIndx].legend(loc='lower right')
    ax1[0, myIndx].set_xlim([0, 75])
    ax1[0, myIndx].set_ylim([0, 75])
    ax1[0, myIndx].text(3, 65, "$R^2$ = {0: .2f}".format(model1.score(xx, yy)), fontsize=10)
    ax1[0, myIndx].text(3, 60, "Points = {}".format(str(my_atl08_df.shape[0])), fontsize=10)
    ax1[0, myIndx].title.set_text(plotting_df_names[myIndx])
    if myIndx == 0:
        ax1[0, myIndx].set_ylabel('ICESat-2 RH98 Values [m]')
    else:
        ax1[0, myIndx].set_ylabel('')
    ax1[0, myIndx].set_xlabel('Airborne LiDAR Forest Heights [m]')
    ax1[0, myIndx].set_aspect('equal')
    ax1[0, myIndx].grid()
    # ---------------------------------------------------------
    # plot histogram of all differences
    # ---------------------------------------------------------
    bin_min = np.floor(my_atl08_df['Diff'].min())
    bin_max = np.ceil(my_atl08_df['Diff'].max())
    #
    # bin_min = -20
    # bin_max = 20
    my_step = 1
    my_bins = np.arange(bin_min, bin_max + my_step, my_step)
    #
    my_atl08_df['Diff'].hist(bins=my_bins, edgecolor='black', ax=ax1[1, myIndx])
    # ax1[1, myIndx].set_xlabel('Forest Height Difference ($\Delta H_{GEDI} = H_{GEDI} - H_{Ref}$) [m]')
    ax1[1, myIndx].set_xlabel('Forest Height Difference [m]')
    ax1[1, myIndx].set_xlim([-30, 30])
    ax1[1, myIndx].set_ylim([0, 13])
    ax1[1, myIndx].set_box_aspect(1)
    if myIndx == 0:
        ax1[1, myIndx].set_ylabel('Number of ICESat-2 Points')

fig1.tight_layout()

out_FigName = height_col + '_scatterPlots_Para_MG.png'
plt.savefig(os.path.join(os.path.dirname(inFilePath), out_FigName), dpi=150)


# ###################################################################################
# plot scatter with color-coded pts
# ###################################################################################
atl08_SN_filt = atl08_SN[atl08_SN.caPhoNum < 140]
atl08_SND_filt = atl08_SN_filt.append(atl08_SD)
# get RMSE:
(atl08_SN_filt['Diff']**2).mean()**0.5
(atl08_SND_filt['Diff']**2).mean()**0.5
# get mean error:
atl08_SN_filt['Diff'].mean()
atl08_SND_filt['Diff'].mean()
# get max absolute error:
atl08_SN_filt['Diff'].abs().max()
atl08_SND_filt['Diff'].abs().max()
# Num of points
atl08_SN_filt.shape[0]
atl08_SND_filt.shape[0]
    # combine orbit and cycle:
    my_gdf['orbit_cycle'] = 100*my_gdf.orbit + my_gdf.cycle
    #
    my_orbits.append(my_gdf['orbit_cycle'].unique().shape[0])



#
plotting_gdf_list = [atl08_SN, atl08_SN, atl08_SN, atl08_SN_filt]
# print the unique orbits:
atl08_SN.orbit.unique()
# prepare the color list:
colors = {4974: 'tab:blue', 5416: 'tab:green'}
my_colors = atl08_SN['orbit'].map(colors)
#
my_cmap = matplotlib.cm.summer
#
fig1, ax1 = plt.subplots(1, 4, figsize=(12, 4))
for myIndx in np.arange(len(plotting_gdf_list)):
    # select the df
    my_gedi_df = plotting_gdf_list[myIndx]
    # ---------------------------------------------------------
    # plot scatter plot of all differences
    # ---------------------------------------------------------
    # get the regression line:
    xx = my_gedi_df['als_h'].values.reshape((-1, 1))
    yy = my_gedi_df[height_col].values
    #
    model1 = LinearRegression()
    model1.fit(xx, yy)
    #
    aux_x = np.arange(0, 76, 1)
    aux_y = model1.intercept_ + aux_x * model1.coef_[0]
    # plot
    if myIndx == 0 or myIndx == 3:
        my_gedi_df.plot.scatter(x='als_h', y=height_col, s=6, c='grey', ax=ax1[myIndx])
    elif myIndx == 1:
        my_gedi_df.plot.scatter(x='als_h', y=height_col, s=6, c=my_colors, ax=ax1[myIndx])
    elif myIndx == 2:
        #my_atl08_df.plot.scatter(x='als_h', y=height_col, s=6, c=my_atl08_df.caPhoNum, colormap=my_cmap.reversed(), colorbar=False, ax=ax1[myIndx])
        my_sk = ax1[myIndx].scatter(my_gedi_df.als_h, my_gedi_df[height_col], s=6, c=my_gedi_df.caPhoNum, cmap=my_cmap.reversed())
    #
    ax1[myIndx].plot(aux_x, aux_y, 'r-', label="y={0:.2f}x+{1:.2f}".format(model1.coef_[0], model1.intercept_), lw=0.75)
    ax1[myIndx].plot(aux_x, aux_x, 'k--', label="y=x", lw=1.5)
    ax1[myIndx].legend(loc='lower right')
    ax1[myIndx].set_xlim([0, 75])
    ax1[myIndx].set_ylim([0, 75])
    ax1[myIndx].text(3, 65, "$R^2$ = {0: .2f}".format(model1.score(xx, yy)), fontsize=10)
    ax1[myIndx].text(3, 60, "Points = {}".format(str(my_gedi_df.shape[0])), fontsize=10)
    if myIndx==0:
        ax1[myIndx].set_ylabel('ICESat-2 RH98 Values [m]')
    else:
        ax1[myIndx].set_ylabel('')
    ax1[myIndx].set_xlabel('Airborne LiDAR Forest Heights [m]')
    ax1[myIndx].set_aspect('equal')
    ax1[myIndx].grid()
    # add a legend
    if myIndx == 1:
        handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=v, label=k, markersize=8) for k, v in colors.items()]
        ax1[myIndx].legend(title='Orbit', handles=handles, bbox_to_anchor=(0.5, 1.01), loc='lower center', ncol=len(handles))
    # set colorbar label
    divider = make_axes_locatable(ax1[myIndx])
    cax1 = divider.append_axes("top", size="5%", pad=0.25)
    if myIndx == 2:
        fig1.colorbar(my_sk, cax=cax1, orientation="horizontal")
        cax1.xaxis.set_ticks_position("top")
        cax1.set_xlabel('Canopy Photons per Segment')
    else:
        cax1.remove()
    if myIndx == 3:
        ax1[myIndx].title.set_text('Filter: Canopy Photons < 140')

    if myIndx == 0:
        ax1[myIndx].title.set_text('SN')

fig1.tight_layout()

out_FigName = 'FilteringSN_' + height_col + '_scatterPlots_Para_MG.png'
plt.savefig(os.path.join(os.path.dirname(inFilePath), out_FigName), dpi=150)



