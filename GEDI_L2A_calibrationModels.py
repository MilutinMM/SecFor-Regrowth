# Author:       Milutin Milenkovic
# Copyright:
# Licence:
# ----------------------------------------------------------------------------------------------------------------------
# -- Short Description --
"""
This script derives calibration models (linear fits) for different GEDI shot subgroups
"""
# -------- Input --------
# (1) a single geodataframe (geopandas) including GEDI shots from both calibration sites.
# -------- Output -------
# (1) xlsx files with calibration models and statistics reported in Section 4.1 (Milenkovic et al. 2022)
# (2) Figures from the Section 4.1 (Milenkovic et al. 2022)
# ----------------------------------------------------------------------------------------------------------------------

import os
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import wkt
from rasterstats import zonal_stats
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# #########################################
# Input
# #########################################
# shots with the default sensitives:
#inFilePath = r'/mnt/ssd/milutin/Para_upScaling/Validation_GEDI_ICESat2/GEDI_Para_v002_L2A_allTime/output/gedi_L2A_allTime_gdf_Para_MG.json'

# shots with sensitives from the algorithm setting group 2
inFilePath = r'/mnt/ssd/milutin/Para_upScaling/Validation_GEDI_ICESat2/GEDI_Para_v002_L2A/output_sens_a2/gedi_L2A_gdf_Para_MG_sens_a2.json'
#
gedi_gdf = gpd.read_file(inFilePath)

# #####################################################################################
# specify processing parameters:
# #####################################################################################
# specify if only shots from algorithm setting group 2 should be considered:
GROUP2_ONLY = False
# specify if only sensitivity from algorithm 2 should be used:
SENS_ALG2 = False
#
if SENS_ALG2:
    my_Sensitivity = 'geolocation_sensitivity_a2'
else:
    my_Sensitivity = 'sensitivity'

# ######################################################################################################
# exclude shots corresponding to deforestation between airborne liDAR and GEDI acquisitions (2018-2019)
# ######################################################################################################
deForPoly_gdf = gpd.read_file(r'/mnt/ssd/milutin/Para_upScaling/Validation_GEDI_ICESat2/ALS_CHMs/Deforestation_2018_2019/deforested_poligons_2018_2019.shp')
# set the crs:
gedi_gdf.set_crs(epsg=31981, inplace=True, allow_override=True)
# exclude gedi shots
gedi_gdf = gedi_gdf[~gedi_gdf.centroid.geometry.map(lambda x: deForPoly_gdf.geometry.contains(x)[0])]

# ########################################################################################
# select only shoots from the algorithm selection group 2
# ########################################################################################
# get the unique groups:
gedi_gdf.selected_algorithm.unique()
# get the number of shots per group
gedi_gdf.selected_algorithm.eq(1).sum()
gedi_gdf.selected_algorithm.eq(2).sum()
# select the group 2
if GROUP2_ONLY:
    gedi_gdf = gedi_gdf[gedi_gdf['selected_algorithm'] == 2]

# ########################################################
# final pre-processing and parameter setting
# ########################################################
# set the height column to analyse
height_col = 'rh_98'
# select the sensitivity level (for power and coverage day and night analysis):
# possible selections: 'QS90', 'QS95', 'QS98', or 'QS99'
my_name = 'QS90'
# filter out height equal to zero and heights above 75m:
gedi_gdf = gedi_gdf[gedi_gdf[height_col] != 0]
gedi_gdf = gedi_gdf[gedi_gdf[height_col] <= 75]
# calculate the GEDI - ALS difference column
gedi_gdf['Diff'] = gedi_gdf[height_col] - gedi_gdf['als_h']
# convert back to pandas df:
gedi_df = pd.DataFrame(gedi_gdf)

# #################################
# GEDI filtering - set I
# #################################
# apply quality flags
gedi_df_Q = gedi_df[gedi_df.degrade_flag == 0]
gedi_df_Q = gedi_df_Q[gedi_df_Q.quality_flag == 1]
# apply different sensitivity flags:
gedi_df_QS95 = gedi_df_Q[gedi_df_Q[my_Sensitivity] >= 0.95]
gedi_df_QS98 = gedi_df_Q[gedi_df_Q[my_Sensitivity] >= 0.98]
gedi_df_QS99 = gedi_df_Q[gedi_df_Q[my_Sensitivity] >= 0.99]
#
my_df_dict = {'QS90': gedi_df_Q, 'QS95': gedi_df_QS95, 'QS98': gedi_df_QS98, 'QS99': gedi_df_QS99}
# #################################
# GEDI filtering - set II
# #################################
power_beams = ['BEAM0101', 'BEAM0110', 'BEAM1000', 'BEAM1011']
coverage_beams = ['BEAM0000', 'BEAM0001', 'BEAM0010', 'BEAM0011']
# set starting sensitivity:
gedi_df_QS = my_df_dict.get(my_name)
# high power and coverage beams (including day and nigth)
gedi_df_QP = gedi_df_QS[gedi_df_QS.BEAM.isin(power_beams)]
gedi_df_QC = gedi_df_QS[gedi_df_QS.BEAM.isin(coverage_beams)]
# further split to day and night data
gedi_df_QPD = gedi_df_QP[gedi_df_QP.solar_elevation >= 0]
gedi_df_QPN = gedi_df_QP[gedi_df_QP.solar_elevation < 0]
gedi_df_QCD = gedi_df_QC[gedi_df_QC.solar_elevation >= 0]
gedi_df_QCN = gedi_df_QC[gedi_df_QC.solar_elevation < 0]
# day and night data (including pow and coverage beams)
gedi_df_QD = gedi_df_QS[gedi_df_QS.solar_elevation >= 0]
gedi_df_QN = gedi_df_QS[gedi_df_QS.solar_elevation < 0]
# ##############################################################################################
# make lists for iterations
# ##############################################################################################
# make a list of gdf-s:
gdf_list = [gedi_df_Q, gedi_df_QS95, gedi_df_QS98, gedi_df_QS99, gedi_df]
df_names = ["Q", "QS95", "QS98", "QS99", "ALL"]
#
# gdf_list = [gedi_df_QPN, gedi_df_QPD, gedi_df_QP,
#             gedi_df_QCN, gedi_df_QCD, gedi_df_QC,
#             gedi_df_QN, gedi_df_QD, gedi_df_QS]
#
# df_names = ["QPN", "QPD", "QPND",
#             "QCN", "QCD", "QCND",
#             "QPCN", "QPCD", my_name]

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
    #
    aux_list = my_gdf.shot_number.astype('str').to_list()
    aux_orbits = np.array([int(shotNumber[:5]) for shotNumber in aux_list])
    #
    my_orbits.append(np.unique(aux_orbits).shape[0])
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
#out_htmlName = height_col + '_stats_hDiff_Para_MG.html'
#out_htmlName = height_col + '_stats_hDiff_Para_MG_deFor2018_2019.html'
#out_htmlName = height_col + '_stats_hDiff_Para_MG_' + my_name + '.html'
#out_htmlName = height_col + '_stats_hDiff_Para_MG_' + my_name + '_deFor2018_2019.html'
out_htmlName = height_col + '_stats_hDiff_Para_MG_' + 'Sensitivity' + '_sens_2.html'
#out_htmlName = height_col + '_stats_hDiff_Para_MG_' + my_name + '_sens_a2.html'
my_stats_df.to_html(os.path.join(os.path.dirname(inFilePath), out_htmlName))
# write to a excel file:
my_stats_df.to_excel(os.path.join(os.path.dirname(inFilePath), out_htmlName[:-4] + 'xlsx'), index_label=False)

# ###################################################################################
# plot histogram of diff values and scatter of gedi and als heights
# ###################################################################################
plotting_gdf_list = [gedi_df, gedi_df_QS95, gedi_df_QS98, gedi_df_QS99]
# specify proper names
if GROUP2_ONLY:
    plotting_df_names = ["ALL-A2", "QS95-A2", "QS98-A2", "QS99-A2"]
elif SENS_ALG2:
    plotting_df_names = ["ALL-S2", "QS95-S2", "QS98-S2", "QS99-S2"]
else:
    plotting_df_names = ["ALL", "QS95", "QS98", "QS99"]
#
#plotting_gdf_list = [gedi_df_QS, gedi_df_QC, gedi_df_QP, gedi_df_QPN]
#plotting_df_names = ["QS90", "QS90-CND", "QS90-PND", "QS90-PN"]

fig1, ax1 = plt.subplots(2, 4, figsize=(12, 6))
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
    my_gedi_df.plot.scatter(x='als_h', y=height_col, s=2, color='grey', ax=ax1[0, myIndx])
    ax1[0, myIndx].plot(aux_x, aux_y, 'r-', label="y={0:.2f}x+{1:.2f}".format(model1.coef_[0], model1.intercept_), lw=0.75)
    ax1[0, myIndx].plot(aux_x, aux_x, 'k--', label="y=x", lw=1.5)
    ax1[0, myIndx].legend(loc='upper left')
    ax1[0, myIndx].set_xlim([0, 75])
    ax1[0, myIndx].set_ylim([0, 75])
    ax1[0, myIndx].text(45, 10, "$R^2$ = {0: .2f}".format(model1.score(xx, yy)), fontsize=10)
    ax1[0, myIndx].text(45, 5, "Shots = {}".format(str(my_gedi_df.shape[0])), fontsize=10)
    ax1[0, myIndx].title.set_text(plotting_df_names[myIndx])
    if myIndx == 0:
        ax1[0, myIndx].set_ylabel('GEDI RH98 Values [m]')
    else:
        ax1[0, myIndx].set_ylabel('')
    ax1[0, myIndx].set_xlabel('Airborne LiDAR Forest Heights [m]')
    ax1[0, myIndx].set_aspect('equal')
    ax1[0, myIndx].grid()
    # ---------------------------------------------------------
    # plot histogram of all differences
    # ---------------------------------------------------------
    bin_min = np.floor(my_gedi_df['Diff'].min())
    bin_max = np.ceil(my_gedi_df['Diff'].max())
    #
    #bin_min = -20
    #bin_max = 20
    my_step = 1
    my_bins = np.arange(bin_min, bin_max + my_step, my_step)
    #
    my_gedi_df['Diff'].hist(bins=my_bins, edgecolor='black', ax=ax1[1, myIndx])
    #ax1[1, myIndx].set_xlabel('Forest Height Difference ($\Delta H_{GEDI} = H_{GEDI} - H_{Ref}$) [m]')
    ax1[1, myIndx].set_xlabel('Forest Height Difference [m]')
    ax1[1, myIndx].set_xlim([-30, 30])
    ax1[1, myIndx].set_ylim([0, 120])
    ax1[1, myIndx].set_box_aspect(1)
    if myIndx == 0:
        ax1[1, myIndx].set_ylabel('Number of GEDI Shots')

fig1.tight_layout()

# save figure:
out_FigName = height_col + 'scatterPlots_Para_MG_Sensitivity_sens_a2.png'
#
#out_FigName = height_col + '_scatterPlots_Para_MG_' + my_name + '_group2.png'
plt.savefig(os.path.join(os.path.dirname(inFilePath), out_FigName), dpi=150)

# ###################################################################################
# plot histogram of sensitivity
# ###################################################################################
fig1, ax1 = plt.subplots(1, 4, figsize=(12, 3))
#ax1[0].hist(gedi_df['sensitivity'].values, bins=np.arange(0, 1.01, 0.01), edgecolor='black', facecolor='seagreen', alpha=1, label='ALL')
ax1[0].hist(gedi_df_Q['sensitivity'].values, bins=np.arange(0, 1.01, 0.01), edgecolor='black', facecolor='lightgreen', alpha=1, label='QS90')
ax1[0].hist(gedi_df_QP['sensitivity'].values, bins=np.arange(0, 1.01, 0.01), edgecolor='black', facecolor='tomato', alpha=1, label='QS90-PND')
ax1[0].hist(gedi_df_QC['sensitivity'].values, bins=np.arange(0, 1.01, 0.01), edgecolor='black', facecolor='gold', alpha=0.75, label='QS90-CND')
ax1[0].set_xlim([0.9, 1])
ax1[0].legend()
#
#ax1[1].hist(gedi_df['sensitivity'].values, bins=np.arange(0, 1.01, 0.01), edgecolor='black', facecolor='seagreen', alpha=1, label='ALL')
ax1[1].hist(gedi_df_Q['sensitivity'].values, bins=np.arange(0, 1.01, 0.01), edgecolor='black', facecolor='lightgreen', alpha=1, label='QS90')
ax1[1].hist(gedi_df_QN['sensitivity'].values, bins=np.arange(0, 1.01, 0.01), edgecolor='black', facecolor='dimgrey', alpha=1, label='QS90-NPC')
ax1[1].hist(gedi_df_QD['sensitivity'].values, bins=np.arange(0, 1.01, 0.01), edgecolor='black', facecolor='lightgrey', alpha=0.75, label='QS90-DPC')
ax1[1].set_xlim([0.9, 1])
ax1[1].legend()
#
ax1[2].hist(gedi_df_QP['sensitivity'].values, bins=np.arange(0, 1.01, 0.01), edgecolor='black', facecolor='tomato', alpha=1, label='QS90-PND')
ax1[2].hist(gedi_df_QPN['sensitivity'].values, bins=np.arange(0, 1.01, 0.01), edgecolor='black', facecolor='dimgrey', alpha=1, label='QS90-PN')
ax1[2].hist(gedi_df_QPD['sensitivity'].values, bins=np.arange(0, 1.01, 0.01), edgecolor='black', facecolor='lightgrey', alpha=0.75, label='QS90-PD')
ax1[2].set_xlim([0.90, 1])
ax1[2].legend()
#
ax1[3].hist(gedi_df_QC['sensitivity'].values, bins=np.arange(0, 1.01, 0.01), edgecolor='black', facecolor='gold', alpha=1, label='QS90-CND')
ax1[3].hist(gedi_df_QCN['sensitivity'].values, bins=np.arange(0, 1.01, 0.01), edgecolor='black', facecolor='dimgrey', alpha=1, label='QS90-CN')
ax1[3].hist(gedi_df_QCD['sensitivity'].values, bins=np.arange(0, 1.01, 0.01), edgecolor='black', facecolor='lightgrey', alpha=0.75, label='QS90-CD')
ax1[3].set_xlim([0.9, 1])
ax1[3].legend()
#
ax1[0].set_xlabel('Beam Sensitivity')
ax1[1].set_xlabel('Beam Sensitivity')
ax1[2].set_xlabel('Beam Sensitivity')
ax1[3].set_xlabel('Beam Sensitivity')
ax1[0].set_ylabel('GEDI Shots')
#
fig1.tight_layout()

out_FigName = 'Histograms_Sensitivity.png'
plt.savefig(os.path.join(os.path.dirname(inFilePath), out_FigName), dpi=150)

# ###################################################################################
# plot sensitivity as bar plots
# ###################################################################################
# set width of bars
barWidth = 0.14

# set heights of bars
bars_Q = np.histogram(gedi_df_Q['sensitivity'].values, bins=np.arange(0.9, 1.01, 0.01))[0]
bars_QP = np.histogram(gedi_df_QP['sensitivity'].values, bins=np.arange(0.9, 1.01, 0.01))[0]
bars_QC = np.histogram(gedi_df_QC['sensitivity'].values, bins=np.arange(0.9, 1.01, 0.01))[0]
bars_QN = np.histogram(gedi_df_QN['sensitivity'].values, bins=np.arange(0.9, 1.01, 0.01))[0]
bars_QD = np.histogram(gedi_df_QD['sensitivity'].values, bins=np.arange(0.9, 1.01, 0.01))[0]

# Set position of bar on X axis
r1 = np.arange(len(bars_Q))
r2 = [x - barWidth for x in r1]
r3 = [x - barWidth for x in r2]
r4 = [x + barWidth for x in r1]
r5 = [x + barWidth for x in r4]

# Make the plot
fig1, ax1 = plt.subplots(1, 1, figsize=(6, 3))
ax1.bar(r1, bars_Q, color='lightgreen', width=barWidth, edgecolor='black', label='QS90')
ax1.bar(r2, bars_QP, color='tomato', width=barWidth, edgecolor='black', label='QS90-PND')
ax1.bar(r3, bars_QC, color='gold', width=barWidth, edgecolor='black', label='QS90-CND')
ax1.bar(r4, bars_QN, color='dimgrey', width=barWidth, edgecolor='black', label='QS90-PCN')
ax1.bar(r5, bars_QD, color='lightgrey', width=barWidth, edgecolor='black', label='QS90-PCD')

# Add xticks on the middle of the group bars
ax1.set_xlabel('Beam Sensitivity')
my_ticks = ["[0.90,0.91)", "[0.91,0.92)", "[0.92,0.93)", "[0.93,0.94)", "[0.94,0.95)", "[0.95,0.96)", "[0.96,0.97)",
            "[0.97,0.98)", "[0.98,0.99)", "[0.99,1)"]

ax1.set_xticks(r1)
ax1.set_xticklabels(my_ticks, rotation=-45, ha='left')
#
ax1.set_ylabel('GEDI Shots')
# Create legend & Show graphic
plt.legend()
fig1.tight_layout()

out_FigName = 'Bins_Sensitivity.png'
plt.savefig(os.path.join(os.path.dirname(inFilePath), out_FigName), dpi=150)










