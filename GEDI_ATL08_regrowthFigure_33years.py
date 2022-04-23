# Author:       Milutin Milenkovic
# Copyright:
# Licence:
# ----------------------------------------------------------------------------------------------------------------------
# -- Short Description --
"""
This script plots a figure with forest heigths interquantiles of GEDI ALL and ICESat-2 ALL subgroups over 33-year
regrowth period
"""
# -------- Input --------
# (1) GEDI shots and ATL08 segments over secondary forest in the Rondônia state, Brazil.
# The input is hardcoded in the 'regrowth_modeling_auxiliary_sctipt/gedi_icesat2_processor' function.
# -------- Output -------
# (1) A figure reported in Section 4.3 (Milenkovic et al. 2022)
# ----------------------------------------------------------------------------------------------------------------------

import os
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import wkt
#from rasterstats import zonal_stats
from rasterstats import point_query
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
#
from atl08_to_dataFrame import atl08_to_dataFrame
from rasterio.plot import show_hist
#
from modeling_33years_sctipt import gedi_icesat2_processor

# #####################################################################################
# processing parameters
# #####################################################################################
# specify if only shots from algorithm setting group 2 should be considered:
GROUP2_ONLY = False
# specify if only sensitivity from algorithm 2 should be used:
SENS_ALG2 = True
# specify that calibrated RH98 values should be calculated and added as extra column:
CALIBRATED_RH98 = True
# specify if only subgroups with different sensitivity level should be returned
SENS_SUBGROUPS_ONLY = True
# #####################################################################################
# process the GEDI and ICESat-2 data
# #####################################################################################
# obtain the subgroups:
[gedi_df_subgroups, atl08_df_subgroups] = gedi_icesat2_processor(CALIBRATED_RH98, SENS_ALG2, GROUP2_ONLY, SENS_SUBGROUPS_ONLY)
# specify the gedi subgroups' names:
if SENS_SUBGROUPS_ONLY:
    gedi_df_names = ["ALL", "QS90", "QS95", "QS98", "QS99"]
else:
    gedi_df_names = ["QS90-QPN", "QS90-QPD", "QS90-QPND",
                     "QS90-QCN", "QS90-QCD", "QS90-QCND",
                     "QS90-QPCN", "QS90-QPCD", "QS90"]

# specify the ICESat-2 subgroups' names:
atl08_df_names = ["SN", "SD", "SND",
                  "WN", "WD", "WDN",
                  "SWN", "SWD", "ALL"]

# #######################################################
# Fit the regrowth exponential and log model and asses the errors
# #######################################################
# specify the GEDI and ICESat-2 subgroups to analyse:
my_gedi_name = 'ALL'
my_ATL08_name = 'ALL'
# get the subgroups' data frames:
my_gedi_df = gedi_df_subgroups[gedi_df_names.index(my_gedi_name)]
my_ATL08_df = atl08_df_subgroups[atl08_df_names.index(my_ATL08_name)]
# specify the forest age max years:
maxYears = 33
# specify if calibrated RH98 values should be considered for regrowth rate:
if CALIBRATED_RH98:
    height_col_gedi = 'rh_98_cal'
    height_col_atl08 = 'hCanopy_cal'

# ############################################
# GEDI regrowth heights
# ############################################
# group the data by stand age:
df_standAge_grouped_gedi = my_gedi_df.groupby('forestAge').quantile(0.5)
# select only first
df_standAge_grouped_gedi = df_standAge_grouped_gedi[0:maxYears]
# prepare data for sklearn fit:
xx_gedi = df_standAge_grouped_gedi.index.values.reshape((-1, 1))
yy_gedi = df_standAge_grouped_gedi[height_col_gedi].values
# --------
# get inter-quartile range:
df_gedi_q25 = my_gedi_df.groupby('forestAge').quantile(0.25)
df_gedi_q75 = my_gedi_df.groupby('forestAge').quantile(0.75)
#
interQ_gedi = df_gedi_q75[height_col_gedi].values - df_gedi_q25[height_col_gedi].values
# --------
# number of shots per stand age:
NumShots = my_gedi_df.groupby('forestAge').count().id.values

# ############################################
# ICSat-2 regrowth heights
# ############################################
# group the data by stand age:
df_standAge_grouped_atl08 = my_ATL08_df.groupby('forestAge').quantile(0.5)
# select only first
df_standAge_grouped_atl08 = df_standAge_grouped_atl08[0:maxYears]
# prepare data for sklearn fit:
xx_atl08 = df_standAge_grouped_atl08.index.values.reshape((-1, 1))
yy_atl08 = df_standAge_grouped_atl08[height_col_atl08].values
# --------
# get inter-quartile range:
df_atl08_q25 = my_ATL08_df.groupby('forestAge').quantile(0.25)
df_atl08_q75 = my_ATL08_df.groupby('forestAge').quantile(0.75)
#
interQ_atl08 = df_atl08_q75[height_col_atl08].values - df_atl08_q25[height_col_atl08].values
# --------
# number of segments per stand age:
NumSegments = my_ATL08_df.groupby('forestAge').count().id.values
# ############################################
# Plotting
# #########################################################################
fig1, ax1 = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
#

ax1[0, 0].errorbar(xx_gedi, yy_gedi, yerr=interQ_gedi, fmt='o', color='C0', markersize=4, ecolor='grey', capsize=2,
                   label='Median and interquartile range')
#ax1[0, 0].scatter(xx_gedi, yy_gedi, s=20, c='C0', label='Median')
#
ax1[0, 1].errorbar(xx_atl08, yy_atl08, yerr=interQ_atl08, fmt='o', color='C0', markersize=4, ecolor='grey', capsize=2,
                   label='Median and interquartile range')
#ax1[0, 1].scatter(xx_atl08, yy_atl08, s=20, c='C0', label='Median')
#
ax1[1, 0].bar(xx_gedi.reshape(NumShots.shape), NumShots, width=0.8, align='center')
ax1[1, 1].bar(xx_atl08.reshape(NumSegments.shape), NumSegments, width=0.8, align='center')
#
ax1[0, 0].set_axisbelow(True)
ax1[0, 0].grid()
ax1[0, 0].yaxis.set_minor_locator(MultipleLocator(5))
ax1[0, 0].legend(loc='lower right', prop={"size": 11.5})
#ax1[0, 0].tick_params(axis='x', which='minor', bottom=False)
#ax1[0, 0].grid(b=True, which='minor', color='grey', linestyle=':', axis='y')
#
ax1[0, 1].set_axisbelow(True)
ax1[0, 1].grid()
ax1[0, 1].yaxis.set_minor_locator(MultipleLocator(5))
ax1[0, 1].legend(loc='lower right', prop={"size": 11.5})
#ax1[0, 1].tick_params(axis='x', which='minor', bottom=False)
#ax1[0, 1].grid(b=True, which='minor', color='grey', linestyle=':', axis='y')
#
ax1[1, 0].set_axisbelow(True)
ax1[1, 0].grid()
ax1[1, 1].set_axisbelow(True)
ax1[1, 1].grid()
#
ax1[0, 0].set_yticks(np.arange(-10, 51, 10))
ax1[0, 0].set_ylim([-15, 50])
ax1[0, 0].set_xlim([0, maxYears+1])
ax1[0, 1].set_yticks(np.arange(-10, 51, 10))
ax1[0, 1].set_ylim([-15, 50])
ax1[0, 1].set_xlim([0, maxYears+1])
#ax1[1, 0].set_ylim([0, 4000])
ax1[1, 0].set_xlim([0, maxYears+1])
ax1[1, 1].set_xlim([0, maxYears+1])
#
ax1[0, 0].set_ylabel('GEDI RH98 [m]')
#ax1[0, 0].set_xlabel('Stand Age [years]')
ax1[0, 1].set_ylabel('ICESat-2 RH98 [m]')
#ax1[0, 1].set_xlabel('Stand Age [years]')
ax1[1, 0].set_ylabel('GEDI Shots (x$10^2$)')
ax1[1, 0].set_xlabel('Stand Age [years]')
ax1[1, 1].set_ylabel('ICESat-2 Segments (x$10^2$)')
ax1[1, 1].set_xlabel('Stand Age [years]')
#
fig1.tight_layout()
#
labels = [item.get_text() for item in ax1[1, 0].get_yticklabels()]
labels2 = [str(int(int(label)/100)) for label in labels]
ax1[1, 0].set_yticklabels(labels2)
#
labels = [item.get_text() for item in ax1[1, 1].get_yticklabels()]
labels2 = [str(int(int(label)/100)) for label in labels]
ax1[1, 1].set_yticklabels(labels2)
#
out_FigDir = r'/mnt/raid/milutin/upScaling/Rondonia/GEDI/Rondonia_L2A_v002/output/directAnalysis/figures_Regrowth_33years/'
outFigName = 'regrowth_Rondonia_33years_GEDI_ICESat2_calibrated_' + my_gedi_name + '_'+ my_ATL08_name + '.png'
plt.savefig(os.path.join(out_FigDir, outFigName), dpi=150)

# remove vertical gap between subplots
# plt.subplots_adjust(hspace=.0)








