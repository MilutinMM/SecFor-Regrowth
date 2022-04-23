# Author:       Milutin Milenkovic
# Copyright:
# Licence:
# ----------------------------------------------------------------------------------------------------------------------
# -- Short Description --
"""
This script fits linear regrowth models over 20-year regrowth period to a specific ICESat-2 subgroups.
"""
# -------- Input --------
# (1) ATL08 segments over secondary forest in the Rondônia state, Brazil.
# The input is hardcoded in the 'regrowth_modeling_auxiliary_sctipt/gedi_icesat2_processor' function.
# -------- Output -------
# (1) figures and statistics reported in Section 4.3.2 (Milenkovic et al. 2022)
# ----------------------------------------------------------------------------------------------------------------------

import os
import numpy as np
import geopandas as gpd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from scipy.optimize import curve_fit
#
from regrowth_modeling_auxiliary_sctipt import gedi_icesat2_processor
# ------------------------------------
# read the ATL08 data frame
# ------------------------------------
outFilePath = r'/mnt/raid/milutin/upScaling/Rondonia/ICESat2/ATL08_v003/directAnalysis/ATL08_gdf.json'
#
atl08_gdf = gpd.read_file(outFilePath)

# #####################################################################################
# process the GEDI and ICESat-2 data
# #####################################################################################
# specify that calibrated hCanopy values should be calculated and added as extra column:
CALIBRATED_RH98 = True
# obtain the subgroups:
# !!! Note !!! setting of gedi-related parameters is irrelevant (SENS_ALG2, GROUP2_ONLY, SENS_SUBGROUPS_ONLY)
[ignore, atl08_df_subgroups] = gedi_icesat2_processor(CALIBRATED_RH98, SENS_ALG2=False, GROUP2_ONLY=False, SENS_SUBGROUPS_ONLY=False)
# specify the ICESat-2 subgroups' names:
atl08_df_names = ["SN", "SD", "SND",
                  "WN", "WD", "WND",
                  "SWN", "SWD", "ALL"]

# ##############################################################################################
# set the parameters
# ##############################################################################################
# ensure that calibrated height is used
if CALIBRATED_RH98:
    height_col = 'hCanopy_cal'

# specify max forest age year to use for plotting
maxYears = 20

# ##############################################################################################
# plot regrowth rates
# ##############################################################################################
# define my function for curve_fit
def my_fun1(xx, a, b):
    return a*xx + b
# ---------------------------------
#
sns.set(color_codes=True)
sns.set_style('white')
#mpl.style.use('default')
fig1, ax1 = plt.subplots(3, 3, figsize=(12, 9), sharex=True)
for myIndx in np.arange(len(atl08_df_names)):
    # -----------------------------------------------------------------------
    # set the parameters
    # -----------------------------------------------------------------------
    #
    maxYears = 20
    #
    row, col = np.unravel_index(myIndx, (3, 3))
    # set the current gdf:
    my_ATL08_df = atl08_df_subgroups[myIndx]
    # -----------------------------------------------------------------------
    # Analyse StandAge
    # ----------------------------------------------------------------------
    # group the data by stand age:
    df_standAge_grouped = my_ATL08_df.groupby('forestAge').quantile(0.5)
    # select only first
    df_standAge_grouped = df_standAge_grouped[0:maxYears]
    # -----------------------------------------------------------------------
    # derive the model for combined dataset
    # -----------------------------------------------------------------------
    # prepare data for sklearn fit:
    xx = df_standAge_grouped.index.values.reshape((-1, 1))
    yy = df_standAge_grouped[height_col].values
    #weights1 = my_ATL08_df.groupby('forestAge').count().id.values[0:17]
    weights1 = np.ones(yy.shape[0])
    # fit the weighted regression
    model1 = LinearRegression()
    model1.fit(xx, yy, weights1)
    # get the Standard errors from curve_fit:
    xx1 = df_standAge_grouped.index.values
    pout, pcov = curve_fit(f=my_fun1, xdata=xx1, ydata=yy)
    perr = np.sqrt(np.diag(pcov))
    # -----------------------------------------------------------------------
    # plot with regression line
    sns.regplot(data=df_standAge_grouped, x=df_standAge_grouped.index, y=height_col, color='r', marker='o',
                line_kws={'label': "y={0:.2f}x+{1:.2f}".format(model1.coef_[0], model1.intercept_), 'lw': 1.5},
                scatter_kws={'s': 10}, ax=ax1[row][col])
    ax1[row][col].legend(loc='upper left')
    ax1[row][col].text(0.55*maxYears, 5.5, "$R^2$ = {0: .2f}".format(model1.score(xx, yy, weights1)), fontsize=11)
    ax1[row][col].text(0.55*maxYears, 3, "Segments = {}".format(str(my_ATL08_df[my_ATL08_df.forestAge <= maxYears].shape[0])), fontsize=11)
    ax1[row][col].text(1, 26, r"$\sigma_{\hat{\beta}_1}$ = %.2f ; $\sigma_{\hat{\beta}_0}$ = %.2f" % (perr[0], perr[1]), fontsize=11)
    ax1[row][col].set_title('{}'.format(atl08_df_names[myIndx]))
    #
    ax1[row][col].set_xticks(np.arange(1, maxYears+1))
    ax1[row][col].set_xlim([0.5, maxYears + 0.5])
    ax1[row][col].set_xticks([1, 5, 10, 15, 20])
    ax1[row][col].set_ylim([0, 35])
    ax1[row][col].set_xlabel('')
    ax1[row][col].set_ylabel('')
    #
    ax1[row][col].set_axisbelow(True)
    ax1[row][col].grid()

fig1.supxlabel('Stand Age [years]')
fig1.supylabel('Median of ICESat-2 RH98 [m]')
fig1.tight_layout()


outFilePath = r'/mnt/raid/milutin/upScaling/Rondonia/ICESat2/ATL08_v003/directAnalysis/figures_regroth_20years/'
out_FigName = height_col + '_regrowth' + '_eroded.png'
plt.savefig(os.path.join(os.path.dirname(outFilePath), out_FigName), dpi=150)

# ##############################################################################################
# plot height histograms
# ##############################################################################################
#sns.set(color_codes=True)
mpl.style.use('default')
my_bins = np.arange(-15, 76, 1)
my_bins2 = np.arange(0, 150, 5)
#
fig1, ax1 = plt.subplots(2, 2, figsize=(7, 6))
#
atl08_df = atl08_df_subgroups[atl08_df_names.index('ALL')]
ax1[0][0].hist(atl08_df[height_col].values, bins=my_bins, edgecolor='black', facecolor='lightgreen', alpha=1, label='ALL')
#
atl08_S = atl08_df_subgroups[atl08_df_names.index('SND')]
ax1[0][0].hist(atl08_S[height_col].values, bins=my_bins, edgecolor='black', facecolor='tomato', alpha=1, label='SND')
#
atl08_SN = atl08_df_subgroups[atl08_df_names.index('SN')]
ax1[0][0].hist(atl08_SN[height_col].values, bins=my_bins, edgecolor='black', facecolor='dimgrey', alpha=1, label='SN')
#
atl08_SD = atl08_df_subgroups[atl08_df_names.index('SD')]
ax1[0][0].hist(atl08_SD[height_col].values, bins=my_bins, edgecolor='black', facecolor='lightgrey', alpha=0.5, label='SD')
ax1[0][0].set_xlim([-15, 50])
ax1[0][0].set_xticks(np.arange(-20, 80, 10))
ax1[0][0].legend()
#
ax1[0][1].hist(atl08_df[height_col].values, bins=my_bins, edgecolor='black', facecolor='lightgreen', alpha=1, label='ALL')
#
atl08_W = atl08_df_subgroups[atl08_df_names.index('WND')]
ax1[0][1].hist(atl08_W[height_col].values, bins=my_bins, edgecolor='black', facecolor='gold', alpha=1, label='WND')
#
atl08_WN = atl08_df_subgroups[atl08_df_names.index('WN')]
ax1[0][1].hist(atl08_WN[height_col].values, bins=my_bins, edgecolor='black', facecolor='dimgrey', alpha=1, label='WN')
#
atl08_WD = atl08_df_subgroups[atl08_df_names.index('WD')]
ax1[0][1].hist(atl08_WD[height_col].values, bins=my_bins, edgecolor='black', facecolor='lightgrey', alpha=0.75, label='WD')
ax1[0][1].set_xlim([-15, 50])
ax1[0][1].set_xticks(np.arange(-20, 80, 10))
ax1[0][1].legend()
#
ax1[1][0].hist(atl08_df['caPhoNum'].values, bins=my_bins2, edgecolor='black', facecolor='lightgreen', alpha=1, label='ALL')
ax1[1][0].hist(atl08_S['caPhoNum'].values, bins=my_bins2, edgecolor='black', facecolor='tomato', alpha=1, label='SND')
ax1[1][0].hist(atl08_SN['caPhoNum'].values, bins=my_bins2, edgecolor='black', facecolor='dimgrey', alpha=1, label='SN')
ax1[1][0].hist(atl08_SD['caPhoNum'].values, bins=my_bins2, edgecolor='black', facecolor='lightgrey', alpha=0.75, label='SD')
ax1[1][0].set_xlim([0, 140])
ax1[1][0].set_xticks([0, 50, 100, 140])
ax1[1][0].legend(loc='upper right')
#
ax1[1][1].hist(atl08_df['caPhoNum'].values, bins=my_bins2, edgecolor='black', facecolor='lightgreen', alpha=1, label='ALL')
ax1[1][1].hist(atl08_W['caPhoNum'].values, bins=my_bins2, edgecolor='black', facecolor='gold', alpha=1, label='WND')
ax1[1][1].hist(atl08_WN['caPhoNum'].values, bins=my_bins2, edgecolor='black', facecolor='dimgrey', alpha=1, label='WN')
ax1[1][1].hist(atl08_WD['caPhoNum'].values, bins=my_bins2, edgecolor='black', facecolor='lightgrey', alpha=0.75, label='WD')
ax1[1][1].set_xlim([0, 140])
ax1[1][1].set_xticks([0, 50, 100, 140])
ax1[1][1].legend(loc='upper right')
#
labels = [item.get_text() for item in ax1[0][0].get_yticklabels()]
labels2 = [str(int(int(label)/10)) for label in labels]
ax1[0][0].set_yticklabels(labels2)
#
labels = [item.get_text() for item in ax1[0][1].get_yticklabels()]
labels2 = [str(int(int(label)/10)) for label in labels]
ax1[0][1].set_yticklabels(labels2)
#
labels = [item.get_text() for item in ax1[1][0].get_yticklabels()]
labels2 = [str(int(int(label)/10)) for label in labels]
ax1[1][0].set_yticklabels(labels2)
#
labels = [item.get_text() for item in ax1[1][1].get_yticklabels()]
labels2 = [str(int(int(label)/10)) for label in labels]
ax1[1][1].set_yticklabels(labels2)
#
ax1[0][0].set_xlabel('ICESat-2 RH98 [m]')
ax1[0][1].set_xlabel('ICESat-2 RH98 [m]')
ax1[0][0].set_ylabel('ICESat-2 Segments (x10)')
#
ax1[1][0].set_xlabel('Canopy Photons')
ax1[1][1].set_xlabel('Canopy Photons')
ax1[1][0].set_ylabel('ICESat-2 Segments (x10)')
#
ax1[0][0].set_xlim([-15, 50])
ax1[0][1].set_xlim([-15, 50])
#
plt.tight_layout()

outFilePath = r'/mnt/raid/milutin/upScaling/Rondonia/ICESat2/ATL08_v003/directAnalysis/figures_regroth_20years/'
out_FigName = height_col + '_histograms_RH98_and_CanopyPhotons.png'
plt.savefig(os.path.join(os.path.dirname(outFilePath), out_FigName), dpi=150)
