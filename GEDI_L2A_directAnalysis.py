# Author:       Milutin Milenkovic
# Copyright:
# Licence:
# ----------------------------------------------------------------------------------------------------------------------
# -- Short Description --
"""
This script fits linear regrowth models over 20-year regrowth period to a specific GEDI subgroups.
"""
# -------- Input --------
# (1) GEDI shots over secondary forest in the Rondônia state, Brazil.
# The input is hardcoded in the 'regrowth_modeling_auxiliary_sctipt/gedi_icesat2_processor' function.
# -------- Output -------
# (1) A figure and statistics reported in Section 4.3.1 (Milenkovic et al. 2022)
# ----------------------------------------------------------------------------------------------------------------------

import os
import geopandas as gpd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#
from regrowth_modeling_auxiliary_sctipt import gedi_icesat2_processor

# ------------------------------------
#  read the gedi geo data frame
# ------------------------------------
outFilePath = r'/mnt/raid/milutin/upScaling/Rondonia/GEDI/Rondonia_L2A_v002/output/directAnalysis/gedi_L2A_gdf_sens_a2.json'
gedi_gdf = gpd.read_file(outFilePath)
# gedi_gdf.plot()

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
[gedi_df_subgroups, ignore] = gedi_icesat2_processor(CALIBRATED_RH98, SENS_ALG2, GROUP2_ONLY, SENS_SUBGROUPS_ONLY)
# specify the gedi subgroups' names:
if SENS_SUBGROUPS_ONLY:
    if GROUP2_ONLY:
        gedi_df_names = ["ALL-A2", "QS90-A2", "QS95-A2", "QS98-A2", "QS99-A2"]
    elif SENS_ALG2:
        gedi_df_names = ["ALL-S2", "QS90-S2", "QS95-S2", "QS98-S2", "QS99-S2"]
    else:
        gedi_df_names = ["ALL", "QS90", "QS95", "QS98", "QS99"]
else:
    gedi_df_names = ["QS90-QPN", "QS90-QPD", "QS90-QPND",
                     "QS90-QCN", "QS90-QCD", "QS90-QCND",
                     "QS90-QPCN", "QS90-QPCD", "QS90"]

# ##############################################################################################
# set the parameters
# ##############################################################################################
# ensure that calibrated height is used
if CALIBRATED_RH98:
    height_col = 'rh_98_cal'
else:
    height_col = 'rh_98'

# specify max forest age year to use for plotting
maxYears = 20
# ##############################################################################################
# plot regrowth rates
# ##############################################################################################
# define my function for curve_fit
def my_fun1(xx, a, b):
    return a*xx + b
# ---------------------------------
sns.set(color_codes=True)
sns.set_style('white')
#mpl.style.use('default')
fig1, ax1 = plt.subplots(2, 3, figsize=(12, 6))
for myIndx in np.arange(len(gedi_df_names)):
    # set the subplot row and column:
    row, col = np.unravel_index(myIndx, (3, 3))
    # set the current gdf:
    my_gedi_df = gedi_df_subgroups[myIndx]
    # -----------------------------------------------------------------------
    # Analyse StandAge
    # ----------------------------------------------------------------------
    # group the data by stand age:
    df_standAge_grouped = my_gedi_df.groupby('forestAge').quantile(0.5)
    # select only first
    df_standAge_grouped = df_standAge_grouped[0:maxYears]
    # -----------------------------------------------------------------------
    # derive the model for combined dataset
    # -----------------------------------------------------------------------
    # prepare data for sklearn fit:
    xx = df_standAge_grouped.index.values.reshape((-1, 1))
    yy = df_standAge_grouped[height_col].values
    #weights1 = my_gedi_df.groupby('forestAge').count().id.values[0:17]
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
    ax1[row][col].text(0.6*maxYears, 5.5, "$R^2$ = {0: .2f}".format(model1.score(xx, yy, weights1)), fontsize=11)
    ax1[row][col].text(0.6*maxYears, 3, "Shots = {}".format(str(my_gedi_df[my_gedi_df.forestAge <= maxYears].shape[0])), fontsize=11)
    ax1[row][col].text(1, 18, r"$\sigma_{\hat{\beta}_1}$ = %.2f ; $\sigma_{\hat{\beta}_0}$ = %.2f" %(perr[0], perr[1]), fontsize=11)
    ax1[row][col].set_title('{}'.format(gedi_df_names[myIndx]))
    #
    ax1[row][col].set_xticks([1, 5, 10, 15, 20])
    ax1[row][col].set_xlim([0.5, maxYears + 0.5])
    ax1[row][col].set_ylim([0, 25])
    #
    ax1[row][col].set_xlabel('Stand Age [years]')
    ax1[row][col].set_ylabel('')
    #
    ax1[row][col].set_axisbelow(True)
    ax1[row][col].grid()

ax1[0][0].set_ylabel('Median of GEDI RH98')
ax1[1][0].set_ylabel('Median of GEDI RH98')
# add the histograms
#mpl.style.use('default')
my_bins = np.arange(-20, 76, 1)
#
gedi_df = gedi_df_subgroups[gedi_df_names.index('ALL-S2')]
ax1[1][2].hist(gedi_df[height_col].values, bins=my_bins, edgecolor='black', facecolor='seagreen', alpha=1, label='ALL')
#
gedi_df_Q = gedi_df_subgroups[gedi_df_names.index('QS90-S2')]
ax1[1][2].hist(gedi_df_Q[height_col].values, bins=my_bins, edgecolor='black', facecolor='lightgreen', alpha=1, label='QS90')
#
gedi_df_QS95 = gedi_df_subgroups[gedi_df_names.index('QS95-S2')]
ax1[1][2].hist(gedi_df_QS95[height_col].values, bins=my_bins, edgecolor='black', facecolor='dodgerblue', alpha=1, label='QS95')
#
gedi_df_QS98 = gedi_df_subgroups[gedi_df_names.index('QS98-S2')]
ax1[1][2].hist(gedi_df_QS98[height_col].values, bins=my_bins, edgecolor='black', facecolor='skyblue', alpha=1, label='QS98')
#
gedi_df_QS99 = gedi_df_subgroups[gedi_df_names.index('QS99-S2')]
ax1[1][2].hist(gedi_df_QS99[height_col].values, bins=my_bins, edgecolor='black', facecolor='cyan', alpha=1, label='QS99')
#
ax1[1][2].set_xlabel('GEDI RH98 [m]')
ax1[1][2].set_xticks(np.arange(-10, 76, 10))
ax1[1][2].set_xlim([-10, 50])
ax1[1][2].set_ylabel('GEDI Shots (x$10^2$)')
ax1[1][2].legend()
#
ax1[1][2].set_yticks(np.arange(0, 1400, 200))
labels = [item.get_text() for item in ax1[1][2].get_yticklabels()]
labels2 = [str(int(int(label)/100)) for label in labels]
ax1[1][2].set_yticklabels(labels2)
#
ax1[1][2].set_axisbelow(True)
ax1[1][2].grid()
#
fig1.tight_layout()
# #######################
outFilePath = r'/mnt/raid/milutin/upScaling/Rondonia/GEDI/Rondonia_L2A_v002/output/directAnalysis/figures_regroth_20years/'
out_FigName = height_col + '_regrowthSensitiviy_calibrated_eroded_sens_a2.png'
plt.savefig(os.path.join(os.path.dirname(outFilePath), out_FigName), dpi=150)
