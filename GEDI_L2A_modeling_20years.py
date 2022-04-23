# Author:       Milutin Milenkovic
# Copyright:
# Licence:
# ----------------------------------------------------------------------------------------------------------------------
# -- Short Description --
"""
This script fits linear regrowth models over 20-year regrowth period to different GEDI subgroups and provides
an overview figure including GEDI subgroups with different sensitivities, calibrated heights, and non-calibrated heights
"""
# -------- Input --------
# (1) GEDI shots over secondary forest in the Rondônia state, Brazil.
# The input is hardcoded in the 'regrowth_modeling_auxiliary_sctipt/gedi_icesat2_processor' function.
# -------- Output -------
# (1) A figure and statistics reported in Section 4.3.1 (Milenkovic et al. 2022)
# ----------------------------------------------------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
#
from scipy.optimize import curve_fit
#
from regrowth_modeling_auxiliary_sctipt import gedi_icesat2_processor

# #####################################################################################
# parameter description
# #####################################################################################
# NOTE: all processing parameters are boolean type, i.e. True/False
# ------------------
# CALIBRATED_RH98 - if true, GEDI and ICESat-2 height values will be calibrated and calibrated heigths will be added
# as an extra column in the output pandas dataframes.
# ------------------
# SENS_ALG2 -  if true, only sensitivity from the algorithm setting group 2 will be used.
# (The auto-detected sensitivity will be ignored)
# ------------------
# GROUP2_ONLY - if true, only shots from the algorithm setting group 2 will be considered.
# ------------------
# SENS_SUBGROUPS_ONLY - if true, only subgroups with different sensitivity level (ALL, QS90, QS95, QS98, and QS99)
# will be returned. Keep this argument always set as True.
# #####################################################################################
# define different processing scenarios
# #####################################################################################
# my_ParSet_X = [CALIBRATED_RH98, SENS_ALG2, GROUP2_ONLY, SENS_SUBGROUPS_ONLY]
my_ParSet_1 = [True, True, False, True]
my_ParSet_2 = [True, False, False, True]
my_ParSet_3 = [True, False, True, True]
my_ParSet_4 = [False, False, False, True]
#
my_settings = [my_ParSet_1, my_ParSet_2, my_ParSet_3, my_ParSet_4]
my_ProcNames = ['Cal_SensA2', 'Cal', 'Cal_Group2', 'NotCal']

# #######################################################
# define the models
# #######################################################
# define my function for curve_fit
def my_fun1(xx, a, b):
    return a*xx + b

# specify max years to analyse
maxYears = 20
# #####################################################################################
# process the GEDI and ICESat-2 data
# #####################################################################################
# initiate empty dataframes:
df_r2 = pd.DataFrame({'Cal_SensA2': [], 'Cal': [], 'Cal_Group2': [], 'NotCal': []})
df_Intc = pd.DataFrame({'Cal_SensA2': [], 'Cal': [], 'Cal_Group2': [], 'NotCal': []})
df_Slope = pd.DataFrame({'Cal_SensA2': [], 'Cal': [], 'Cal_Group2': [], 'NotCal': []})
df_StdIntc = pd.DataFrame({'Cal_SensA2': [], 'Cal': [], 'Cal_Group2': [], 'NotCal': []})
df_StdSlope = pd.DataFrame({'Cal_SensA2': [], 'Cal': [], 'Cal_Group2': [], 'NotCal': []})
#
# loop trough each processing scenario
for my_setting, my_ProcName in zip(my_settings, my_ProcNames):
    [CALIBRATED_RH98, SENS_ALG2, GROUP2_ONLY, SENS_SUBGROUPS_ONLY] = my_setting
    # obtain the subgroups:
    [gedi_df_subgroups, ignore] = gedi_icesat2_processor(*my_setting)
    # specify the gedi subgroups' names:
    if SENS_SUBGROUPS_ONLY:
        gedi_df_names = ["ALL", "QS90", "QS95", "QS98", "QS99"]
    else:
        gedi_df_names = ["QS90-QPN", "QS90-QPD", "QS90-QPND",
                         "QS90-QCN", "QS90-QCD", "QS90-QCND",
                         "QS90-QPCN", "QS90-QPCD", "QS90"]

    # specify what RH98 value should be considered for regrowth rate:
    if CALIBRATED_RH98:
        height_col = 'rh_98_cal'
    else:
        height_col = 'rh_98'
    # #######################################################
    # Fit the linear model
    # #######################################################
    #
    my_r2 = np.full([len(gedi_df_names)], np.nan)
    my_Intercept = np.full([len(gedi_df_names)], np.nan)
    my_Slope = np.full([len(gedi_df_names)], np.nan)
    my_StdIntc = np.full([len(gedi_df_names)], np.nan)
    my_StdSlope = np.full([len(gedi_df_names)], np.nan)
    # ------------------------------------------------
    # loop trough each sensitivity subgroup (all, QS90, QS95, QS98, and QS99)
    for myIndx in np.arange(len(gedi_df_names)):
        # set the current gdf:
        my_gedi_df = gedi_df_subgroups[myIndx]
        # get the subgroup' data frames:
        my_gedi_df = gedi_df_subgroups[myIndx]
        # group the data by stand age:
        df_standAge_grouped_gedi = my_gedi_df.groupby('forestAge').quantile(0.5)
        # select only first
        df_standAge_grouped_gedi = df_standAge_grouped_gedi[0:maxYears]
        # prepare data for sklearn fit:
        xx = df_standAge_grouped_gedi.index.values.reshape((-1, 1))
        yy = df_standAge_grouped_gedi[height_col].values
        #weights1 = my_gedi_df.groupby('forestAge').count().id.values[0:17]
        weights1 = np.ones(yy.shape[0])
        # ------------------------------------------------
        # fit the weighted regression
        model1 = LinearRegression()
        model1.fit(xx, yy, weights1)
        # get the Standard errors from curve_fit:
        xx1 = df_standAge_grouped_gedi.index.values
        pout, pcov = curve_fit(f=my_fun1, xdata=xx1, ydata=yy)
        perr = np.sqrt(np.diag(pcov))
        #
        my_r2[myIndx] = model1.score(xx, yy, weights1)
        my_Intercept[myIndx] = model1.intercept_
        my_Slope[myIndx] = model1.coef_[0]
        my_StdIntc[myIndx] = perr[1]
        my_StdSlope[myIndx] = perr[0]

    df_Intc[my_ProcName] = my_Intercept
    df_Intc['name'] = gedi_df_names
    #
    df_r2[my_ProcName] = my_r2
    df_r2['name'] = gedi_df_names
    #
    df_Slope[my_ProcName] = my_Slope
    df_Slope['name'] = gedi_df_names
    #
    df_StdSlope[my_ProcName] = my_StdSlope
    df_StdSlope['name'] = gedi_df_names
    #
    df_StdIntc[my_ProcName] = my_StdIntc
    df_StdIntc['name'] = gedi_df_names


# #######################################################
# plot the fitting parameters as bar plots
# #######################################################
# set width of bars
barWidth = 0.14
# set heights of bars
bars1_Cal_a2 = df_r2['Cal_SensA2'].values
bars1_Cal = df_r2['Cal'].values
bars1_Cal_Gr2 = df_r2['Cal_Group2'].values
bars1_noCal = df_r2['NotCal'].values
#
bars2_Cal_a2 = df_Intc['Cal_SensA2'].values
bars2_Cal = df_Intc['Cal'].values
bars2_Cal_Gr2 = df_Intc['Cal_Group2'].values
bars2_noCal = df_Intc['NotCal'].values
#
bars3_Cal_a2 = df_Slope['Cal_SensA2'].values
bars3_Cal = df_Slope['Cal'].values
bars3_Cal_Gr2 = df_Slope['Cal_Group2'].values
bars3_noCal = df_Slope['NotCal'].values

#
# Set position of bar on X axis
r1 = np.arange(len(bars1_Cal_a2))
r2 = [x - barWidth for x in r1]
r3 = [x - barWidth for x in r2]
r4 = [x + barWidth for x in r1]

# Make the plot
fig1, ax1 = plt.subplots(1, 3, figsize=(12, 3.5))
ax1[0].bar(r1, bars1_Cal_a2, color='lightgreen', width=barWidth, edgecolor='black', label='QS9X-S2 Subgroups')
ax1[0].bar(r2, bars1_Cal, color='dimgrey', width=barWidth, edgecolor='black', label='QS9X Subgroups')
#ax1[0].bar(r3, bars1_Cal_Gr2, color='gold', width=barWidth, edgecolor='black', label='QS9X-A2 Subgroups')
ax1[0].bar(r4, bars1_noCal, color='tomato', width=barWidth, edgecolor='black', label='Not-calibrated Heights')
#
ax1[0].set_ylabel('R$^2$  ', rotation=0)
ax1[0].set_ylim([0.6, 1])
# -----------
ax1[1].bar(r1, bars2_Cal_a2, color='lightgreen',
           yerr=2*df_StdIntc['Cal_SensA2'].values, error_kw={'linewidth': 1, 'capsize': 1.5},
           width=barWidth, edgecolor='black', label='QS9X-S2 Subgroups')
ax1[1].bar(r2, bars2_Cal, color='dimgrey',
           yerr=2*df_StdIntc['Cal'].values, error_kw={'linewidth': 1, 'capsize': 1.5},
           width=barWidth, edgecolor='black', label='QS9X Subgroups')
# ax1[1].bar(r3, bars2_Cal_Gr2, color='gold',
#            yerr=2*df_StdIntc['Cal_Group2'].values, error_kw={'linewidth': 1, 'capsize': 1.5},
#            width=barWidth, edgecolor='black', label='QS9X-A2 Subgroups')
ax1[1].bar(r4, bars2_noCal, color='tomato',
           yerr=2*df_StdIntc['NotCal'].values, error_kw={'linewidth': 1, 'capsize': 1.5},
           width=barWidth, edgecolor='black', label='Not-calibrated Heights')
#
ax1[1].set_ylabel('Intercept [m]')
# -----------
ax1[2].bar(r1, bars3_Cal_a2, color='lightgreen',
           yerr=2*df_StdSlope['Cal_SensA2'].values, error_kw={'linewidth': 1, 'capsize': 1.5},
           width=barWidth, edgecolor='black', label='QS9X-S2 Subgroups')
ax1[2].bar(r2, bars3_Cal, color='dimgrey',
           yerr=2*df_StdSlope['Cal'].values, error_kw={'linewidth': 1, 'capsize': 1.5},
           width=barWidth, edgecolor='black', label='QS9X Subgroups')
# ax1[2].bar(r3, bars3_Cal_Gr2, color='gold',
#            yerr=2*df_StdSlope['Cal_Group2'].values, error_kw={'linewidth': 1, 'capsize': 1.5},
#            width=barWidth, edgecolor='black', label='QS9X-A2 Subgroups')
ax1[2].bar(r4, bars3_noCal, color='tomato',
           yerr=2*df_StdSlope['NotCal'].values, error_kw={'linewidth': 1, 'capsize': 1.5},
           width=barWidth, edgecolor='black', label='Not-calibrated Heights')
#
ax1[2].set_ylabel('Slope [m/year]')

# --------------
# Add xticks on the middle of the group bars
#ax1[0].set_xlabel('GEDI Subgroup')
my_ticks = ["ALL", "QS90", "QS95", "QS98", "QS99"]
ax1[0].set_xticks(r1 - barWidth / 2)
ax1[0].set_xticklabels(my_ticks)
#
ax1[1].set_xticks(r1 - barWidth / 2)
ax1[1].set_xticklabels(my_ticks)
#ax1[1].set_xlabel('GEDI Subgroup')
#
ax1[2].set_xticks(r1 - barWidth / 2)
ax1[2].set_xticklabels(my_ticks)
#
ax1[0].set_axisbelow(True)
ax1[0].yaxis.grid()
ax1[1].set_axisbelow(True)
ax1[1].yaxis.grid()
ax1[2].set_axisbelow(True)
ax1[2].yaxis.grid()
#
#ax1[0].text(-1.5, 0.7, r'(a)', fontsize=13)
#ax1[1].text(-1.5, 0, r'(b)', fontsize=13)
#ax1[2].text(-1.5, 0, r'(c)', fontsize=13)
#
ax1[1].legend(ncol=3, loc='upper center', bbox_to_anchor=(-0.6, 1.05, 2.26, .102), mode="expand", borderaxespad=0)
fig1.tight_layout()

# #######################
outFilePath = r'/mnt/raid/milutin/upScaling/Rondonia/GEDI/Rondonia_L2A_v002/output/directAnalysis/figures_regroth_20years/'
out_FigName = 'GEDI_modeling20years_differentProcessingStrategies.png'
plt.savefig(os.path.join(os.path.dirname(outFilePath), out_FigName), dpi=150)