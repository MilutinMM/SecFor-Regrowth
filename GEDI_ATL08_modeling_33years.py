# Author:       Milutin Milenkovic
# Copyright:
# Licence:
# ----------------------------------------------------------------------------------------------------------------------
# -- Short Description --
"""
This script fits non-linear (exponential, spherical, and logarithmic) regrowth models over 33-year regrowth period
to a user-specified GEDI subgroup and ICESat-2 subgroup.
"""
# -------- Input --------
# (1) GEDI shots and ATL08 segments over secondary forest in the Rondônia state, Brazil.
# The input is hardcoded in the 'regrowth_modeling_auxiliary_sctipt/gedi_icesat2_processor' function.
# -------- Output -------
# (1) A figure and statistics reported in Section 4.5 (Milenkovic et al. 2022)
# ----------------------------------------------------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#
from scipy.optimize import curve_fit
#
from regrowth_modeling_auxiliary_sctipt import gedi_icesat2_processor
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

# specify the ICESat-2 subgroups' names:
atl08_df_names = ["SN", "SD", "SND",
                  "WN", "WD", "WDN",
                  "SWN", "SWD", "ALL"]

# #######################################################
# Fit the regrowth exponential and log model and asses the errors
# #######################################################
# specify the GEDI and ICESat-2 subgroups to analyse:
my_gedi_name = 'QS98-S2'
my_ATL08_name = 'SN'
# get the subgroups' data frames:
my_gedi_df = gedi_df_subgroups[gedi_df_names.index(my_gedi_name)]
my_ATL08_df = atl08_df_subgroups[atl08_df_names.index(my_ATL08_name)]
# specify the forest age max years:
maxYears = 33
# specify if calibrated RH98 values should be considered for regrowth rate:
if CALIBRATED_RH98:
    height_col_gedi = 'rh_98_cal'
    height_col_atl08 = 'hCanopy_cal'

# #######################################################
# Fit the regrowth exponential and log model and asses the errors
# #######################################################
# define the model:
def my_exponental(x, a, b, c):
    return a + b * (1 - np.exp(-1 * x / c))

# define the model:
def my_log(x, a, b):
    return a + b*np.log(x)

# define the model:
def my_spherical(x, a, b, c):
    return a + b * (1.5 * (x / c) - 0.5 * ((x / c) ** 3))

# -----------------------------------------------------------------------
# Analyse StandAge
# ----------------------------------------------------------------------
# group the data by stand age:
df_standAge_grouped_gedi = my_gedi_df.groupby('forestAge').quantile(0.5)
df_standAge_grouped_atl08 = my_ATL08_df.groupby('forestAge').quantile(0.5)
# select only first
df_standAge_grouped_gedi = df_standAge_grouped_gedi[0:maxYears]
df_standAge_grouped_atl08 = df_standAge_grouped_atl08[0:maxYears]
# -----------------------------------------------------------------------
# derive the model for combined dataset
# -----------------------------------------------------------------------
# prepare data for sklearn fit:
xx_gedi = df_standAge_grouped_gedi.index.values
yy_gedi = df_standAge_grouped_gedi[height_col_gedi].values
xx_atl08 = df_standAge_grouped_atl08.index.values
yy_atl08 = df_standAge_grouped_atl08[height_col_atl08].values
# weights1 = my_gedi_df.groupby('forestAge').count().id.values[0:17]
# weights1 = np.ones(yy.shape[0])
# -----------------------------------------------------------------------
# initiate empty lists to store stats:
my_a_gedi = []
my_b_gedi = []
my_c_gedi = []
my_a_atl08 = []
my_b_atl08 = []
my_c_atl08 = []
#
my_se_a_gedi = []
my_se_b_gedi = []
my_se_c_gedi = []
my_se_a_atl08 = []
my_se_b_atl08 = []
my_se_c_atl08 = []
#
my_r2_gedi = []
my_r2_atl08 = []
# -----------------------------------------------------------------------
my_models = ['sph', 'exp', 'log']
fig1, ax1 = plt.subplots(2, 3, figsize=(12, 6))
for myIndx in np.arange(len(my_models)):
    #
    model = my_models[myIndx]
    # get the coefficients and covariance matrix:
    if model == 'exp':
        pout_gedi, pcov_gedi = curve_fit(f=my_exponental, xdata=xx_gedi, ydata=yy_gedi)
        pout_atl08, pcov_atl08 = curve_fit(f=my_exponental, xdata=xx_atl08, ydata=yy_atl08)
    elif model == 'log':
        pout_gedi, pcov_gedi = curve_fit(f=my_log, xdata=xx_gedi, ydata=yy_gedi)
        pout_atl08, pcov_atl08 = curve_fit(f=my_log, xdata=xx_atl08, ydata=yy_atl08)
    elif model == 'sph':
        pout_gedi, pcov_gedi = curve_fit(f=my_spherical, xdata=xx_gedi, ydata=yy_gedi)
        pout_atl08, pcov_atl08 = curve_fit(f=my_spherical, xdata=xx_atl08, ydata=yy_atl08)
    # get the standard errors of the parameters (1 sigma error):
    perr_gedi = np.sqrt(np.diag(pcov_gedi))
    perr_atl08 = np.sqrt(np.diag(pcov_atl08))
    #
    my_se_a_gedi.append(perr_gedi[0])
    my_se_b_gedi.append(perr_gedi[1])
    if model == 'log':
        my_se_c_gedi.append(np.nan)
    else:
        my_se_c_gedi.append(perr_gedi[2])
    #
    my_se_a_atl08.append(perr_atl08[0])
    my_se_b_atl08.append(perr_atl08[1])
    if model == 'log':
        my_se_c_atl08.append(np.nan)
    else:
        my_se_c_atl08.append(perr_atl08[2])
    # -----------------------------------------------------------------------
        #
    my_a_gedi.append(pout_gedi[0])
    my_b_gedi.append(pout_gedi[1])
    if model == 'log':
        my_c_gedi.append(np.nan)
    else:
        my_c_gedi.append(pout_gedi[2])
    #
    my_a_atl08.append(pout_atl08[0])
    my_b_atl08.append(pout_atl08[1])
    if model == 'log':
        my_c_atl08.append(np.nan)
    else:
        my_c_atl08.append(pout_atl08[2])
    # -----------------------------------------------------------------------
    # get the prediction and residuals:
    if model == 'exp':
        y_hat_gedi = my_exponental(xx_gedi, *pout_gedi)
        y_hat_atl08 = my_exponental(xx_atl08, *pout_atl08)
    elif model == 'log':
        y_hat_gedi = my_log(xx_gedi, *pout_gedi)
        y_hat_atl08 = my_log(xx_atl08, *pout_atl08)
    elif model == 'sph':
        y_hat_gedi = my_spherical(xx_gedi, *pout_gedi)
        y_hat_gedi[xx_gedi > pout_gedi[2]] = pout_gedi[0] + pout_gedi[1]
        #
        y_hat_atl08 = my_spherical(xx_atl08, *pout_atl08)
        y_hat_atl08[xx_atl08 > pout_atl08[2]] = pout_atl08[0] + pout_atl08[1]

    #
    my_res_gedi = yy_gedi - y_hat_gedi
    my_res_atl08 = yy_atl08 - y_hat_atl08
    # sum of squared residuals:
    ss_res_e_gedi = np.sum(my_res_gedi ** 2)
    ss_tot_gedi = np.sum((yy_gedi - np.mean(yy_gedi)) ** 2)
    #
    ss_res_e_atl08 = np.sum(my_res_atl08 ** 2)
    ss_tot_atl08 = np.sum((yy_atl08 - np.mean(yy_atl08)) ** 2)
    # calculate the R2
    r2_gedi = 1 - ss_res_e_gedi / ss_tot_gedi
    r2_atl08 = 1 - ss_res_e_atl08 / ss_tot_atl08
    my_r2_gedi.append(r2_gedi)
    my_r2_atl08.append(r2_atl08)
    # #######################################################
    # plot the exponential, or log model
    # #######################################################
    ax1[0][myIndx].scatter(xx_gedi, yy_gedi, s=15, color='grey')
    ax1[1][myIndx].scatter(xx_atl08, yy_atl08, s=15, color='grey')
    if model == 'exp':
        ax1[0][myIndx].plot(xx_gedi, y_hat_gedi, 'r', lw=0.75, label=r'$a+b\cdot(1-e^{-\frac{x}{c}})$')
        ax1[1][myIndx].plot(xx_atl08, y_hat_atl08, 'r', lw=0.75, label=r'$a+b\cdot(1-e^{-\frac{x}{c}})$')
        #ax1[0][myIndx].plot(xx_gedi, y_hat_gedi, 'r', lw=0.75, label=r'EXP Model')
        #ax1[1][myIndx].plot(xx_atl08, y_hat_atl08, 'r', lw=0.75, label=r'EXP Model')
    elif model == 'log':
        #ax1[0][myIndx].plot(xx_gedi, y_hat_gedi, 'r', lw=0.75, label=r'LOG Model')
        #ax1[1][myIndx].plot(xx_atl08, y_hat_atl08, 'r', lw=0.75, label=r'LOG Model')
        ax1[0][myIndx].plot(xx_gedi, y_hat_gedi, 'r', lw=0.75, label=r'$a+b\cdot \log (x)$')
        ax1[1][myIndx].plot(xx_atl08, y_hat_atl08, 'r', lw=0.75, label=r'$a+b\cdot \log (x)$')
    elif model == 'sph':
        #ax1[0][myIndx].plot(xx_gedi, y_hat_gedi, 'r', lw=0.75, label=r'SPH Model')
        #ax1[1][myIndx].plot(xx_atl08, y_hat_atl08, 'r', lw=0.75, label=r'SPH Model')
        ax1[0][myIndx].plot(xx_gedi, y_hat_gedi, 'r', lw=0.75, label=r'$a+b\cdot(1.5\cdot\frac{x}{c} - 0.5\cdot(\frac{x}{c})³)$')
        ax1[1][myIndx].plot(xx_atl08, y_hat_atl08, 'r', lw=0.75, label=r'$a+b\cdot(1.5\cdot\frac{x}{c} - 0.5\cdot(\frac{x}{c})³)$')
    #
    #ax1[myIndx].set_xlabel('Stand Age [years]')
    ax1[0][myIndx].set_xlim([0, maxYears+1])
    ax1[1][myIndx].set_xlim([0, maxYears + 1])
    #
    ax1[0][myIndx].set_title('{} Model; {}'.format(my_models[myIndx].upper(), my_gedi_name))
    ax1[1][myIndx].set_title('{} Model; {}'.format(my_models[myIndx].upper(), my_ATL08_name))
    #
    ax1[0][myIndx].plot([], [], ' ', label='R$^2$ = {}'.format(str(round(r2_gedi, 2))))
    ax1[1][myIndx].plot([], [], ' ', label='R$^2$ = {}'.format(str(round(r2_atl08, 2))))
    # ax1[myIndx].plot([], [], ' ', label='R$^2$ = {} ; $a$={}'.format(str(round(my_r2, 2)),str(round(pout[0], 2))))
    # if model == 'exp' or model == 'sph':
    #     ax1[myIndx].plot([], [], ' ', label='$b$ = {} ; $c$ = {}'.format(str(round(pout[1], 2)), str(round(pout[2], 2))))
    # else:
    #     ax1[myIndx].plot([], [], ' ', label='$b$ = {}'.format(str(round(pout[1], 2))))
    ax1[0][myIndx].legend(loc='lower right', prop={"size": 11.5})
    ax1[1][myIndx].legend(loc='lower right' , prop={"size": 11.5})
    #
    ax1[0][myIndx].set_ylim([0, 25])
    ax1[1][myIndx].set_ylim([0, 25])
    #
    ax1[0][myIndx].grid()
    ax1[1][myIndx].grid()
    #
    ax1[1][myIndx].set_xlabel('Stand Age [years]')
    #ax1[0][myIndx].set_xlabel('')

ax1[0][0].set_ylabel('Median of GEDI RH98 [m]')
ax1[1][0].set_ylabel('Median of ICESat-2 RH98 [m]')
fig1.tight_layout()

outFilePath = r'/mnt/raid/milutin/upScaling/Rondonia/GEDI/Rondonia_L2A_v002/output/directAnalysis/figures_modeling_33years/'
out_FigName = 'Modeling_GEDI_ICESat2_33years.png'
plt.savefig(os.path.join(os.path.dirname(outFilePath), out_FigName), dpi=150)


# prepare a table to save:
my_stats_df_gedi = pd.DataFrame(list(zip(my_models, my_r2_gedi, my_a_gedi, my_b_gedi, my_c_gedi, my_se_a_gedi, my_se_b_gedi, my_se_c_gedi)),
                           columns=['Model', 'R2', 'a', 'b', 'c', 'SE_a', 'SE_b', 'SE_c'])

my_stats_df_atl08 = pd.DataFrame(list(zip(my_models, my_r2_atl08, my_a_atl08, my_b_atl08, my_c_atl08, my_se_a_atl08, my_se_b_atl08, my_se_c_atl08)),
                           columns=['Model', 'R2', 'a', 'b', 'c', 'SE_a', 'SE_b', 'SE_c'])


# save the statistics:
outFilePath = r'/mnt/raid/milutin/upScaling/Rondonia/GEDI/Rondonia_L2A_v002/output/directAnalysis/figures_modeling_33years/'
out_htmlName_gedi = height_col_gedi + '_modelParameters_GEDI.html'
out_htmlName_atl08 = height_col_atl08 + '_modelParameters_ATL08.html'
my_stats_df_gedi.to_html(os.path.join(os.path.dirname(outFilePath), out_htmlName_gedi))
my_stats_df_atl08.to_html(os.path.join(os.path.dirname(outFilePath), out_htmlName_atl08))
# write to a excel file:
my_stats_df_gedi.to_excel(os.path.join(os.path.dirname(outFilePath), out_htmlName_gedi[:-4] + 'xlsx'), index_label=False)
my_stats_df_atl08.to_excel(os.path.join(os.path.dirname(outFilePath), out_htmlName_atl08[:-4] + 'xlsx'), index_label=False)

