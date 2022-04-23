

def gedi_icesat2_processor(CALIBRATED_RH98=True, SENS_ALG2 = True, GROUP2_ONLY = False, SENS_SUBGROUPS_ONLY=True):
    '''
    This script pre-process GEDI and ICESat-2 data to prepare different subgroups for further analysis.

    :param CALIBRATED_RH98: [True/False] if True, additional column will be calculated with calibrated heights
    :param SENS_ALG2: [True/False] if True, only sensitivity from algorithm 2 and auto derived heights will be used:
    :param GROUP2_ONLY: [True/False] if True, only shots with auto-selected algorithm 2 will be used
    :param SENS_SUBGROUPS_ONLY: [True/False] if True, returns only GEDI subgroups with different sensitivity levels
    :return: a list with elements corresponding to GEDI and ICESat-2 subroups
    '''
    import geopandas as gpd
    import pandas as pd
    import numpy as np
    # #####################################################################################
    # read GEDI shots and ATL08 segments:
    # #####################################################################################
    gediFilePath = r'/mnt/raid/milutin/upScaling/Rondonia/GEDI/Rondonia_L2A_v002/output/directAnalysis/gedi_L2A_gdf_sens_a2.json'
    gedi_gdf = gpd.read_file(gediFilePath)
    #
    atl08FilePath = r'/mnt/raid/milutin/upScaling/Rondonia/ICESat2/ATL08_v003/directAnalysis/ATL08_gdf.json'
    atl08_gdf = gpd.read_file(atl08FilePath)
    # #####################################################################################
    # read the calibration models to ALS
    # #####################################################################################
    # read GEDI RH98 calibration models
    if GROUP2_ONLY:
        # group 2 calibration models
        calModelSensFile = r'/mnt/ssd/milutin/Para_upScaling/Validation_GEDI_ICESat2/GEDI_Para_v002_L2A/output/rh_98_stats_hDiff_Para_MG_Sensitivity_group2.xlsx'
        calModelsQS90File = r'/mnt/ssd/milutin/Para_upScaling/Validation_GEDI_ICESat2/GEDI_Para_v002_L2A/output/rh_98_stats_hDiff_Para_MG_QS90_group2.xlsx'
    elif SENS_ALG2:
        # calibration models for heights from algorithm 1 and 2, but only with sensitivities from algorithm 2
        calModelSensFile = r'/mnt/ssd/milutin/Para_upScaling/Validation_GEDI_ICESat2/GEDI_Para_v002_L2A/output_sens_a2/rh_98_stats_hDiff_Para_MG_Sensitivity_sens_2.xlsx'
        calModelsQS90File = r'/mnt/ssd/milutin/Para_upScaling/Validation_GEDI_ICESat2/GEDI_Para_v002_L2A/output_sens_a2/rh_98_stats_hDiff_Para_MG_QS90_sens_a2.xlsx'
    else:
        # groups 1 and 2 calibration models
        calModelSensFile = r'/mnt/ssd/milutin/Para_upScaling/Validation_GEDI_ICESat2/GEDI_Para_v002_L2A/output/rh_98_stats_hDiff_Para_MG_deFor2018_2019.xlsx'
        calModelsQS90File = r'/mnt/ssd/milutin/Para_upScaling/Validation_GEDI_ICESat2/GEDI_Para_v002_L2A/output/rh_98_stats_hDiff_Para_MG_QS90_deFor2018_2019.xlsx'

    # read models:
    calModels_Sens = pd.read_excel(calModelSensFile)
    calModels_QS90 = pd.read_excel(calModelsQS90File)

    # read ATL08 calibration models
    calModelFile = r'/mnt/ssd/milutin/Para_upScaling/Validation_GEDI_ICESat2/ICESat2_Para/ATL08_validation/hCanopy_stats_hDiff_Para_MG_canopyPhoton_lt140.xlsx'
    calModels = pd.read_excel(calModelFile)
    # set for NaNs: slope and intercept 1 and 0 values, respectively:
    calModels['Slope'] = calModels['Slope'] .fillna(1.)
    calModels['Intercept'] = calModels['Intercept'] .fillna(0.)
    # ############################################
    # pre-processing GEDI:
    # ############################################
    # specify which sensitivity column to use:
    if SENS_ALG2:
        my_Sensitivity = 'geolocation_sensitivity_a2'
    else:
        my_Sensitivity = 'sensitivity'
    # select only shots inside eroded forest age (to minimize geolocation/border errors):
    gedi_gdf = gedi_gdf[gedi_gdf['eroded_age'] == 1]
    # select the group 2
    if GROUP2_ONLY:
        gedi_gdf = gedi_gdf[gedi_gdf['selected_algorithm'] == 2]
    # specify the height column
    height_col_gedi = 'rh_98'
    # convert back to data frame
    gedi_df = pd.DataFrame(gedi_gdf)
    # drop geometry to apply groupby
    gedi_df.drop('geometry', axis='columns', inplace=True)
    # -------------------------------------
    # filter out height equal to zero and heights above 75m
    gedi_df = gedi_df[gedi_df[height_col_gedi] != 0]
    gedi_df = gedi_df[gedi_df[height_col_gedi] <= 75]
    # filter out points with sensitivity outside 0, 1 interval
    gedi_df = gedi_df[(gedi_df['sensitivity'] >= 0) & (gedi_df['sensitivity'] <= 1)]
    # ############################################
    # pre-processing ICSat-2:
    # ############################################
    # select only shots inside eroded forest age (to minimize geolocation/border errors):
    atl08_gdf = atl08_gdf[atl08_gdf['eroded_age'] == 1]
    # select height column
    height_col_atl08 = 'hCanopy'
    hPerc = 98
    # convert back to data frame
    atl08_df = pd.DataFrame(atl08_gdf)
    # drop geometry to apply groupby
    atl08_df.drop('geometry', axis='columns', inplace=True)
    # -------------------------------------
    # filter out height equal to zero and heights above 75m
    atl08_df = atl08_df[atl08_df[height_col_atl08] != 0]
    atl08_df = atl08_df[atl08_df[height_col_atl08] <= 75]
    # filter out points with canopy photons larger than 140
    atl08_df = atl08_df[atl08_df['caPhoNum'] < 140]
    # #################################
    # ATL filtering
    # #################################
    # select strong and weak points
    atl08_S = atl08_df[atl08_df.beamStrength == 1]
    atl08_W = atl08_df[atl08_df.beamStrength == 0]
    # select night and day points
    atl08_SN = atl08_S[atl08_S.nightFlag == 1]
    atl08_SD = atl08_S[atl08_S.nightFlag == 0]
    #
    atl08_WN = atl08_W[atl08_W.nightFlag == 1]
    atl08_WD = atl08_W[atl08_W.nightFlag == 0]
    # select day-night in strong+weak points
    atl08_SWN = atl08_SN.append(atl08_WN)
    atl08_SWD = atl08_SD.append(atl08_WD)
    # ##############################################################################################
    # Calculate Calibrate ATL08 RH98 heights
    # ##############################################################################################
    if CALIBRATED_RH98:
        # make a list of gdf-s:
        gdf_list = [atl08_SN, atl08_SD, atl08_S,
                    atl08_WN, atl08_WD, atl08_W,
                    atl08_SWN, atl08_SWD, atl08_df]
        indexListModels = np.arange(9)
        for my_df, myInd in zip(gdf_list, indexListModels):
            b = calModels.at[myInd, 'Slope']
            a = calModels.at[myInd, 'Intercept']
            my_df['hCanopy_cal'] = my_df['hCanopy'].apply(lambda x: (x-a)/b)
    # #################################
    # GEDI filtering - set I
    # #################################
    # apply quality flags
    #gedi_df_Q = gedi_df[gedi_df.degrade_flag == 0]
    #gedi_df_Q = gedi_df_Q[gedi_df_Q.quality_flag == 1]
    gedi_df_Q = gedi_df[gedi_df[my_Sensitivity] >= 0.90]
    # apply different sensitivity flags:
    gedi_df_QS95 = gedi_df_Q[gedi_df_Q[my_Sensitivity] >= 0.95]
    gedi_df_QS98 = gedi_df_Q[gedi_df_Q[my_Sensitivity] >= 0.98]
    gedi_df_QS99 = gedi_df_Q[gedi_df_Q[my_Sensitivity] >= 0.99]
    # #################################
    # GEDI filtering - set II
    # #################################
    power_beams = ['BEAM0101', 'BEAM0110', 'BEAM1000', 'BEAM1011']
    coverage_beams = ['BEAM0000', 'BEAM0001', 'BEAM0010', 'BEAM0011']
    # set starting sensitivity:
    gedi_df_QS = gedi_df_Q
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
    # Calculate Calibrate GEDI RH98 heights
    # ##############################################################################################
    if CALIBRATED_RH98:
        # make a list of gdf-s:
        gdf_list = [gedi_df, gedi_df_Q, gedi_df_QS95, gedi_df_QS98, gedi_df_QS99]
        indexListModels = [4, 0, 1, 2, 3]
        for my_df, myInd in zip(gdf_list, indexListModels):
            b = calModels_Sens.at[myInd, 'Slope']
            a = calModels_Sens.at[myInd, 'Intercept']
            my_df['rh_98_cal'] = my_df['rh_98'].apply(lambda x: (x-a)/b)

    if CALIBRATED_RH98:
        gdf_list = [gedi_df_QPN, gedi_df_QPD, gedi_df_QP,
                    gedi_df_QCN, gedi_df_QCD, gedi_df_QC,
                    gedi_df_QN, gedi_df_QD, gedi_df_Q]
        indexListModels = np.arange(9)
        for my_df, myInd in zip(gdf_list, indexListModels):
            b = calModels_QS90.at[myInd, 'Slope']
            a = calModels_QS90.at[myInd, 'Intercept']
            my_df['rh_98_cal'] = my_df['rh_98'].apply(lambda x: (x-a)/b)
    # define gedi output
    if SENS_SUBGROUPS_ONLY:
        fun_out_gedi = [gedi_df, gedi_df_Q, gedi_df_QS95, gedi_df_QS98, gedi_df_QS99]
    else:
        fun_out_gedi = [gedi_df_QPN, gedi_df_QPD, gedi_df_QP,
                        gedi_df_QCN, gedi_df_QCD, gedi_df_QC,
                        gedi_df_QN, gedi_df_QD, gedi_df_Q]

    # define ICESat-2 output
    fun_out_atl08 = [atl08_SN, atl08_SD, atl08_S,
                    atl08_WN, atl08_WD, atl08_W,
                    atl08_SWN, atl08_SWD, atl08_df]
    #
    return [fun_out_gedi, fun_out_atl08]