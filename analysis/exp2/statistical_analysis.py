from asyncio import Task
from turtle import color
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.anova import AnovaRM
import seaborn as sns
import pingouin as pg
import numpy as np
from scipy.stats import boxcox
import csv
from scipy.stats import shapiro
from numpy.random import randn
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
pandas2ri.activate()
import rpy2.robjects as ro




def minmax_scaling(data):
    return (data - data.min()) / (data.max() - data.min())

def arousal_valence_ttest():
    all_f2f = pd.read_csv("../../experimental_data/exp2-0/all_f2f.csv")
    all_remote = pd.read_csv("../../experimental_data/exp2-0/all_remote.csv")
    all_data = pd.concat([all_f2f, all_remote], ignore_index=True, keys=all_f2f.index)
    variables = ["intensity", "arousal", "valence", "dominance"] 
    with open("result/arousal-valence-ttest.csv", 'a') as statistics_csv_file:
        csv_writer = csv.writer(statistics_csv_file)
        for variable in variables:
            a = all_data.query('task == 0')[variable]
            b = all_data.query('task == 1')[variable]
            result = pg.ttest(a, b, paired=True)
            print(result)
            x, y = result.values.shape
            for i in range(x):
                row = [variable] + list(result.values[i])
                csv_writer.writerow(row)
            statistics_csv_file.flush()
            fig = plt.figure()
            sns.set()
            sns.pointplot(data=all_data, x="task", y=variable, dodge=True, markers=['o', 's'],
                                capsize=.1, errwidth=1, palette='colorblind')
    plt.show()
def arousal_valence_anova():
    all_f2f = pd.read_csv("../../experimental_data/exp2-0/all_f2f.csv")
    all_remote = pd.read_csv("../../experimental_data/exp2-0/all_remote.csv")
    all_data = pd.concat([all_f2f, all_remote], ignore_index=True, keys=all_f2f.index)
    variables = ["intensity", "arousal", "valence", "dominance"] 
    with open("result/av-anova-emotion.csv", 'a') as statistics_csv_file:
        csv_writer = csv.writer(statistics_csv_file)
        for variable in variables:
            df = all_data.groupby(["participant", "task", "emotion"], as_index=False)[variable].mean()
            df.to_csv("temp.csv", index = False)
            formula = "{0} ~ factor({1})*factor({2}) + (1|{3})".format(variable, "task", "emotion", "participant")
            result = call_r_function(formula, "temp.csv")
            print(result, type(result))
            df_result = pandas2ri.rpy2py(result)
            print(df_result.values[0])
            x, y = df_result.values.shape
            for i in range(x):
                row = [variable] + list(df_result.values[i])
                csv_writer.writerow(row)
            statistics_csv_file.flush()
            fig = plt.figure()
            sns.set()
            sns.pointplot(data=all_data, x="emotion", hue="task", y=variable, dodge=True,
                                capsize=.1, errwidth=1, palette='colorblind')
    plt.show()

def empathy_artanova_test():
    all_empathy = pd.read_csv("../../experimental_data/exp2-0/empathy-all.csv")
    variables = ['Int_to_part', 'Part_to_int', 'identify', 'reflect', 'interact', 'understand']
    # anova
    '''
    with open("result/empathy-gender.csv", 'a') as statistics_csv_file:
        csv_writer = csv.writer(statistics_csv_file)
        for variable in variables:
            formula = "{0} ~ factor({1}) + (1|{2})".format(variable, "task", "participant")
            formula = "{0} ~ factor({1})*factor({2}) + (1|{3})".format(variable, "task", "gender", "participant")
            result = call_r_function(formula, "../../experimental_data/exp2-0/empathy-all.csv")
            print(result, type(result))
            df_result = pandas2ri.rpy2py(result)
            print(df_result.values[0])
            x, y = df_result.values.shape
            for i in range(x):
                row = [variable] + list(df_result.values[i])
                csv_writer.writerow(row)
            statistics_csv_file.flush()
    
    # t-test
    with open("result/empathy-ttest.csv", 'a') as statistics_csv_file:
        csv_writer = csv.writer(statistics_csv_file)
        for variable in variables:
            a = all_empathy.query('task == 0')[variable]
            b = all_empathy.query('task == 1')[variable]
            result = pg.ttest(a, b, paired=True)
            x, y = result.values.shape
            for i in range(x):
                row = [variable] + list(result.values[i])
                csv_writer.writerow(row)
            statistics_csv_file.flush()

    '''
    for variable in variables:
        fig = plt.figure()
        sns.set()
        sns.pointplot(data=all_empathy, x="task", hue="gender", y=variable, dodge=True, markers=['o', 's'],
                        capsize=.1, errwidth=1, palette='colorblind')
    plt.show()
    


def anova_test():
    all_f2f = pd.read_csv("../../experimental_data/exp2-0/all_f2f.csv")
    all_remote = pd.read_csv("../../experimental_data/exp2-0/all_remote.csv")

    modality_data = pd.read_csv("extracted_data/not_normalized/gsr_features_conv.csv")

    all_data = pd.concat([all_f2f, all_remote], ignore_index=True, keys=all_f2f.index)

    gsr_header = ["participant", "stimuli", "task",
              "mean", "median", "max", "min", "var", "std", "skew", "kurtosis",
              "pos_derivatives", "neg_deravatives","phasic_mean", "phasic_median", 
              "phasic_max", "phasic_min", "phasic_var", "phasic_std", "phasic_skew",
              "phasic_kurtosis", "phasic_pos_derivatives", "phasic_neg_deravatives",
              "tonic_mean", "tonic_median", "tonic_max", "tonic_min", "tonic_var", 
              "tonic_std", "tonic_skew", "tonic_kurtosis", "tonic_pos_derivatives",
              "tonic_neg_deravatives", "peaks_mean", "peaks_median","peaks_max",
              "peaks_min", "peaks_var", "peaks_std", "peaks_skew", "peaks_kurtosis",
              "peaks_pos_derivatives", "peaks_freq"]
    
    hp_features_names = ["mean", "std", "median", "max", "hrv_madnn", "ibi", "bpm", "pnn50", "sdnn", "sd1", "rmssd"]

    peaks_features_names = ["peak_mean", "peak_median", "peak_max", "peak_min",
                            "peak_var", "peak_std", "peak_skew", "peak_kurt", 
                            "peak_pos_derivative", "peak_neg_derivative", "count"]
    rate_features_names = ["rate_mean", "rate_median", "rate_max", "rate_min",
                            "rate_var", "rate_std", "rate_skew", "rate_kurt", 
                            "rate_pos_derivative", "rate_neg_derivative"]

    hrv_features_names = ['HRV_MeanNN', 'HRV_SDNN',  'HRV_RMSSD', 'HRV_SDSD', 
                            'HRV_CVNN', 'HRV_CVSD', 'HRV_MedianNN', 'HRV_MadNN', 'HRV_MCVNN', 
                            'HRV_IQRNN', 'HRV_pNN50', 'HRV_pNN20', 'HRV_HTI', 'HRV_TINN']
                            #'HRV_SDANN1', 'HRV_SDNNI1', 'HRV_SDANN2', 'HRV_SDNNI2', 'HRV_SDANN5', 'HRV_SDNNI5',

    ppg_header = hp_features_names + peaks_features_names + rate_features_names + hrv_features_names

    header = gsr_header[3:]
    #header = ppg_header
    header = ["arousal", "valence", "intensity", "dominance"]
    for variable in header:
        try:
            print(variable)
            with open("result/conv/task-gender.csv", 'a') as statistics_csv_file:
        
                csv_writer = csv.writer(statistics_csv_file)
                treatment = "task"
                condition = "gender"
                joined_data = all_data.set_index(["participant", "stimuli", treatment]).join(modality_data.set_index(["participant", "stimuli", treatment]))
                joined_data.reset_index(inplace=True)
                df = joined_data.filter(['participant', treatment, condition, variable], axis=1)
                #df = joined_data.filter(['participant', treatment, condition1, condition2, variable], axis=1)
                df = df.groupby(["participant", treatment, condition], as_index=False)[variable].mean()
                #df = df.groupby(["participant", condition], as_index=False)[variable].mean()
                df.to_csv("temp.csv", index = False)
                

            
                # I tried to use the ARTool in python, but it raises some errors. 
                # So I write the code in R and called it here
                formula = "{0} ~ factor({1})*factor({2}) + (1|{3})".format(variable, treatment, condition, "participant")
                #formula = "{0} ~ factor({1}) + (1|{2})".format(variable, condition, "participant")
                #formula = "{0} ~ factor({1})*factor({2})*factor({3}) + (1|{4})".format(variable, treatment, condition1, condition2, "participant")
                result = call_r_function(formula, "temp.csv")
                print(result, type(result))
                df_result = pandas2ri.rpy2py(result)
                print(df_result.values[0])
                x, y = df_result.values.shape
                for i in range(x):
                    row = [variable] + list(df_result.values[i])
                    csv_writer.writerow(row)
                '''
                row = [variable] + list(df_result.values[1])
                csv_writer.writerow(row)
                row = [variable] + list(df_result.values[2])
                csv_writer.writerow(row)
                '''
                statistics_csv_file.flush()
                fig = plt.figure()
                sns.set()
                sns.pointplot(data=df, x=condition, y=variable, hue=treatment, dodge=True, markers=['o', 's'],
                            capsize=.1, errwidth=1, palette='colorblind')
        
        except Exception as error:
            print(error)
            print("not available for {0}".format(variable))
    plt.show()


def threeway_anova_test():
    all_f2f = pd.read_csv("../../experimental_data/exp2-0/all_f2f.csv")
    all_remote = pd.read_csv("../../experimental_data/exp2-0/all_remote.csv")

    modality_data = pd.read_csv("extracted_data/normalized/ppg_features_conv.csv")
    modality_data = pd.read_csv("extracted_data/normalized/eeg_normalized.csv")

    all_data = pd.concat([all_f2f, all_remote], ignore_index=True, keys=all_f2f.index)

    gsr_header = ["participant", "stimuli", "task",
              "mean", "median", "max", "min", "var", "std", "skew", "kurtosis",
              "pos_derivatives", "neg_deravatives","phasic_mean", "phasic_median", 
              "phasic_max", "phasic_min", "phasic_var", "phasic_std", "phasic_skew",
              "phasic_kurtosis", "phasic_pos_derivatives", "phasic_neg_deravatives",
              "tonic_mean", "tonic_median", "tonic_max", "tonic_min", "tonic_var", 
              "tonic_std", "tonic_skew", "tonic_kurtosis", "tonic_pos_derivatives",
              "tonic_neg_deravatives", "peaks_mean", "peaks_median","peaks_max",
              "peaks_min", "peaks_var", "peaks_std", "peaks_skew", "peaks_kurtosis",
              "peaks_pos_derivatives", "peaks_freq"]
    
    hp_features_names = ["mean", "std", "median", "max", "hrv_madnn", "ibi", "bpm", "pnn50", "sdnn", "sd1", "rmssd"]

    peaks_features_names = ["peak_mean", "peak_median", "peak_max", "peak_min",
                            "peak_var", "peak_std", "peak_skew", "peak_kurt", 
                            "peak_pos_derivative", "peak_neg_derivative", "count"]
    rate_features_names = ["rate_mean", "rate_median", "rate_max", "rate_min",
                            "rate_var", "rate_std", "rate_skew", "rate_kurt", 
                            "rate_pos_derivative", "rate_neg_derivative"]

    hrv_features_names = ['HRV_MeanNN', 'HRV_SDNN',  'HRV_RMSSD', 'HRV_SDSD', 
                            'HRV_CVNN', 'HRV_CVSD', 'HRV_MedianNN', 'HRV_MadNN', 'HRV_MCVNN', 
                            'HRV_IQRNN', 'HRV_pNN50', 'HRV_pNN20', 'HRV_HTI', 'HRV_TINN']
                            #'HRV_SDANN1', 'HRV_SDNNI1', 'HRV_SDANN2', 'HRV_SDNNI2', 'HRV_SDANN5', 'HRV_SDNNI5',

    ppg_header = hp_features_names + peaks_features_names + rate_features_names + hrv_features_names

    eeg_header = ["participant", "stimuli", "task",
              "psd_delta", "psd_theta", "psd_alpha", "psd_beta", "psd_gamma",
              "mean_delta", "mean_theta", "mean_alpha", "mean_beta", "mean_gamma",
              "entropy_delta", "entropy_alpha", "entropy_beta"]

    header = gsr_header[3:]
    header = ppg_header
    header = eeg_header[3:]
    #header = ["arousal", "valence"]
    for variable in header:
        try:
            print(variable)
            with open("result/conv/eeg_three_way.csv", 'a') as statistics_csv_file:
        
                csv_writer = csv.writer(statistics_csv_file)
                treatment = "task"
                condition1 = "arousal_b"
                condition2 = "valence_b"
                #joined_data = all_data.set_index(["participant", "stimuli", treatment]).join(modality_data.set_index(["participant", "stimuli", treatment]))
                joined_data = all_data.set_index(["participant", "stimuli", treatment]).join(modality_data.set_index(["participant", "stimuli", treatment]), on=["participant", "stimuli", treatment], how="inner")
                joined_data.reset_index(inplace=True)
                joined_data.to_csv("temp00.csv", index = False)
                df = joined_data.filter(['participant', treatment, condition1, condition2, variable], axis=1)
                df.to_csv("temp0.csv", index = False)
                df = df.groupby(["participant", treatment, condition1, condition2], as_index=False)[variable].mean()
                df.to_csv("temp.csv", index=False)
                
                # I tried to use the ARTool in python, but it raises some errors. 
                # So I write the code in R and called it here
                formula = "{0} ~ factor({1}) * factor({2}) * factor({3}) + (1|{4})".format(variable, treatment, condition1, condition2, "participant")
                result = call_r_function(formula, "temp.csv")
                print(result, type(result))
                df_result = pandas2ri.rpy2py(result)
                print(df_result.values[0])
                x, y = df_result.values.shape
                for i in range(x):
                    row = [variable] + list(df_result.values[i])
                    csv_writer.writerow(row)
                statistics_csv_file.flush()
        except Exception as error:
            print(error)
            print("not available for {0}".format(variable))


def drop_outliers_IQR(df):
   q1=df.quantile(0.25)
   q3=df.quantile(0.75)
   IQR=q3-q1
   outliers = df[((df<(q1-1.5*IQR)) | (df>(q3+1.5*IQR)))]
   outliers_dropped = outliers.dropna().reset_index()

   return outliers_dropped

def cap_outliers(df):
    upper_limit = df.mean() + 3*df.std()
    lower_limit = df.mean() - 3*df.std()


    df = np.where(df > upper_limit, upper_limit,
                  np.where(df < lower_limit, lower_limit,
                  df))
    return df

def call_r_function(formula, file_path):
    r=ro.r
    r.source("art-anova.r")
    output = r.art_anova(formula, file_path)
    return output


def display_plots():
    all_f2f = pd.read_csv("../../experimental_data/exp2-0/all_f2f.csv")
    all_remote = pd.read_csv("../../experimental_data/exp2-0/all_remote.csv")


    modality_data = pd.read_csv("extracted_data/normalized/gsr_features_conv.csv")

    all_data = pd.concat([all_f2f, all_remote], ignore_index=True, keys=all_f2f.index)

    treatment = "task"
    condition1 = "arousal_b"
    condition2 = "valence_b"
    significant_variables1 = ["tonic_min", "peaks_pos_derivatives", "peaks_freq", "peaks_var", "phasic_mean", "phasic_median", "max"]
    for variable in significant_variables1:
        joined_data = all_data.set_index(["participant", "stimuli", treatment]).join(modality_data.set_index(["participant", "stimuli", treatment]))
        joined_data.reset_index(inplace=True)
        df = joined_data.filter(['participant', treatment, condition1, condition2, variable], axis=1)
        df = df.groupby(["participant", treatment, condition1, condition2], as_index=False)[variable].mean()
        joined_data.reset_index(inplace=True)
        ''' 
        Correlation analysis
        corr = joined_data.corr()
        x, y = corr.values.shape
        sets = []
        for i in range(x):
            setv = [i]
            for j in range(y):
                for k in range(len(setv)):
                    if corr.values[setv[k]][j] >= 0.7 or corr.values[setv[k]][j] <= -0.7:
                        setv.append(j)
            sets.append(list(set(setv)))
        names = []
        for item in sets:
            item_names = []
            for i in range(len(item)):
                item_names.append(corr.columns[item[i]])
            item_names.sort()
            names.append(item_names)
        for item in names:
            print(item)
        print("*************************")
        print(x, y)
        input()
        #corr.to_csv("PPG_correlation.csv")
        #corr = corr.unstack()
        corr_positive = corr[corr >= 0.7]
        f
        corr_negative = corr[corr <= -0.7]
        print(type(corr_positive))
        print("corr_positive", corr_positive[0:5])
        print("corr_negative", corr_negative)
        '''   
        #df.boxplot(column=[variable], by=[treatment, condition1])

        fig = plt.figure()
        sns.set()
        sns.pointplot(data=df, x=condition2, y=variable, hue=condition1, dodge=True, markers=['o', 's'],
                    capsize=.1, errwidth=1, palette='colorblind')
    plt.show()


threeway_anova_test()
#arousal_valence_anova()
#arousal_valence_ttest()
#empathy_artanova_test()
#display_plots()
#anova_test()


# Rubish code
'''
    #df[variable] = cap_outliers(df[variable])
    df[variable].hist(grid=False)
    #plt.show()
    print(df.agg(['skew', 'kurtosis']).transpose())
    shapiro_before =  shapiro(df[variable])[1]
    print(shapiro(df[variable]), type(shapiro(df[variable])), shapiro(df[variable])[1])

    # For boxcox data should be positive
    data = df[variable]
    postdata = data[data>0]
    bcdata, lam = boxcox(postdata)
    x = np.empty_like(data)
    x[data > 0] = bcdata
    x[data == 0] = -1/lam

    df.insert(len(df.columns), 'transformed', x)
    df['transformed'].hist(grid=False)
    df.boxplot(column=[variable], by=[treatment])
    shapiro_after = shapiro(df['transformed'])[1]
    print(shapiro_after)
    #plt.show()

    sns.set()
    sns.pointplot(data=df, x=condition, y='transformed', hue=treatment, dodge=True, markers=['o', 's'],
                capsize=.1, errwidth=1, palette='colorblind')
    #plt.show()
    #df.groupby(['arousal_b', 'task'])['transformed'].agg(['mean', 'std']).round(2)

    artool = importr('ARTool')

    df_r = pandas2ri.py2rpy_pandasdataframe(df)
    print(df_r)
    print(type(df_r))

    formula = "{0} ~ {1} + {2} + {1}:{2} + (1|{3})".format(variable, treatment, condition, "participant")
    formula = "{0} ~ factor({1})*factor({2}) + (1|{3})".format(variable, treatment, condition, "participant")
    m = artool.art(formula, data=df_r)
    df_result_r = artool.anova(m)
    df_result = pandas2ri.rpy2py_pandasdataframe(df_result_r)
    print(df_result)         

    aov = pg.rm_anova(dv='transformed', within=[condition, treatment], subject='participant', data=df, detailed=True)
    print(aov.round(3).values[0, :])
    # Pretty printing of ANOVA summary
    pg.print_table(aov.round(3))
    csv_writer.writerow([variable, shapiro_before, shapiro_after] + list(aov.round(3).values[0, :]))
    csv_writer.writerow([variable, shapiro_before, shapiro_after] + list(aov.round(3).values[1, :]))
    csv_writer.writerow([variable, shapiro_before, shapiro_after] + list(aov.round(3).values[2, :]))
    statistics_csv_file.flush()
    # Put a loop around for all variables to see which one is more significant
'''