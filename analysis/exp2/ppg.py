import os
import csv
import pandas as pd
import numpy as np
import neurokit2 as nk
import heartpy as hp
import matplotlib.pyplot as plt
from processing.preprocessing.ppg import PpgPreprocessing
import processing.feature_extraction.statistics as stat

SAMPLING_RATE = 128
BASELINE_LENGTH = 3


hp_features_names = ["mean", "std", "median", "max", "hrv_madnn", "ibi", "bpm", "pnn50", "sdnn", "sd1", "rmssd"]

peaks_features_names = ["peak_mean", "peak_median", "peak_max", "peak_min",
                        "peak_var", "peak_std", "peak_skew", "peak_kurt", 
                        "peak_pos_derivative", "peak_neg_derivative", "count"]
rate_features_names = ["rate_mean", "rate_median", "rate_max", "rate_min",
                        "rate_var", "rate_std", "rate_skew", "rate_kurt", 
                        "rate_pos_derivative", "rate_neg_derivative"]
hrv_features_names = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_SDANN1', 'HRV_SDNNI1', 'HRV_SDANN2', 
                        'HRV_SDNNI2', 'HRV_SDANN5', 'HRV_SDNNI5', 'HRV_RMSSD', 'HRV_SDSD', 
                        'HRV_CVNN', 'HRV_CVSD', 'HRV_MedianNN', 'HRV_MadNN', 'HRV_MCVNN', 
                        'HRV_IQRNN', 'HRV_pNN50', 'HRV_pNN20', 'HRV_HTI', 'HRV_TINN']


def preprocessing(data, sampling_rate=SAMPLING_RATE):
    #return data[:,0]
    preprocessing = \
        PpgPreprocessing(data[:,0],
                         sampling_rate=sampling_rate)

    preprocessing.neurokit_filtering()
    preprocessing.filtering()

    # If we don't want to do baseline normalization and just remove baseline should pass False to normalization parameter
    preprocessing.baseline_normalization(baseline_duration=BASELINE_LENGTH, normalization=True)
    preprocessed_data = preprocessing.get_data()
    return preprocessed_data

def get_features(data, sampling_rate):
    wd, dict_data = hp.process(data, 128)
    temp = [np.mean(data), np.std(data), np.median(data), np.max(data),
            dict_data["hr_mad"], dict_data["ibi"], dict_data["bpm"],
            dict_data["pnn50"],dict_data["sdnn"],
            dict_data["sd1"], dict_data["rmssd"]]

    return temp

def measure_ppg_features(input_path, output_path):
    all_participants = os.listdir(input_path)
    all_participants.sort()
    ppg_file_path = os.path.join(output_path, "ppg_features_conv.csv")

    '''
                            , 'HRV_ULF', 
                            'HRV_VLF', 'HRV_LF', 'HRV_HF', 'HRV_VHF', 'HRV_LFHF', 'HRV_LFn', 'HRV_HFn', 
                            'HRV_LnHF', 'HRV_SD1', 'HRV_SD2', 'HRV_SD1SD2', 'HRV_S', 'HRV_CSI', 
                            'HRV_CVI', 'HRV_CSI_Modified', 'HRV_PIP', 'HRV_IALS', 'HRV_PSS', 
                            'HRV_PAS', 'HRV_GI', 'HRV_SI', 'HRV_AI', 'HRV_PI', 'HRV_C1d', 
                            'HRV_C1a', 'HRV_SD1d', 'HRV_SD1a', 'HRV_C2d', 'HRV_C2a', 'HRV_SD2d', 
                            'HRV_SD2a', 'HRV_Cd', 'HRV_Ca', 'HRV_SDNNd', 'HRV_SDNNa', 'HRV_DFA_alpha1',
                            'HRV_DFA_alpha1_ExpRange', 'HRV_DFA_alpha1_ExpMean', 'HRV_DFA_alpha1_DimRange',
                            'HRV_DFA_alpha1_DimMean', 'HRV_DFA_alpha2', 'HRV_DFA_alpha2_ExpRange', 
                            'HRV_DFA_alpha2_ExpMean', 'HRV_DFA_alpha2_DimRange', 'HRV_DFA_alpha2_DimMean',
                            'HRV_ApEn', 'HRV_SampEn', 'HRV_ShanEn', 'HRV_FuzzyEn', 'HRV_MSE', 'HRV_CMSE',
                            'HRV_RCMSE', 'HRV_CD', 'HRV_HFD', 'HRV_KFD', 'HRV_LZC']
    '''
    
    features_names = hp_features_names + peaks_features_names + rate_features_names + hrv_features_names


    header = ["participant", "stimuli", "task"] + features_names
        
    with open(ppg_file_path, "a") as ppg_csv_file:
        csv_writer = csv.writer(ppg_csv_file)
        csv_writer.writerow(header)
        for participant in all_participants:
            pnumber = int(participant[1:3])
            f2f_path = os.path.join(input_path, "{0}/preprocessed_output/shimmer/ppg".format(participant))
            remote_path = os.path.join(input_path, "{0}/preprocessed_remote_output/shimmer/ppg".format(participant))
            f2f_rows = get_trials_features(f2f_path, pnumber, 0)
            print("shape f2f", np.array(f2f_rows).shape)
            remote_rows = get_trials_features(remote_path, pnumber, 1)
            print("shape remote", np.array(remote_rows).shape)
            all_participant_rows = f2f_rows + remote_rows
            all_participant_features = np.array(all_participant_rows)
            print("shape", all_participant_features.shape)
            normalized = all_participant_features#normalization(all_participant_features, non_fetures=3)
            np.savetxt(ppg_csv_file, normalized, delimiter=',')

def get_statistical_features(data):
    features = (stat.mean(data) + stat.median(data) + stat.maximum(data) + stat.minimum(data) +
                stat.variance(data) + stat.standard_deviation(data) + stat.skewness(data) +
                stat.kurtosis(data) + stat.sum_of_positive_derivative(data) +
                stat.sum_of_negative_derivative(data))
    return features

def get_trials_features(path, pnumber, task):
    '''
    Measures ppg features for all trials of remote or face-to-face
    '''
    trials = os.listdir(path)
    trials.sort()
    rows = []
    for trial in trials:
        try:
            data_frame = pd.read_csv(os.path.join(path, trial))
            data = data_frame.values
            print(data.shape)
            # Shape of data here is samples * channels
            data = preprocessing(data, sampling_rate=SAMPLING_RATE)


            # Shape of data here is channels * samples
            
            tnumber = int(trial[-6:-4])
            print("participant **************", pnumber, tnumber)
            #display_signal(data)
            row = [pnumber, tnumber, task]
            # Conv data
            data = data[6*SAMPLING_RATE:]

            # Image data, It does not have hrv features
            #data = data[0:6*SAMPLING_RATE]

            ppg_processed, info = nk.ppg_process(data, sampling_rate=SAMPLING_RATE)
            #fig = nk.ppg_plot(ppg_processed)
            #fig 
            #plt.show()
            #print(ppg_processed, type(ppg_processed))
            ppg_clean = ppg_processed["PPG_Clean"].values
            ppg_rate = ppg_processed["PPG_Rate"].values
            ppg_peaks = ppg_processed["PPG_Peaks"]

            try:
                hrv_df = nk.hrv_time(ppg_peaks, sampling_rate=SAMPLING_RATE)
                hrv_features = hrv_df.values
                hrv_features = list(hrv_features[0])
            except:
                hrv_features = [np.nan]*len(hrv_features_names)

            try:
                ppg_peaks = ppg_peaks.values

                ppg_peaks_values = []
                for i in range(ppg_clean.shape[0]):
                    if ppg_peaks[i] == 1:
                        ppg_peaks_values.append(ppg_clean[i])

                peaks_statistics = get_statistical_features(np.array(ppg_peaks_values))
                peaks_statistics.extend([len(ppg_peaks_values)])
                rate_statistics = get_statistical_features(ppg_rate)
            except:
                peaks_statistics = [np.nan]*len(peaks_features_names)
                rate_statistics = [np.nan]*len(rate_features_names)
           
            
            try:
                hp_features = \
                    get_features(ppg_clean, SAMPLING_RATE)
            except:
                hp_features = [np.nan]*len(hp_features_names)
            
            
            all_features = hp_features + peaks_statistics + rate_statistics + hrv_features
        
            # Plot and compare methods
            #ppg_nabian = nk.ppg_clean(data, method='nabian2018', heart_rate=75)
            #signals = pd.DataFrame({"PPG_Raw" : data,
            #                        "PPG_Nabian" : ppg_nabian})
            #signals.plot()
            #plt.show()


            row.extend(all_features)
            print("*********************", len(row))

            rows.append(row)
        except Exception as error:
            pass

    return rows

def display_signal(signal):
    plt.plot(signal)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()

def normalization(data, non_fetures=3):
    non_features_columns = data[:, 0:non_fetures]
    features = data[:, non_fetures:]
    mins = np.nanmin(features, axis=0)
    maxs = np.nanmax(features, axis=0)
    normalized = np.round((features - mins) / (maxs - mins), 4)
    all_data = np.concatenate((non_features_columns, normalized), axis=1)
    return all_data

def standardization(data, non_fetures=3):
    non_features_columns = data[:, 0:non_fetures]
    features = data[:, non_fetures:]
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    normalized = np.round((features - mean) / (std), 4)
    all_data = np.concatenate((non_features_columns, normalized), axis=1)
    return all_data