import os
import csv
import pandas as pd
from processing.preprocessing.gsr import GsrPreprocessing
from processing.feature_extraction.gsr import GsrFeatureExtraction
import processing.feature_extraction.statistics as stat
import neurokit2 as nk
import numpy as np

SAMPLING_RATE = 128
BASELINE_LENGTH = 3


def preprocessing(data, sampling_rate=SAMPLING_RATE):
    preprocessing = \
        GsrPreprocessing(data,
                        sampling_rate=sampling_rate)
    # neurokit method
    preprocessing.gsr_noise_cancelation()
    # If we don't want to do baseline normalization and just remove baseline should pass False to normalization parameter
    preprocessing.baseline_normalization(baseline_duration=BASELINE_LENGTH, normalization=False)
    preprocessed_data = preprocessing.get_data()
    return preprocessed_data

def get_phasic_tonic(data, sampling_rate=SAMPLING_RATE):
    
    eda_clean = nk.eda_clean(data, sampling_rate=sampling_rate)
    phasic_tonic_dataframe = nk.eda_phasic(eda_clean, sampling_rate=sampling_rate)
    eda_phasic = phasic_tonic_dataframe["EDA_Phasic"].values
    eda_tonic = phasic_tonic_dataframe["EDA_Tonic"].values
    vanhalem2020 = nk.eda_findpeaks(eda_phasic, method="vanhalem2020")
    peaks = vanhalem2020["SCR_Peaks"]
    #peaks = nk.events_plot(vanhalem2020["SCR_Peaks"], eda_phasic)
    #plt.show()
    return eda_phasic, eda_tonic, peaks

def get_statistical_features(data):
    features = (stat.mean(data) + stat.median(data) + stat.maximum(data) + stat.minimum(data) +
                stat.variance(data) + stat.standard_deviation(data) + stat.skewness(data) +
                stat.kurtosis(data) + stat.sum_of_positive_derivative(data) +
                stat.sum_of_negative_derivative(data))
    return features

def measure_gsr_features(input_path, output_path):
    all_participants = os.listdir(input_path)
    all_participants.sort()
    gsr_file_path = os.path.join(output_path, "gsr_features_img.csv")
    header = ["participant", "stimuli", "task",
              "mean", "median", "max", "min", "var", "std", "skew", "kurtosis",
              "pos_derivatives", "neg_deravatives","phasic_mean", "phasic_median", 
              "phasic_max", "phasic_min", "phasic_var", "phasic_std", "phasic_skew",
              "phasic_kurtosis", "phasic_pos_derivatives", "phasic_neg_deravatives",
              "tonic_mean", "tonic_median", "tonic_max", "tonic_min", "tonic_var", 
              "tonic_std", "tonic_skew", "tonic_kurtosis", "tonic_pos_derivatives",
              "tonic_neg_deravatives", "peaks_mean", "peaks_median","peaks_max",
              "peaks_min", "peaks_var", "peaks_std", "peaks_skew", "peaks_kurtosis",
              "peaks_pos_derivatives", "peaks_neg_deravatives", "peaks_freq"]

    with open(gsr_file_path, "a") as gsr_csv_file:
        csv_writer = csv.writer(gsr_csv_file)
        csv_writer.writerow(header)
        for participant in all_participants:
            pnumber = int(participant[1:3])
            f2f_path = os.path.join(input_path, "{0}/preprocessed_output/shimmer/gsr".format(participant))
            remote_path = os.path.join(input_path, "{0}/preprocessed_remote_output/shimmer/gsr".format(participant))
            f2f_rows = get_trials_features(f2f_path, pnumber, 0)
            remote_rows = get_trials_features(remote_path, pnumber, 1)
            all_participant_rows = f2f_rows + remote_rows
            all_participant_features = np.array(all_participant_rows)
            normalized = all_participant_features#normalization(all_participant_features, non_fetures=3)
            np.savetxt(gsr_csv_file, normalized, delimiter=',')

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

def get_trials_features(path, pnumber, task):
    '''
    Measures gsr features for all trials of remote or face-to-face
    '''
    trials = os.listdir(path)
    trials.sort()
    rows = []
    for trial in trials:
        data_frame = pd.read_csv(os.path.join(path, trial))
        data = data_frame.values
        # Shape of data here is samples * channels
        data = preprocessing(data, sampling_rate=SAMPLING_RATE)
        
        # Conv data
        #data = data[6*SAMPLING_RATE:]

        # Image data
        data = data[0:6*SAMPLING_RATE]
        
        tnumber = int(trial[-6:-4])
        row = [pnumber, tnumber, task]

        gsr_features = \
            get_statistical_features(data)
        phasic, tonic, peaks = get_phasic_tonic(data)
        if len(peaks) == 0:
            print(pnumber, tnumber, task)
            peaks = np.array([0])
        phasic_features = \
            get_statistical_features(phasic)
        tonic_features = \
            get_statistical_features(tonic)
        peak_statistics = get_statistical_features(peaks)
        peak_statistics.extend([len(peaks)])
        row.extend(gsr_features)
        row.extend(phasic_features)
        row.extend(tonic_features)
        row.extend(peak_statistics)

        rows.append(row)

    return rows
