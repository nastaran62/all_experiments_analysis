import os
import csv
import sys
import pandas as pd
import numpy as np
from analysis.exp1_0.feature_extraction import display_signal

sys.path.append('../')
from processing.feature_extraction.eeg import EegFeatures
from processing.preprocessing.eeg import exp2_eeg_preprocessing

SAMPLING_RATE=125
BASELINE_LENGTH = 3
CHANNEL_NAMES = ["Fp1", "Fp2", "F7", "F3", "F4", "F8", "T3", "C3",
                 "C4", "T4", "T5", "P3", "P4", "T6", "O1", "O2"]


def measure_eeg_features(input_path, output_path):
    all_participants = os.listdir(input_path)
    all_participants.sort()
    eeg_file_path = os.path.join(output_path, "eeg_normalized_conv.csv")
    header = ["participant", "stimuli", "task",
              "psd_delta", "psd_theta", "psd_alpha", "psd_beta", "psd_gamma",
              "mean_delta", "mean_theta", "mean_alpha", "mean_beta", "mean_gamma",
              "entropy_delta", "entropy_theta", "entropy_alpha", "entropy_beta", "entropy_gamma"]
              #"f_delta", "f_theta", "f_alpha", "f_beta", "f_gamma",
              #"p_delta", "p_theta", "p_alpha", "p_beta", "p_gamma",
              #"t_delta", "t_theta", "t_alpha", "t_beta", "t_gamma",
              #"c_delta", "c_theta", "c_alpha", "c_beta", "c_gamma",
              #"o_delta", "o_theta", "o_alpha", "o_beta", "o_gamma"]
              
    with open(eeg_file_path, "a") as eeg_csv_file:
        csv_writer = csv.writer(eeg_csv_file)
        csv_writer.writerow(header)
        for participant in ["p14", "p15", "p16"]:
            print("participant", participant)
            pnumber = int(participant[1:3])
            f2f_path = os.path.join(input_path, "{0}/preprocessed_output/eeg".format(participant))
            remote_path = os.path.join(input_path, "{0}/preprocessed_remote_output/eeg".format(participant))
            f2f_rows = get_trials_features(f2f_path, pnumber, 0)
            remote_rows = get_trials_features(remote_path, pnumber, 1)
            all_participant_rows = f2f_rows + remote_rows
            all_participant_features = np.array(all_participant_rows)
            normalized = normalization(all_participant_features, non_fetures=3)
            np.savetxt(eeg_csv_file, normalized, delimiter=',')


def get_eeg_features(data, sampling_rate, method="psd"):
    eeg_feature_extraction = EegFeatures(data, sampling_rate)
    features = eeg_feature_extraction.get_total_power_bands(method=method)
    return features

def normalization(data, non_fetures=3):
    non_features_columns = data[:, 0:non_fetures]
    features = data[:, non_fetures:]
    mins = np.nanmin(features, axis=0)
    maxs = np.nanmax(features, axis=0)
    normalized = np.round((features - mins) / (maxs - mins), 4)
    all_data = np.concatenate((non_features_columns, normalized), axis=1)
    return all_data

def get_trials_features(path, pnumber, task, method="psd"):
    '''
    Measures eeg features for all trials of remote or face-to-face
    '''
    trials = os.listdir(path)
    trials.sort()
    rows = []
    for trial in trials:
        print("trial****************************", trial)
        data_frame = pd.read_csv(os.path.join(path, trial))
        data = data_frame.values
        # Shape of data here is samples * channels
        data = exp2_eeg_preprocessing(data, sampling_rate=SAMPLING_RATE)

        # Shape of data here is channels * samples
        data = \
            eeg_baseline_normalization(data[:, SAMPLING_RATE*BASELINE_LENGTH:],
                                        data[:, 0:SAMPLING_RATE*BASELINE_LENGTH])
        data = data[:, 6*SAMPLING_RATE:]
        preprocessed_trial = "preprocessed_" + trial
        print(os.path.join(path, preprocessed_trial), "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        np.savetxt(os.path.join(path, preprocessed_trial), np.transpose(data))
        
        tnumber = int(trial[-6:-4])
        row = [pnumber, tnumber, task]

        # All channels
        all_channels_features = \
            get_eeg_features(data, SAMPLING_RATE, method="psd")
        row.extend(list(all_channels_features))
        all_channels_features = \
            get_eeg_features(data, SAMPLING_RATE, method="mean")
        row.extend(list(all_channels_features))
        all_channels_features = \
            get_eeg_features(data, SAMPLING_RATE, method="entropy")
        row.extend(list(all_channels_features))
        '''
        # Frontal channels
        frontal_channels_features = \
            get_eeg_features(data[[0, 1, 2, 3, 4, 5], :], SAMPLING_RATE, method=method)
        #row.extend(list(frontal_channels_features))

        # Parietal channels
        parietal_channels_features = \
            get_eeg_features(data[[11, 12],:], SAMPLING_RATE, method=method)
        #row.extend(list(parietal_channels_features))

        # Temporal channels
        temporal_channels_features = \
            get_eeg_features(data[[6, 9, 10, 13],:], SAMPLING_RATE, method=method)
        #row.extend(list(temporal_channels_features))

        # Central channels
        central_channels_features = \
            get_eeg_features(data[[7, 8],:], SAMPLING_RATE, method=method)
        #row.extend(list(central_channels_features))

        # Occipital channels
        occipital_channels_features = \
            get_eeg_features(data[[14, 15],:], SAMPLING_RATE, method=method)
        #row.extend(list(occipital_channels_features))
        '''
        rows.append(row)

    return rows


def eeg_baseline_normalization(data, baseline, sampling_rate=SAMPLING_RATE):
    #return data - np.mean(baseline)
    length = int(baseline.shape[1] / sampling_rate)
    all = []
    for i in range(length):
        all.append(baseline[:, i*sampling_rate:(i+1)*sampling_rate])
    baseline = np.mean(np.array(all), axis=0)

    window_count = round(data.shape[1] / sampling_rate)
    for i in range(window_count):
        data[:, i*sampling_rate:(i+1)*sampling_rate] -= baseline
    return data

