import os
import sys
import numpy as np
sys.path.append('../')
import pandas as pd
import matplotlib.pyplot as plt
import neurokit2 as nk
import pickle
import pathlib

from processing.feature_extraction.eeg import EegFeatures
from processing.feature_extraction.gsr import GsrFeatureExtraction
from processing.feature_extraction.ppg import get_ppg_components

emotions = {"Anger":[0, "4", "2"],
             "Fear":[1, "4", "2"],
             "Happiness":[2, "4", "5"],
             "Neutral":[3, "3", "3"],
             "Sadness":[4, "2", "2"]}

def partitioning_and_getting_features(input_path, label_path, feature_path, trial_size=60, window_size=0, calculate=False):
    '''
    Reads all files, partition them based on trial size and save features for
    all partitions of all trials of all participants
    output is list(all_participants(all_trials(all_parts)))
    measure labels based on self reports

    EEG, GSR, ppg data should be the in same length befor this method
    '''

    print("********************************")
    all_participants = os.listdir(input_path)
    all_participants.sort()

    all_participants_eeg_features = []
    all_participants_gsr_features = []
    all_participants_ppg_features = []
    all_participants_emotions = []
    all_participants_arousals = []
    all_participants_valences = []
    all_participants_intensity = []
    for participant in all_participants:
        if calculate is True:
            print(participant)
            labels = pd.read_csv("{0}/{1}.csv".format(label_path, participant))
            trials_path = os.path.join(input_path, participant)
            all_eeg_trials, all_emotions, all_arousal, all_valence, all_intensity = \
                three_part_partitioning_and_windowing(os.path.join(trials_path, "eeg"),
                                      128,
                                      trial_size,
                                      labels,
                                      feature_extraction=get_eeg_features,
                                      eeg=True,
                                      window_size=window_size)


            all_gsr_trials, all_emotions, all_arousal, all_valence, all_intensity = \
                three_part_partitioning_and_windowing(os.path.join(trials_path, "gsr"),
                                      128,
                                      trial_size,
                                      labels,
                                      feature_extraction=get_gsr_features,
                                      window_size=window_size)

            all_ppg_trials, all_emotions, all_arousal, all_valence, all_intensity = \
                three_part_partitioning_and_windowing(os.path.join(trials_path, "ppg"),
                                      128,
                                      trial_size,
                                      labels,
                                      feature_extraction=get_ppg_features,
                                      window_size=window_size)

            if not os.path.exists(feature_path):
                pathlib.Path(feature_path).mkdir(parents=True, exist_ok=True)
            pickle.dump((all_eeg_trials, all_gsr_trials, all_ppg_trials,
                         all_emotions, all_arousal, all_valence, all_intensity),
                         open("{0}/{1}.pickle".format(feature_path, participant), "wb"))
        else:
            (all_eeg_trials, all_gsr_trials, all_ppg_trials,
            all_emotions, all_arousal, all_valence, all_intensity) = \
                pickle.load(open("{0}/{1}.pickle".format(feature_path, participant), "rb"))
        # The output for labels are the same for all modalities
        # Since based on labels we have to apply some changes on data (remove when label is other)
        # I had to put calculating labels inside the modality processing method
        all_participants_eeg_features.append(all_eeg_trials)
        all_participants_gsr_features.append(all_gsr_trials)
        all_participants_ppg_features.append(all_ppg_trials)
        all_participants_emotions.append(all_emotions)
        all_participants_arousals.append(all_arousal)
        all_participants_valences.append(all_valence)
        all_participants_intensity.append(all_intensity)
    return all_participants_eeg_features, \
           all_participants_gsr_features, \
           all_participants_ppg_features, \
               all_participants_emotions, \
                   all_participants_arousals, \
                       all_participants_valences, \
                       all_participants_intensity

def get_eeg_features(data, sampling_rate):
    eeg_feature_extraction = EegFeatures(data, sampling_rate)
    features = eeg_feature_extraction.get_channels_power_bands()
    return features

def get_gsr_features(data, sampling_rate):
    gsr_feature_extraction = GsrFeatureExtraction(data, sampling_rate)
    return gsr_feature_extraction.mean() + gsr_feature_extraction.median() + \
           gsr_feature_extraction.get_frequency_peak() + \
           gsr_feature_extraction.get_var_amp_peak() + \
           gsr_feature_extraction.sum_of_positive_derivative() + \
           gsr_feature_extraction.sum_of_negative_derivative() + \
           gsr_feature_extraction.std_amp_peak() + gsr_feature_extraction.skewness_amp_peak() + \
           gsr_feature_extraction.kurtosis_amp_peak() + gsr_feature_extraction.max_abs_amp_peak() + \
           gsr_feature_extraction.variance() + gsr_feature_extraction.standard_deviation()

def get_ppg_features(data, sampling_rate):
    #hr, hrv, rr = get_ppg_components(data, sampling_rate)
    hrv_time = nk.hrv_time(data, sampling_rate=sampling_rate)

    hrv_madnn = hrv_time['HRV_MadNN'].values.tolist()

    hrv_mcvnn = hrv_time['HRV_MCVNN'].values.tolist()

    hrv_iqrnn = hrv_time['HRV_IQRNN'].values.tolist()

    ppg_mean = [np.mean(data)]

    ppg_std = [np.std(data)]

    temp = ppg_mean + ppg_std + hrv_madnn + hrv_mcvnn + hrv_iqrnn

    return np.array(temp)

def display_signal(signal):
    plt.plot(signal)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()

def three_part_partitioning_and_windowing(trials_path,
                                          sampling_rate,
                                          trial_length,
                                          labels,
                                          window_size=0,
                                          eeg=False,
                                          feature_extraction=None,
                                          baseline_length=5):

    trials = os.listdir(trials_path)
    trials.sort()
    all_trials = []
    all_emotions = []
    all_arousal = []
    all_valence = []
    all_intensity = []
    t = 0

    for trial in trials:
        data = np.loadtxt(os.path.join(trials_path, trial))
        if eeg is True:
            data = \
                eeg_baseline_normalization(data[:, sampling_rate*baseline_length:],
                                           data[:, 0:sampling_rate*baseline_length])
            samples = data.shape[1]
        else:
            data = \
                ppg_gsr_baseline_normalization(data[sampling_rate*baseline_length:],
                                               data[0:sampling_rate*baseline_length])
            samples = data.shape[0]
        part_count = int(samples / (trial_length*sampling_rate))
        remaining = samples - (trial_length*sampling_rate) * part_count
        part_length = trial_length*sampling_rate
        start_part = int(part_count/3)
        end_part = int(part_count/3)
        middle_part = part_count - (start_part + end_part)
        all_parts = []
        j = 0
        start = remaining
        end = part_length + remaining
        trial_emotions = []
        trial_arousals = []
        trial_valences = []
        trial_intensity = []
        while j < part_count:
            if j < start_part:
                emotion = labels.at[t, "emotion-1"]
                arousal = labels.at[t, "arousal-1"].astype(str)
                valence = labels.at[t, "valence-1"].astype(str)
                intensity = labels.at[t, "intensity-1"].astype(str)
            elif j >= start_part and j < (start_part + middle_part):
                emotion = labels.at[t, "emotion-2"]
                arousal = labels.at[t, "arousal-2"].astype(str)
                valence = labels.at[t, "valence-2"].astype(str)
                intensity = labels.at[t, "intensity-2"].astype(str)
            else:
                emotion = labels.at[t, "emotion-3"]
                arousal = labels.at[t, "arousal-3"].astype(str)
                valence = labels.at[t, "valence-3"].astype(str)
                intensity = labels.at[t, "intensity-3"].astype(str)
            if emotion == "other":
                j += 1
                continue

            if arousal == "nan":
                arousal = emotions[emotion][1]
            if valence == "nan":
                valence = emotions[emotion][2]

            emotion = emotions[emotion][0]
            if int(arousal) > 3:
                arousal = 1
            else:
                arousal = 0
            if int(valence) >= 3:
                valence = 1
            else:
                valence = 0
            # For EEG dimention is 2 while for GSR and PPG it is 1
            if eeg is True:
                part = data[:, start:end]
            else:
                part = data[start:end]


            all_windows, all_windows_arousal, all_windows_valence, all_windows_emotion, all_windows_intensity = \
                windowing(part, arousal, valence, emotion, intensity,
                          window_size, sampling_rate,
                          eeg=eeg, feature_extraction=feature_extraction)

            all_parts.append(np.array(all_windows))
            trial_arousals.append(all_windows_arousal)
            trial_valences.append(all_windows_valence)
            trial_emotions.append(all_windows_emotion)
            trial_intensity.append(all_windows_intensity)
            start = end
            end = end + part_length
            j += 1
        t += 1
        all_trials.append(all_parts)
        all_emotions.append(trial_emotions)
        all_arousal.append(trial_arousals)
        all_valence.append(trial_valences)
        all_intensity.append(trial_intensity)
    return all_trials, all_emotions, all_arousal, all_valence, all_intensity

def windowing(data, arousal, valence, emotion, intensity, window_size, sampling_rate,
              eeg=False, feature_extraction=None):
    if window_size == 0:
        if feature_extraction is None:
            return [data], [arousal], [valence], [emotion], [intensity]
        else:
            return [feature_extraction(data, sampling_rate)], [arousal], [valence], [emotion], [intensity]
    else:
        window_count = int(60 / window_size)
        window_length = sampling_rate * window_size
        start = 0
        end = window_length
        all_parts = []
        all_arousal = []
        all_valence = []
        all_emotion = []
        all_intensity = []

        if eeg is True:
            samples = data.shape[1]
        else:
            samples = data.shape[0]

        i = 0
        while i < window_count:
            if eeg is True:
                part = data[:, start:end]
            else:
                part = data[start:end]
            if feature_extraction is None:
                all_parts.append(part)
            else:
                all_parts.append(feature_extraction(part, sampling_rate))
            all_arousal.append(arousal)
            all_valence.append(valence)
            all_emotion.append(emotion)
            all_emotion.append(intensity)
            start = end
            end = end + window_length
            if end > samples:
                end = samples
            i += 1
        return all_parts, all_arousal, all_valence, all_emotion, all_intensity


def eeg_baseline_normalization(data, baseline, sampling_rate=128):
    length = int(baseline.shape[1] / sampling_rate)
    all = []
    for i in range(length):
        all.append(baseline[:, i*sampling_rate:(i+1)*sampling_rate])
    baseline = np.mean(np.array(all), axis=0)

    window_count = round(data.shape[1] / sampling_rate)
    for i in range(window_count):
        data[:, i*sampling_rate:(i+1)*sampling_rate] -= baseline
    return data

def ppg_gsr_baseline_normalization(data, baseline, sampling_rate=128):
    length = int(baseline.shape[0] / sampling_rate)
    all = []
    for i in range(length):
        all.append(baseline[i*sampling_rate:(i+1)*sampling_rate])
    baseline = np.mean(np.array(all), axis=0)

    window_count = round(data.shape[0] / sampling_rate)
    for i in range(window_count):
        data[i*sampling_rate:(i+1)*sampling_rate] -= baseline
    return data
