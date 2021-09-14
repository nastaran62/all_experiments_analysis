import os
import sys
import numpy as np
sys.path.append('../')
import pandas as pd
import matplotlib.pyplot as plt
import neurokit2 as nk

from processing.feature_extraction.eeg import EegFeatures
from processing.feature_extraction.gsr import GsrFeatureExtraction
from processing.feature_extraction.ppg import get_ppg_components


emotions = {"Anger":[0, "4", "2"],
             "Fear":[1, "4", "2"],
             "Happiness":[2, "4", "5"],
             "Neutral":[3, "3", "3"],
             "Sadness":[4, "2", "2"]}

def partitioning_and_getting_features(input_path, label_path, window_size=0):
    '''
    Reads all files, partition them based on trial size and save features for
    all partitions of all trials of all participants
    output is list(all_participants(all_trials(all_parts)))
    measure labels based on self reports
    '''
    print("HEllllo")
 
    all_participants = os.listdir(input_path)
    all_participants.sort()

    all_participants_eeg_features = []
    all_participants_gsr_features = []
    all_participants_ppg_features = []
    all_participants_emotions = []
    all_participants_arousals = []
    all_participants_valences = []
    all_participants_dominances = []
    for participant in all_participants:
        print(participant)
        labels = pd.read_csv("{0}/{1}.csv".format(label_path, participant))
        trials_path = os.path.join(input_path, participant)
        all_eeg_trials, all_emotions, all_arousal, all_valence, all_dominance = \
            getting_features(os.path.join(trials_path, "eeg"),
                                  128,
                                  labels,
                                  feature_extraction=get_eeg_features,
                                  eeg=True,
                                  window_size=window_size)
        
 
        all_gsr_trials, all_emotions, all_arousal, all_valence, all_dominance = \
            getting_features(os.path.join(trials_path, "gsr"),
                                  128,
                                  labels,
                                  feature_extraction=get_gsr_features,
                                  window_size=window_size)
        
        all_ppg_trials, all_emotions, all_arousal, all_valence, all_dominance = \
            getting_features(os.path.join(trials_path, "ppg"),
                                  128,
                                  labels,
                                  feature_extraction=get_ppg_features,
                                  window_size=window_size)
        print(np.array(all_eeg_trials).shape)
        print(np.array(all_gsr_trials).shape)
        print(np.array(all_ppg_trials).shape)
        # The output for labels are the same for all modalities
        # Since based on labels we have to apply some changes on data (remove when label is other)
        # I had to put calculating labels inside the modality processing method
        all_participants_eeg_features.append(all_eeg_trials)
        all_participants_gsr_features.append(all_gsr_trials)
        all_participants_ppg_features.append(all_ppg_trials)
        all_participants_emotions.append(all_emotions)
        all_participants_arousals.append(all_arousal)
        all_participants_valences.append(all_valence)
        all_participants_dominances.append(all_dominance)
    return np.array(all_participants_eeg_features), \
           np.array(all_participants_gsr_features), \
           np.array(all_participants_ppg_features), \
               np.array(all_participants_emotions), \
                   np.array(all_participants_arousals), \
                       np.array(all_participants_valences), \
                           np.array(all_participants_dominances)

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
    '''hr, hrv, rr = get_ppg_components(data, sampling_rate)
    try:
        mean_hr = np.mean(hr)
        mean_hrv = np.mean(hrv)
        mean_rr = np.mean(rr)
    except:
        mean_hr = 0
        mean_hrv = 0
        mean_rr = 0
        print(hr)
        print(hrv)
        print(rr)
        print("********************************************")
    '''
    hrv_time = nk.hrv_time(data, sampling_rate=sampling_rate, show=True)

    HRV_MadNN = hrv_time['HRV_MadNN'].values.tolist()

    HRV_MCVNN = hrv_time['HRV_MCVNN'].values.tolist()

    HRV_IQRNN = hrv_time['HRV_IQRNN'].values.tolist()

    HRV_MeanNN = [np.mean(data)]

    HRV_SDNN = [np.std(data)]

    temp = HRV_IQRNN + HRV_MadNN + HRV_MCVNN + HRV_MeanNN + HRV_SDNN

    return np.array(temp)

def display_signal(signal):
    plt.plot(signal)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()

def getting_features(trials_path,
                sampling_rate,
                labels,
                window_size=0,
                eeg=False,
                feature_extraction=None,
                baseline_length=3):
    print(trials_path)
    trials = os.listdir(trials_path)
    trials.sort()
    all_trials = []
    all_emotions = []
    all_arousal = []
    all_valence = []
    all_dominance = []
    t = 0

    for trial in trials:
        data = np.loadtxt(os.path.join(trials_path, trial))
        '''
        # Make all data with the same length
        if eeg is True:
            data = data[:, -10624:]
        else:
            data = data[-10624:]
        np.savetxt(os.path.join(trials_path, trial), data)
        data = np.loadtxt(os.path.join(trials_path, trial))
        print(data.shape, "*************&&&&&&&&&&&&&&&&&&&&&&&")
        '''
        emotion = labels.at[t, "emotion"]
        arousal = labels.at[t, "arousal"]
        valence = labels.at[t, "valence"]
        dominance = labels.at[t, "dominance"]
            
        all_windows, trial_emotions, trial_arousals, trial_valences, trial_dominance = \
            windowing(data, arousal, valence, emotion, dominance,
                      window_size, sampling_rate,
                      eeg=eeg, feature_extraction=feature_extraction,
                      baseline_length=baseline_length)

        all_trials.append(all_windows)
        all_emotions.append(trial_emotions)
        all_arousal.append(trial_arousals)
        all_valence.append(trial_valences)
        all_dominance.append(trial_dominance)
        t += 1
    return all_trials, all_emotions, all_arousal, all_valence, all_dominance

def windowing(data, arousal, valence, emotion, dominance,
              window_size, sampling_rate, eeg=False, feature_extraction=None,
              baseline_length=3):
    if eeg is True:
        data = \
            eeg_baseline_normalization(data[:, 128*baseline_length:],
                                       data[:, 0:128*baseline_length])
        samples = data.shape[1]
    else:
        data = \
            ppg_gsr_baseline_normalization(data[128*baseline_length:],
                                           data[0:128*baseline_length])
        samples = data.shape[0]
    if window_size == 0:
        if feature_extraction is None:
            return [data], [emotion], [arousal], [valence], [emotion]
        else:
            return [feature_extraction(data, sampling_rate)], [emotion],[arousal], [valence], [emotion]
    else:
        window_count = round(samples / (window_size * sampling_rate))
        window_length = sampling_rate * window_size
        start = 0
        end = window_length
        all_parts = []
        all_arousal = []
        all_valence = []
        all_emotion = []
        all_dominance = []
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
            all_dominance.append(dominance)
            start = end
            end = end + window_length
            if end > samples:
                end = samples
            i += 1
        return all_parts, all_emotion, all_arousal, all_valence, all_dominance

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

    
        
            

            


