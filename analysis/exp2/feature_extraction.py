import os
import pathlib
import sys
import numpy as np
import pickle
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


def eeg_baseline_normalization(data, baseline, sampling_rate=128):
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

def ppg_gsr_baseline_normalization(data, baseline, sampling_rate=128):
    #return data - np.mean(baseline)
    length = int(baseline.shape[0] / sampling_rate)
    all = []
    for i in range(length):
        all.append(baseline[i*sampling_rate:(i+1)*sampling_rate])
    baseline = np.mean(np.array(all), axis=0)

    window_count = round(data.shape[0] / sampling_rate)
    for i in range(window_count):
        data[i*sampling_rate:(i+1)*sampling_rate] -= baseline
    return data
