#  read data from raw_data path,
#   resample them based on input sampling rate,
#   split based on marker and save in prepared_data path
#   (for each experiment we save raw_data in a folder in this path and then
#    save prepared data and preprocessed data in the same path)

import os
import sys
import pandas as pd
import pickle
import numpy as np
sys.path.append('../')
import pathlib


from processing.preprocessing.eeg import EegPreprocessing
from processing.preprocessing.gsr import GsrPreprocessing
from processing.preprocessing.ppg import PpgPreprocessing

def preprocessing_exp1_1(eeg=True, gsr=True, ppg=True, face=False,
                         start_trim=0, end_trim=0, baseline_duration=0):
    input_path = "exp1_1/prepared_data"
    output_path = "exp1_1/preprocessed_data"
    all_participants = os.listdir(input_path)
    all_participants.sort()
    print(all_participants)
    print(all_participants[7:8])
    for participant in all_participants[7:8]:
        participant_output_path = os.path.join(output_path, participant)
        if not os.path.exists(participant_output_path):
            os.mkdir(participant_output_path)
        if eeg is True:
            eeg_output_path = os.path.join(participant_output_path, "eeg")
            if not os.path.exists(eeg_output_path):
                os.mkdir(eeg_output_path)
            eeg_path = os.path.join(input_path, participant, "openbci")
            eeg_files = os.listdir(eeg_path)
            for eeg_file in eeg_files:
                data_frame = pd.read_csv(os.path.join(eeg_path, eeg_file))
                eeg_data = data_frame.to_numpy()
                channel_names = ["Fp1", "Fp2", "F7", "F3", "F4", "F8", "T3", "C3",
                            "C4", "T4", "T5", "P3", "P4", "T6", "O1", "O2"]
                preprocessing = \
                    EegPreprocessing(eeg_data,
                                    channel_names=channel_names,
                                    sampling_rate=128)
                preprocessing.filter_data()
                preprocessing.interpolate_bad_channels()
                preprocessing.rereferencing(referencing_value='average')
                #preprocessing.channel_wise_baseline_normalization(baseline_duration=baseline_duration)
                preprocessed_data = preprocessing.get_data()
                #preprocessed_data = \
                #    preprocessed_data[:, start_trim*128:-end_trim*128]
                file_name = os.path.join(eeg_output_path, eeg_file)
                np.savetxt(file_name, preprocessed_data)

        if gsr is True:
            gsr_output_path = os.path.join(participant_output_path, "gsr")
            if not os.path.exists(gsr_output_path):
                os.mkdir(gsr_output_path)
            gsr_path = os.path.join(input_path, participant, "gsr")
            gsr_files = os.listdir(gsr_path)
            for gsr_file in gsr_files:
                data_frame = pd.read_csv(os.path.join(gsr_path, gsr_file))
                gsr_data = data_frame.to_numpy()
                preprocessing = \
                    GsrPreprocessing(gsr_data[:,0],
                                    sampling_rate=128)
                preprocessing.gsr_noise_cancelation()
                #preprocessing.baseline_normalization(baseline_duration=baseline_duration)
                preprocessed_data = preprocessing.get_data()
                #preprocessed_data = \
                #    preprocessed_data[start_trim*128:-end_trim*128]
                file_name = os.path.join(gsr_output_path, gsr_file)
                np.savetxt(file_name, preprocessed_data)

        if ppg is True:
            ppg_output_path = os.path.join(participant_output_path, "ppg")
            if not os.path.exists(ppg_output_path):
                os.mkdir(ppg_output_path)
            ppg_path = os.path.join(input_path, participant, "ppg")
            ppg_files = os.listdir(ppg_path)
            for ppg_file in ppg_files:
                data_frame = pd.read_csv(os.path.join(ppg_path, ppg_file))
                ppg_data = data_frame.to_numpy()
                preprocessing = \
                    PpgPreprocessing(ppg_data[:, 0],
                                     sampling_rate=128)
                preprocessing.filtering()
                #preprocessing.baseline_normalization(baseline_duration=baseline_duration)
                preprocessed_data = preprocessing.get_data()
                #preprocessed_data = \
                #    preprocessed_data[start_trim*128:-end_trim*128]
                file_name = os.path.join(ppg_output_path, ppg_file)
                np.savetxt(file_name, preprocessed_data)

def preprocessing_exp1_0(eeg=True, gsr=True, ppg=True, face=False,
                         start_trim=0, end_trim=0, baseline_duration=3):
    input_path = "exp1_0/prepared_data"
    output_path = "exp1_0/preprocessed_data"
    all_participants = os.listdir(input_path)
    all_participants.sort()
    for participant in all_participants[22:23]:
        print(participant)
        participant_output_path = os.path.join(output_path, participant)
        if not os.path.exists(participant_output_path):
            os.mkdir(participant_output_path)
        if eeg is True:
            eeg_output_path = os.path.join(participant_output_path, "eeg")
            if not os.path.exists(eeg_output_path):
                os.mkdir(eeg_output_path)
            eeg_path = os.path.join(input_path, participant, "openbci")
            eeg_files = os.listdir(eeg_path)
            for eeg_file in eeg_files:
                data_frame = pd.read_csv(os.path.join(eeg_path, eeg_file))
                eeg_data = data_frame.to_numpy()
                channel_names = ["Fp1", "Fp2", "F7", "F3", "F4", "F8", "T3", "C3",
                            "C4", "T4", "T5", "P3", "P4", "T6", "O1", "O2"]
                preprocessing = \
                    EegPreprocessing(eeg_data,
                                    channel_names=channel_names,
                                    sampling_rate=128)
                preprocessing.filter_data()
                preprocessing.interpolate_bad_channels(bad_channels=['Fp2', 'P3', 'P4', 'T5'])
                preprocessing.rereferencing(referencing_value='average')
                #preprocessing.channel_wise_baseline_normalization(baseline_duration=baseline_duration)
                preprocessed_data = preprocessing.get_data()
                #preprocessed_data = \
                #    preprocessed_data[:, start_trim*128:-end_trim*128]
                file_name = os.path.join(eeg_output_path, eeg_file)
                np.savetxt(file_name, preprocessed_data)

        if gsr is True:
            gsr_output_path = os.path.join(participant_output_path, "gsr")
            if not os.path.exists(gsr_output_path):
                os.mkdir(gsr_output_path)
            gsr_path = os.path.join(input_path, participant, "gsr")
            gsr_files = os.listdir(gsr_path)
            for gsr_file in gsr_files:
                data_frame = pd.read_csv(os.path.join(gsr_path, gsr_file))
                gsr_data = data_frame.to_numpy()
                preprocessing = \
                    GsrPreprocessing(gsr_data[:,0],
                                    sampling_rate=128)
                preprocessing.gsr_noise_cancelation()
                #preprocessing.baseline_normalization(baseline_duration=baseline_duration)
                preprocessed_data = preprocessing.get_data()
                #preprocessed_data = \
                #    preprocessed_data[start_trim*128:-end_trim*128]
                file_name = os.path.join(gsr_output_path, gsr_file)
                np.savetxt(file_name, preprocessed_data)

        if ppg is True:
            ppg_output_path = os.path.join(participant_output_path, "ppg")
            if not os.path.exists(ppg_output_path):
                os.mkdir(ppg_output_path)
            ppg_path = os.path.join(input_path, participant, "ppg")
            ppg_files = os.listdir(ppg_path)
            for ppg_file in ppg_files:
                data_frame = pd.read_csv(os.path.join(ppg_path, ppg_file))
                ppg_data = data_frame.to_numpy()
                preprocessing = \
                    PpgPreprocessing(ppg_data[:, 0],
                                     sampling_rate=128)
                preprocessing.filtering()
                #preprocessing.baseline_normalization(baseline_duration=baseline_duration)
                preprocessed_data = preprocessing.get_data()
                #preprocessed_data = \
                #    preprocessed_data[start_trim*128:-end_trim*128]
                file_name = os.path.join(ppg_output_path, ppg_file)
                np.savetxt(file_name, preprocessed_data)

def make_like_deap_exp1_0(input_path, label_path, output_path):
    all_participants = os.listdir(input_path)
    all_participants.sort()

    for participant in all_participants:
        labels = pd.read_csv("{0}/{1}.csv".format(label_path, participant))
        trials_path = os.path.join(input_path, participant)

        eeg_trials = []
        eeg_trials_path = os.path.join(trials_path, "eeg")
        trials = os.listdir(eeg_trials_path)
        trials.sort()
        for trial in trials:
            eeg_trials.append(np.loadtxt(os.path.join(eeg_trials_path, trial)))

        gsr_trials = []
        gsr_trials_path = os.path.join(trials_path, "gsr")
        trials = os.listdir(gsr_trials_path)
        trials.sort()
        for trial in trials:
            gsr_trials.append(np.loadtxt(os.path.join(gsr_trials_path, trial)))

        ppg_trials = []
        ppg_trials_path = os.path.join(trials_path, "ppg")
        trials = os.listdir(ppg_trials_path)
        trials.sort()
        for trial in trials:
            ppg_trials.append(np.loadtxt(os.path.join(ppg_trials_path, trial)))

        trials_no, channels_no, samples_no = np.array(eeg_trials).shape
        all = np.zeros((trials_no, channels_no+2, samples_no))
        gsr_trials = np.array(gsr_trials)
        gsr_trials = gsr_trials.reshape((gsr_trials.shape[0], 1, gsr_trials.shape[1]))

        ppg_trials = np.array(ppg_trials)
        ppg_trials = ppg_trials.reshape((ppg_trials.shape[0], 1, ppg_trials.shape[1]))
        all[:, 0:16, :] = np.array(eeg_trials)
        all[:, 16:17, :] = gsr_trials
        all[:, 17:18, :] = ppg_trials

        all_labels = np.zeros((trials_no, 4))
        all_labels[:, 0] = labels["valence"]
        all_labels[:, 1] = labels["arousal"]
        all_labels[:, 2] = labels["dominance"]
        all_labels[:, 3] = labels["emotion"]
        print(all.shape, all_labels.shape)
        pickle.dump({'data':all, 'labels':all_labels},
                    open("{0}/{1}.pickle".format(output_path, participant), "wb"))


def exp1_1_same_length(sampling_rate=128):
    input_path = "exp1_1/preprocessed_data"
    output_path = "exp1_1/preprocessed_data_same_length"
    all_participants = os.listdir(input_path)
    all_participants.sort()
    for participant in all_participants:
        participant_output_path = os.path.join(output_path, participant)
        participant_path = os.path.join(input_path, participant)
        eeg_trials_path = os.path.join(participant_path, "eeg")
        eeg_trials = os.listdir(eeg_trials_path)
        eeg_trials.sort()

        gsr_trials_path = os.path.join(participant_path, "gsr")
        gsr_trials = os.listdir(gsr_trials_path)
        gsr_trials.sort()

        ppg_trials_path = os.path.join(participant_path, "ppg")
        ppg_trials = os.listdir(ppg_trials_path)
        ppg_trials.sort()

        eeg_output_path = os.path.join(participant_output_path, "eeg")
        if not os.path.exists(eeg_output_path):
            pathlib.Path(eeg_output_path).mkdir(parents=True, exist_ok=True)

        gsr_output_path = os.path.join(participant_output_path, "gsr")
        if not os.path.exists(gsr_output_path):
            pathlib.Path(gsr_output_path).mkdir(parents=True, exist_ok=True)

        ppg_output_path = os.path.join(participant_output_path, "ppg")
        if not os.path.exists(ppg_output_path):
            pathlib.Path(ppg_output_path).mkdir(parents=True, exist_ok=True)

        for i in range(len(eeg_trials)):
            print(eeg_trials[i], gsr_trials[i], ppg_trials[i])
            eeg_data = np.loadtxt(os.path.join(eeg_trials_path, eeg_trials[i]))
            eeg_samples = eeg_data.shape[1]

            gsr_data = np.loadtxt(os.path.join(gsr_trials_path, gsr_trials[i]))
            gsr_samples = gsr_data.shape[0]

            ppg_data = np.loadtxt(os.path.join(ppg_trials_path, ppg_trials[i]))
            ppg_samples = ppg_data.shape[0]

            print(eeg_data.shape, gsr_data.shape, ppg_data.shape)
            if gsr_samples < eeg_samples:
                while gsr_samples < eeg_samples:
                    gsr_data = np.append(gsr_data, gsr_data[gsr_samples-1])
                    gsr_samples = gsr_data.shape[0]
            elif gsr_samples > eeg_samples:
                gsr_data = gsr_data[0:eeg_samples]
                gsr_samples = eeg_samples

            if ppg_samples < eeg_samples:
                while ppg_samples < eeg_samples:
                    ppg_data = np.append(ppg_data, ppg_data[ppg_samples-1])
                    ppg_samples = ppg_data.shape[0]
            elif ppg_samples > eeg_samples:
                ppg_data = ppg_data[0:eeg_samples]
                ppg_samples = eeg_samples

            samples = int(ppg_samples/128) * 128
            eeg_data = eeg_data[:, 0:samples]
            gsr_data = gsr_data[0:samples]
            ppg_data = ppg_data[0:samples]
            print(eeg_data.shape, gsr_data.shape, ppg_data.shape)
            np.savetxt(os.path.join(eeg_output_path, eeg_trials[i]), eeg_data)
            np.savetxt(os.path.join(gsr_output_path, gsr_trials[i]), gsr_data)
            np.savetxt(os.path.join(ppg_output_path, ppg_trials[i]), ppg_data)




#preprocessing_exp1_1()
#preprocessing_exp1_0()
#make_like_deap_exp1_0("exp1_0/preprocessed_data", "exp1_0/prepared_labels", "exp1_0/pickled")
exp1_1_same_length()
