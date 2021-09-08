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


#preprocessing_exp1_1()
preprocessing_exp1_0()