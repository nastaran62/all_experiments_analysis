import numpy as np
from sklearn.model_selection import train_test_split
import csv

from exp1_0.exp1_0 import prepare_data as exp1_0_prepare_data
from exp1_1.exp1_1 import prepare_data as exp1_1_prepare_data
from classification import random_forest

PARTICIPANTS = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23]
PARTICIPANT = 1
WINDOW_SIZE = 1
LABEL_TYPE = "arousal"

# 39 for exp1-0 should be removed (bad gsr, ppg data)
# 7 and 18 for exp-1 should be removed (eeg has more samples)
exp1_0_all_eeg, exp1_0_all_gsr, exp1_0_all_ppg , exp1_0_all_labels = \
    exp1_0_prepare_data(calculate=True,
                        window_size=WINDOW_SIZE,
                        label_type=LABEL_TYPE)

exp1_1_all_eeg, exp1_1_all_gsr, exp1_1_all_ppg , exp1_1_all_labels = \
    exp1_1_prepare_data(calculate=False,
                        window_size=WINDOW_SIZE,
                        label_type=LABEL_TYPE)

def prepare_exp1_0(data, label=False):
    if label is True:
        data = data.reshape(-1)
        return data
    else:
        print(data.shape, "*************************")
        data = data.reshape(-1, data.shape[-1])
        print(data.shape)
        return data

def prepare_exp1_1(data, label=False, participants=PARTICIPANTS):
    #label parameter is for compatibility with other codes
    all_trials = []
    p = 0
    for participant in data:
        if p in participants:
            for trial in participant:
                for part in trial:
                    for window in part:
                        all_trials.append(window)
        p += 1
    print(len(all_trials))
    return np.array(all_trials) 


def subject_dependent_cross_exp(participant,
                                exp10_writer,
                                exp11_writer,
                                exp10_11_writer,
                                exp11_10_writer):
    eeg1_0 = prepare_exp1_0(exp1_0_all_eeg[participant])
    gsr1_0 = prepare_exp1_0(exp1_0_all_gsr[participant])
    ppg1_0 = prepare_exp1_0(exp1_0_all_ppg[participant])
    labels1_0 = prepare_exp1_0(exp1_0_all_labels[participant], label=True)

    print(eeg1_0.shape, gsr1_0.shape, ppg1_0.shape, labels1_0.shape)

    eeg1_1 = prepare_exp1_1(exp1_1_all_eeg, participants=[participant])
    gsr1_1 = prepare_exp1_1(exp1_1_all_gsr, participants=[participant])
    ppg1_1 = prepare_exp1_1(exp1_1_all_ppg, participants=[participant])
    labels1_1 = prepare_exp1_1(exp1_1_all_labels, label=True, participants=[participant])
    print(eeg1_1.shape, gsr1_1.shape, ppg1_1.shape, labels1_1.shape)

    print("EEG EEG EEG EEG")
    eeg_acc_10, eeg_fscore_10, eeg_acc_11, eeg_fscore_11 = \
        in_dataset_classification(eeg1_0, labels1_0, eeg1_1, labels1_1)

    print("GSR GSR GSR GSR")
    gsr_acc_10, gsr_fscore_10, gsr_acc_11, gsr_fscore_11 = \
        in_dataset_classification(gsr1_0, labels1_0, gsr1_1, labels1_1)

    print("PPG PPG PPG PPG")
    ppg_acc_10, ppg_fscore_10, ppg_acc_11, ppg_fscore_11 = \
        in_dataset_classification(ppg1_0, labels1_0, ppg1_1, labels1_1)
    
    row = [participant, eeg_acc_10, gsr_acc_10, ppg_acc_10,
           eeg_fscore_10, gsr_fscore_10, ppg_fscore_10]
    exp10_writer.writerow(row)

    row = [participant, eeg_acc_11, gsr_acc_11, ppg_acc_11, 
        eeg_fscore_11, gsr_fscore_11, ppg_fscore_11]
    exp11_writer.writerow(row)

    print("EEG EEG EEG EEG")
    eeg_acc_10_11, eeg_fscore_10_11, eeg_acc_11_10, eeg_fscore_11_10 = \
        cross_dataset_classification(eeg1_0, labels1_0, eeg1_1, labels1_1)

    print("GSR GSR GSR GSR")
    gsr_acc_10_11, gsr_fscore_10_11, gsr_acc_11_10, gsr_fscore_11_10 = \
        cross_dataset_classification(gsr1_0, labels1_0, gsr1_1, labels1_1)

    print("PPG PPG PPG PPG")
    ppg_acc_10_11, ppg_fscore_10_11, ppg_acc_11_10, ppg_fscore_11_10 = \
        cross_dataset_classification(ppg1_0, labels1_0, ppg1_1, labels1_1)
    
    row = [participant, eeg_acc_10_11, gsr_acc_10_11, ppg_acc_10_11,
           eeg_fscore_10_11, gsr_fscore_10_11, ppg_fscore_10_11]
    exp10_11_writer.writerow(row)

    row = [participant, eeg_acc_11_10, gsr_acc_11_10, ppg_acc_11_10,
        eeg_fscore_11_10, gsr_fscore_11_10, ppg_fscore_11_10]
    exp11_10_writer.writerow(row)


def in_dataset_classification(data0, labels0, data1, labels1):
    train_x0, test_x0, train_y0, test_y0 = \
        train_test_split(np.array(data0),
                        np.array(labels0),
                        test_size=0.3,
                        random_state=42,
                        stratify=labels0)
    
    train_x1, test_x1, train_y1, test_y1 = \
        train_test_split(np.array(data1),
                        np.array(labels1),
                        test_size=0.3,
                        random_state=42,
                        stratify=labels1)
    
    print("Train and test on exp1-0")
    acc_10, fscore_10 = \
        random_forest(train_x0, test_x0, train_y0, test_y0)
    print("Train and test on exp1-1")
    acc_10, fscore_10 = \
        random_forest(train_x1, test_x1, train_y1, test_y1)
    return acc_10, fscore_10, acc_10, fscore_10 

def cross_dataset_classification(data0, labels0, data1, labels1):
    print("train on exp1-0 and test on exp1-1")
    acc_10_11, fscore_10_11 = \
        random_forest(data0, data1, labels0, labels1)

    print("train on exp1-1 and test on exp1-0")
    acc_11_10, fscore_11_10 = \
        random_forest(data1, data0, labels1, labels0)

    return acc_10_11, fscore_10_11, acc_11_10, fscore_11_10

def cross_dataset_evaluation():
    exp10_csv_file = open("exp10.csv", 'a')
    exp10_writer = csv.writer(exp10_csv_file)

    exp11_csv_file = open("exp11.csv", 'a')
    exp11_writer = csv.writer(exp11_csv_file)

    exp10_11_csv_file = open("exp10_11.csv", 'a')
    exp10_11_writer = csv.writer(exp10_11_csv_file)

    exp11_10_csv_file = open("exp11_10.csv", 'a')
    exp11_10_writer = csv.writer(exp11_10_csv_file)

    row = ["participant", "eeg_acc", "gsr_acc", "ppg_acc",
           "eeg_fscore", "gsr_fscore", "ppg_fscore"]
    
    exp10_writer.writerow(row)
    exp11_writer.writerow(row)
    exp10_11_writer.writerow(row)
    exp11_10_writer.writerow(row)
    participants = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23]
    for participant in participants:
        print(participant, "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        subject_dependent_cross_exp(participant,
                                    exp10_writer,
                                    exp11_writer,
                                    exp10_11_writer,
                                    exp11_10_writer)
    exp10_csv_file.close()
    exp11_csv_file.close()
    exp10_11_csv_file.close()
    exp11_10_csv_file.close()


    
    
    