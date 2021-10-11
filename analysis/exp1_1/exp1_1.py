from sklearn.utils import shuffle
from subject_dependent import subject_dependent_evaluation
import numpy as np
import pickle
from cross_subject_manual import kfold_evaluation, shuffled_kfold_evaluation
from subject_independent import subject_independent_cross_validation
from exp1_1.feature_extraction import partitioning_and_getting_features, partitioning_and_getting_features_liwc
import csv

PARTICIPANTS = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
# for 7 and 18 eeg was longer than other data
#PARTICIPANTS = [9, 11, 12, 14, 15, 16, 17, 18, 19 ,20 ,21, 22, 23]

def prepare_data(label_type="arousal", window_size=0, calculate=False):

    input_path = "../experimental_data/exp1_1/preprocessed_data_same_length"
    label_path = "../experimental_data/exp1_1/prepared_labels"
    feature_path = "../experimental_data/exp1_1/features"
    all_eeg, all_gsr, all_ppg, all_emotions, all_arousals, all_valences, all_intensity = \
        partitioning_and_getting_features(input_path, label_path, feature_path, trial_size=60, window_size=window_size, calculate=calculate)

    

    if label_type == "arousal":
        labels = all_arousals
    elif label_type == "valence":
        labels = all_valences
    else:
        labels = all_emotions

    return all_eeg, all_gsr, all_ppg, labels

def prepare_data_liwc(label_type="arousal", window_size=0, calculate=False):
    '''
    based on LIWC labels
    '''

    input_path = "../experimental_data/exp1_1/preprocessed_data_same_length"
    label_path = "../experimental_data/exp1_1/prepared_liwc_labels"
    feature_path = "../experimental_data/exp1_1/features_liwc"
    all_eeg, all_gsr, all_ppg, all_emotions, all_arousals, all_valences, all_intensity = \
        partitioning_and_getting_features_liwc(input_path, label_path, feature_path, trial_size=60, window_size=window_size, calculate=calculate)

    if label_type == "liwc_valence":
        labels = all_valences
    else:
        labels = all_emotions

    return all_eeg, all_gsr, all_ppg, labels

def cross_subject(label_type="arousal", window_size=0, calculate=False, shuffle=False, fold=5):
    def make_np_array(data, label=False):
        trials = []
        p = 0
        sum = 0
        for participant in data:
            if p in PARTICIPANTS:
                for trial in participant:
                    for part in trial:
                        windows = []
                        if label is True:
                            sum += len(part)
                            print(len(part))
                        for window in part:
                            windows.append(np.array(window))
                        trials.append(windows)
            p += 1
        return np.array(trials)

    all_eeg, all_gsr, all_ppg, all_labels = \
        prepare_data(label_type=label_type, window_size=window_size, calculate=calculate)
    eeg = make_np_array(all_eeg)
    print(eeg.shape, "********************************")
    gsr = make_np_array(all_gsr)
    print(gsr.shape, "********************************")
    ppg = make_np_array(all_ppg)
    print(ppg.shape, "********************************")

    labels = make_np_array(all_labels, label=True)
    print(labels.shape, "********************************")

    if shuffle is True:
        eeg_accuracy, gsr_accuracy, ppg_accuracy, fusion_accuracy, efusion_accuracy, \
            eeg_fscore, gsr_fscore, ppg_fscore, fusion_fscore, efusion_fscore = \
                shuffled_kfold_evaluation(eeg, gsr, ppg, labels, k=fold, model_path="exp1_1/models", label_type=label_type)
    else:
        eeg_accuracy, gsr_accuracy, ppg_accuracy, fusion_accuracy, efusion_accuracy, \
            eeg_fscore, gsr_fscore, ppg_fscore, fusion_fscore, efusion_fscore = \
                kfold_evaluation(eeg, gsr, ppg, labels, k=fold, model_path="exp1_1/models", label_type=label_type)
    print("eeg_accuracy: ", eeg_accuracy, "eeg_fscore: ", eeg_fscore)
    print("gsr_accuracy: ", gsr_accuracy, "gsr_fscore: ", gsr_fscore)
    print("ppg_accuracy: ", ppg_accuracy, "ppg_fscore: ", ppg_fscore)
    print("fusion_accuracy: ", fusion_accuracy, "fusion_fscore: ", fusion_fscore)
    print("efusion_accuracy: ", efusion_accuracy, "efusion_fscore: ", efusion_fscore)


def subject_dependent(label_type="arousal", window_size=0, calculate=False, shuffle=False, fold=3):
    def prepare_data_for_subject_dependent(data, label=False):
        #label parameter is for compatibility with other codes
        all_parts = []
        for trial in data:
            for part in trial:
                all_windows = []
                for window in part:
                    all_windows.append(window)
                all_parts.append(all_windows)
        return np.array(all_parts)

    all_eeg, all_gsr, all_ppg, all_labels = \
        prepare_data(label_type=label_type, window_size=window_size, calculate=calculate)
    subject_dependent_evaluation(all_eeg, all_gsr, all_ppg, all_labels,
                                 PARTICIPANTS,
                                 prepare_data_for_subject_dependent,
                                 fold=fold,
                                 model_path="exp1_1/models",
                                 shuffle=shuffle,
                                 label_type=label_type)

def subject_independent(label_type="arousal", window_size=0, calculate=False, shuffle=False, fold=3):
    def make_train_test_set(data, train_participants, test_participants, label=False):
        test_trials = []
        train_trials = []
        p = 0
        for participant in data:
            for trial in participant:
                for part in trial:
                    for window in part:
                        if p in train_participants:
                            train_trials.append(np.array(window))
                        elif p in test_participants:
                            test_trials.append(np.array(window))
            p += 1
        return np.array(train_trials), np.array(test_trials)

    all_eeg, all_gsr, all_ppg, all_labels = \
        prepare_data(label_type=label_type, window_size=window_size, calculate=calculate)

    subject_independent_cross_validation(all_eeg, all_gsr, all_ppg, all_labels,
                                         PARTICIPANTS,
                                         make_train_test_set,
                                         fold=fold,
                                         model_path="exp1_1/models",
                                         label_type=label_type)
