from subject_dependent import subject_dependent_evaluation, subject_dependent_lstm_evaluation
import numpy as np
import pickle
from cross_subject_manual import kfold_evaluation, lstm_kfold_evaluation
from subject_independent import subject_independent_cross_validation, subject_independent_lstm_cross_validation
from exp1_0.feature_extraction import partitioning_and_getting_features

PARTICIPANTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21, 23]
#PARTICIPANTS = [9, 11, 12, 14, 15, 16, 17, 18, 19 ,20 ,21, 22, 23]

def prepare_data(label_type="arousal", window_size=0, calculate=False): 
    print("here")
    input_path = "../experimental_data/exp1_0/preprocessed_data"
    label_path = "../experimental_data/exp1_0/prepared_labels"
    feature_path =  "../experimental_data/exp1_0/features"
    all_eeg, all_gsr, all_ppg, all_emotions, all_arousals, all_valences, all_dominances = \
        partitioning_and_getting_features(input_path,
                                            label_path,
                                            feature_path,
                                            window_size=window_size,
                                            calculate=calculate)

    if label_type == "arousal":
        labels = all_arousals
    elif label_type == "valence":
        labels = all_valences
    elif label_type == "dominance":
        labels = all_dominances
    else:
        labels = all_emotions

    return all_eeg, all_gsr, all_ppg, labels

def cross_subject(label_type="arousal", window_size=0, calculate=False, fold=5):
    def prepare_labels(labels):
        trials = []
        p = 0
        for participant in labels:
            for trial in participant:
                if p in PARTICIPANTS:
                    trials.append(trial[0])
            p += 1
        return np.array(trials)

    all_eeg, all_gsr, all_ppg, all_labels = \
        prepare_data(label_type=label_type, window_size=window_size, calculate=calculate)
    print(all_eeg.shape)
    eeg = all_eeg.reshape(-1, *all_eeg.shape[-2:])
    print(eeg.shape, "********************************")
    gsr = all_gsr.reshape(-1, *all_gsr.shape[-2:])
    print(gsr.shape, "********************************")
    ppg = all_ppg.reshape(-1, *all_ppg.shape[-2:])
    print(ppg.shape, "********************************")

    labels = prepare_labels(all_labels)
    print(labels.shape, "********************************")
    eeg_accuracy, gsr_accuracy, ppg_accuracy, fusion_accuracy, efusion_accuracy,\
        eeg_fscore, gsr_fscore, ppg_fscore, fusion_fscore, efusion_fscore = \
            lstm_kfold_evaluation(eeg, gsr, ppg, labels, k=fold)
    print("eeg_accuracy: ", eeg_accuracy, "eeg_fscore: ", eeg_fscore)
    print("gsr_accuracy: ", gsr_accuracy, "gsr_fscore: ", gsr_fscore)
    print("ppg_accuracy: ", ppg_accuracy, "ppg_fscore: ", ppg_fscore)
    print("fusion_accuracy: ", fusion_accuracy, "fusion_fscore: ", fusion_fscore)
    print("equal fusion_accuracy: ", efusion_accuracy, "equal fusion_fscore: ", efusion_fscore)

def subject_dependent(label_type="arousal", window_size=0, calculate=False, fold=2):
    def prepare_data_for_subject_dependent(data, label=False):
        if label is True:
            all_parts = []
            for trial in data:
                all_parts.append(trial[0])
            return np.array(all_parts)
        else:
            data.reshape(-1, *data.shape[:-2])
            return data
    
    all_eeg, all_gsr, all_ppg, all_labels = \
        prepare_data(label_type=label_type, window_size=window_size, calculate=calculate)
    subject_dependent_lstm_evaluation(all_eeg, all_gsr, all_ppg, all_labels, 
                                 PARTICIPANTS,
                                 prepare_data_for_subject_dependent,
                                 fold=fold)

def subject_independent(label_type="arousal", window_size=0, fold=3, calculate=False):
    def make_train_test_set(data, train_participants, test_participants, label=False):
        test_trials = []
        train_trials = []
        p = 0
        for participant in data:
            for trial in participant:
                if p in train_participants:
                    if label is True:
                        train_trials.append(trial[0])
                    else:
                        train_trials.append(np.array(trial))
                elif p in test_participants:
                    if label is True:
                        test_trials.append(trial[0])
                    else:
                        test_trials.append(np.array(trial))
            p += 1
        return np.array(train_trials), np.array(test_trials)

    all_eeg, all_gsr, all_ppg, all_labels = \
        prepare_data(label_type=label_type, window_size=window_size, calculate=calculate)
    subject_independent_lstm_cross_validation(all_eeg, all_gsr, all_ppg, all_labels, 
                                         PARTICIPANTS,
                                         make_train_test_set,
                                         fold=fold)