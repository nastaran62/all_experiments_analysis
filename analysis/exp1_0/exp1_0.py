from subject_dependent import subject_dependent_evaluation
import numpy as np
import pickle
from cross_subject_manual import kfold_evaluation 
from subject_independent import subject_independent_cross_validation
from exp1_0.feature_extraction import partitioning_and_getting_features

PARTICIPANTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23]
#PARTICIPANTS = [9, 11, 12, 14, 15, 16, 17, 18, 19 ,20 ,21, 22, 23]

def prepare_data(label_type="arousal", window_size=0):
    ''' 
    input_path = "../experimental_data/exp1_0/preprocessed_data"
    label_path = "../experimental_data/exp1_0/prepared_labels"
    all_eeg, all_gsr, all_ppg, all_emotions, all_arousals, all_valences, all_dominances = \
        partitioning_and_getting_features(input_path, label_path, window_size=window_size)
    
    pickle.dump(all_eeg, open("exp1_0/data/eeg.pickle", "wb"))
    pickle.dump(all_gsr, open("exp1_0/data/gsr.pickle", "wb"))
    pickle.dump(all_ppg, open("exp1_0/data/ppg.pickle", "wb"))
    pickle.dump((all_emotions, all_arousals, all_valences, all_dominances), open("exp1_0/data/labels.pickle", "wb"))
    '''
    all_eeg = pickle.load(open("exp1_0/data/eeg.pickle", "rb"))
    all_gsr = pickle.load(open("exp1_0/data/gsr.pickle", "rb"))
    all_ppg = pickle.load(open("exp1_0/data/ppg.pickle", "rb"))
    (all_emotions, all_arousals, all_valences, all_dominances) = \
        pickle.load(open("exp1_0/data/labels.pickle", "rb"))
    if label_type == "arousal":
        labels = all_arousals
    elif label_type == "valence":
        labels = all_valences
    elif label_type == "dominance":
        labels = all_dominances
    else:
        labels = all_emotions

    return all_eeg, all_gsr, all_ppg, labels

def cross_subject(label_type="arousal", window_size=0):
    def make_np_array(data):
        trials = []
        p = 0
        for participant in data:
            for trial in participant:
                if p in PARTICIPANTS:
                    for window in trial:
                        trials.append(np.array(window))
            p += 1
        return np.array(trials)

    all_eeg, all_gsr, all_ppg, all_labels = \
        prepare_data(label_type=label_type, window_size=window_size)
    print(np.array(all_eeg).shape, np.array(all_gsr).shape, np.array(all_ppg).shape)
    eeg = make_np_array(all_eeg)
    print(eeg.shape, "********************************")
    gsr = make_np_array(all_gsr)
    print(gsr.shape, "********************************")
    ppg = make_np_array(all_ppg)
    print(ppg.shape, "********************************")

    labels = make_np_array(all_labels)
    print(labels.shape, "********************************")
    eeg_accuracy, gsr_accuracy, ppg_accuracy, fusion_accuracy, \
        eeg_fscore, gsr_fscore, ppg_fscore, fusion_fscore = \
            kfold_evaluation(eeg, gsr, ppg, labels, k=5)
    print("eeg_accuracy: ", eeg_accuracy, "eeg_fscore: ", eeg_fscore)
    print("gsr_accuracy: ", gsr_accuracy, "gsr_fscore: ", gsr_fscore)
    print("ppg_accuracy: ", ppg_accuracy, "ppg_fscore: ", ppg_fscore)
    print("fusion_accuracy: ", fusion_accuracy, "fusion_fscore: ", fusion_fscore)

def subject_dependent(label_type="arousal", window_size=0):
    def prepare_data_for_subject_dependent(data):
        all_parts = []
        for trial in data:
            for window in trial:
                all_parts.append(window)
        return np.array(all_parts)
    
    all_eeg, all_gsr, all_ppg, all_labels = \
        prepare_data(label_type=label_type, window_size=window_size)
    subject_dependent_evaluation(all_eeg, all_gsr, all_ppg, all_labels, 
                                 PARTICIPANTS,
                                 prepare_data_for_subject_dependent,
                                 fold=4)

def subject_independent(label_type="arousal", window_size=0):
    def make_train_test_set(data, train_participants, test_participants):
        test_trials = []
        train_trials = []
        p = 0
        for participant in data:
            for trial in participant:
                for window in trial:
                    if p in train_participants:
                        train_trials.append(np.array(window))
                    elif p in test_participants:
                        test_trials.append(np.array(window))
            p += 1
        return np.array(train_trials), np.array(test_trials)

    all_eeg, all_gsr, all_ppg, all_labels = \
        prepare_data(label_type=label_type, window_size=window_size)
    subject_independent_cross_validation(all_eeg, all_gsr, all_ppg, all_labels, 
                                         PARTICIPANTS,
                                         make_train_test_set,
                                         fold=3)



