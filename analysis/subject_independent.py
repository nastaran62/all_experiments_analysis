import numpy as np
import os
import pathlib
from classification import multimodal_classification, voting_fusion

def subject_independent_cross_validation(all_eeg, all_gsr, all_ppg, all_labels, 
                                         participants,
                                         make_train_test_set,
                                         fold=4,
                                         model_path="."):
    if not os.path.exists(model_path):
        pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
    fold = fold
    start = 0
    end = start
    all_eeg_accuracy = []
    all_gsr_accuracy = []
    all_ppg_accuracy = []
    all_fusion_accuracy = []
    all_eeg_fscore = []
    all_gsr_fscore = []
    all_ppg_fscore = []
    all_fusion_fscore = []
    while start < len(participants):
        end = end + fold
        if end > len(participants):
            end = len(participants)
        test_participants = participants[start:end]
        train_participants = []
        for item in participants:
            if item not in test_participants:
                train_participants.append(item)
        
        print(test_participants, "*******************")
        eeg_train, eeg_test = \
            make_train_test_set(all_eeg, train_participants, test_participants)

        gsr_train, gsr_test = \
            make_train_test_set(all_gsr, train_participants, test_participants)

        ppg_train, ppg_test = \
            make_train_test_set(all_ppg, train_participants, test_participants)
        
        train_labels, test_labels = \
            make_train_test_set(all_labels, train_participants, test_participants)
            
        # Shuffling
        permutation = np.random.permutation(train_labels.shape[0])
        eeg_train = eeg_train[permutation, :]
        gsr_train = gsr_train[permutation, :]
        ppg_train = ppg_train[permutation, :]
        train_labels = train_labels[permutation]
         
        eeg_parameters = \
            eeg_train, eeg_test, "random_forest", os.path.join(model_path, "eeg_model.pickle")
        gsr_parameters = \
            gsr_train, gsr_test, "random_forest", os.path.join(model_path, "gsr_model.pickle")
        ppg_parameters = \
            ppg_train, ppg_test, "random_forest", os.path.join(model_path, "ppg_model.pickle")
        
        eeg_result, gsr_result, ppg_result = \
            multimodal_classification(train_labels,
                                      test_labels,
                                      eeg=eeg_parameters,
                                      gsr=gsr_parameters,
                                      ppg=ppg_parameters)
        eeg_accuracy, eeg_fscore, eeg_preds, eeg_probabilities = eeg_result
        gsr_accuracy, gsr_fscore, gsr_preds, gsr_probabilities = gsr_result
        ppg_accuracy, ppg_fscore, ppg_preds, ppg_probabilities = ppg_result

        all_eeg_accuracy.append(eeg_accuracy)
        all_eeg_fscore.append(eeg_fscore)
        
        all_gsr_accuracy.append(gsr_accuracy)
        all_gsr_fscore.append(gsr_fscore)
        
        all_ppg_accuracy.append(ppg_accuracy)
        all_ppg_fscore.append(ppg_fscore)

        fusion_accuracy, fusion_fscore = \
            voting_fusion(eeg_preds, gsr_preds, ppg_preds, test_labels)
        all_fusion_accuracy.append(fusion_accuracy)
        all_fusion_fscore.append(fusion_fscore)
        start = end

    eeg_accuracy = np.mean(np.array(all_eeg_accuracy))
    eeg_fscore = np.mean(np.array(all_eeg_fscore))
    gsr_accuracy = np.mean(np.array(all_gsr_accuracy))
    gsr_fscore = np.mean(np.array(all_gsr_fscore))
    ppg_accuracy = np.mean(np.array(all_ppg_accuracy))
    ppg_fscore = np.mean(np.array(all_ppg_fscore))
    fusion_accuracy = np.mean(np.array(all_fusion_accuracy))
    fusion_fscore = np.mean(np.array(all_fusion_fscore))

    print("eeg_accuracy: ", eeg_accuracy, "eeg_fscore: ", eeg_fscore)
    print("gsr_accuracy: ", gsr_accuracy, "gsr_fscore: ", gsr_fscore)
    print("ppg_accuracy: ", ppg_accuracy, "ppg_fscore: ", ppg_fscore)
    print("fusion_accuracy: ", fusion_accuracy, "fusion_fscore: ", fusion_fscore)

def subject_independent_lstm_cross_validation(all_eeg, all_gsr, all_ppg, all_labels, 
                                         participants,
                                         make_train_test_set,
                                         fold=4,
                                         model_path="."):
    if not os.path.exists(model_path):
        pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
    fold = fold
    start = 0
    end = start
    all_eeg_accuracy = []
    all_gsr_accuracy = []
    all_ppg_accuracy = []
    all_fusion_accuracy = []
    all_eeg_fscore = []
    all_gsr_fscore = []
    all_ppg_fscore = []
    all_fusion_fscore = []
    while start < len(participants):
        end = end + fold
        if end > len(participants):
            end = len(participants)
        test_participants = participants[start:end]
        train_participants = []
        for item in participants:
            if item not in test_participants:
                train_participants.append(item)
        
        print(test_participants, "*******************")
        eeg_train, eeg_test = \
            make_train_test_set(all_eeg, train_participants, test_participants)

        gsr_train, gsr_test = \
            make_train_test_set(all_gsr, train_participants, test_participants)

        ppg_train, ppg_test = \
            make_train_test_set(all_ppg, train_participants, test_participants)
        
        train_labels, test_labels = \
            make_train_test_set(all_labels, train_participants, test_participants, label=True)
        
        print(eeg_train.shape, eeg_test.shape)
        print(gsr_train.shape, gsr_test.shape)
        print(ppg_train.shape, ppg_test.shape)
        print(train_labels.shape, test_labels.shape)
            
        # Shuffling
        permutation = np.random.permutation(train_labels.shape[0])
        eeg_train = eeg_train[permutation, :]
        gsr_train = gsr_train[permutation, :]
        ppg_train = ppg_train[permutation, :]
        train_labels = train_labels[permutation]
        
        eeg_parameters = \
            eeg_train, eeg_test, "lstm", os.path.join(model_path, "eeg_lstm.h5")
        gsr_parameters = \
            gsr_train, gsr_test, "lstm", os.path.join(model_path, "gsr_lstm.h5")
        ppg_parameters = \
            ppg_train, ppg_test, "lstm", os.path.join(model_path, "ppg_lstm.h5")
        
        eeg_result, gsr_result, ppg_result = \
            multimodal_classification(train_labels,
                                      test_labels,
                                      eeg=eeg_parameters,
                                      gsr=gsr_parameters,
                                      ppg=ppg_parameters)
        eeg_accuracy, eeg_fscore, eeg_preds, eeg_probabilities = eeg_result
        gsr_accuracy, gsr_fscore, gsr_preds, gsr_probabilities = gsr_result
        ppg_accuracy, ppg_fscore, ppg_preds, ppg_probabilities = ppg_result

        all_eeg_accuracy.append(eeg_accuracy)
        all_eeg_fscore.append(eeg_fscore) 
        
        all_gsr_accuracy.append(gsr_accuracy)
        all_gsr_fscore.append(gsr_fscore)
        
        all_ppg_accuracy.append(ppg_accuracy)
        all_ppg_fscore.append(ppg_fscore)
 
        fusion_accuracy, fusion_fscore = \
            voting_fusion(eeg_preds, gsr_preds, ppg_preds, test_labels)
        all_fusion_accuracy.append(fusion_accuracy)
        all_fusion_fscore.append(fusion_fscore)
        start = end

    print(all_eeg_accuracy)
    print(all_gsr_accuracy)
    print(all_ppg_accuracy)
    eeg_accuracy = np.mean(np.array(all_eeg_accuracy))
    eeg_fscore = np.mean(np.array(all_eeg_fscore))
    gsr_accuracy = np.mean(np.array(all_gsr_accuracy))
    gsr_fscore = np.mean(np.array(all_gsr_fscore))
    ppg_accuracy = np.mean(np.array(all_ppg_accuracy))
    ppg_fscore = np.mean(np.array(all_ppg_fscore))
    fusion_accuracy = np.mean(np.array(all_fusion_accuracy))
    fusion_fscore = np.mean(np.array(all_fusion_fscore))


    print("eeg_accuracy: ", eeg_accuracy, "eeg_fscore: ", eeg_fscore)
    print("gsr_accuracy: ", gsr_accuracy, "gsr_fscore: ", gsr_fscore)
    print("ppg_accuracy: ", ppg_accuracy, "ppg_fscore: ", ppg_fscore)
    print("fusion_accuracy: ", fusion_accuracy, "fusion_fscore: ", fusion_fscore)
