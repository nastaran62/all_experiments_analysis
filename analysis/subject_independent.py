import numpy as np
from sklearn.utils import shuffle
from classification import feature_selection_random_forest, svm_classification, random_forest
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score

def subject_independent_cross_validation(all_eeg, all_gsr, all_ppg, all_labels, 
                                         participants,
                                         make_train_test_set,
                                         fold=4):
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
         
        print("eeg eeg eeg")
        accuracy, fscore, eeg_preds = \
            random_forest(eeg_train, eeg_test, train_labels, test_labels)
        all_eeg_accuracy.append(accuracy)
        all_eeg_fscore.append(fscore)
        print("gsr gsr gsr")
        accuracy, fscore, gsr_preds = \
            feature_selection_random_forest(gsr_train, gsr_test, train_labels, test_labels)
        all_gsr_accuracy.append(accuracy)
        all_gsr_fscore.append(fscore)
        
        print("ppg ppg ppg")
        accuracy, fscore, ppg_preds = \
            random_forest(ppg_train, ppg_test, train_labels, test_labels)
        all_ppg_accuracy.append(accuracy)
        all_ppg_fscore.append(fscore)
        i = 0
        preds_fusion = []
        for i in range(len(eeg_preds)):
            if eeg_preds[i] + ppg_preds[i] + gsr_preds[i] > 1 :
                preds_fusion.append(1)
            else:
                preds_fusion.append(0)
    
        acc = accuracy_score(preds_fusion, test_labels)
        print(classification_report(test_labels, preds_fusion))
        precision, recall, f_score, support = \
            precision_recall_fscore_support(test_labels,
                                            preds_fusion,
                                            average='weighted')

        all_fusion_accuracy.append(acc)
        all_fusion_fscore.append(f_score)
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
