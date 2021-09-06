import numpy as np
from sklearn.model_selection import KFold
from classification import feature_selection_random_forest, svm_classification, random_forest
#from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score

    
def kfold_evaluation(eeg, gsr, ppg, labels, k=5):
    # Shuffling
    permutation = np.random.permutation(labels.shape[0])
    eeg = eeg[permutation, :]
    gsr = gsr[permutation, :]
    ppg = ppg[permutation, :]
    labels = labels[permutation]
    
    kf = KFold(n_splits=k, shuffle=True, random_state=100)
    all_eeg_accuracy = []
    all_gsr_accuracy = []
    all_ppg_accuracy = []
    all_fusion_accuracy = []
    all_eeg_fscore = []
    all_gsr_fscore = []
    all_ppg_fscore = []
    all_fusion_fscore = []
    for train_index, test_index in kf.split(labels):
        eeg_train, eeg_test = eeg[train_index, :], eeg[test_index, :]
        gsr_train, gsr_test = gsr[train_index, :], gsr[test_index, :]
        ppg_train, ppg_test = ppg[train_index, :], ppg[test_index, :]
        train_labels, test_labels = labels[train_index], labels[test_index]
        
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
            svm_classification(ppg_train, ppg_test, train_labels, test_labels)
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

    eeg_accuracy = np.mean(np.array(all_eeg_accuracy))
    eeg_fscore = np.mean(np.array(all_eeg_fscore))
    gsr_accuracy = np.mean(np.array(all_gsr_accuracy))
    gsr_fscore = np.mean(np.array(all_gsr_fscore))
    ppg_accuracy = np.mean(np.array(all_ppg_accuracy))
    ppg_fscore = np.mean(np.array(all_ppg_fscore))
    fusion_accuracy = np.mean(np.array(all_fusion_accuracy))
    fusion_fscore = np.mean(np.array(all_fusion_fscore))

    return eeg_accuracy, gsr_accuracy, ppg_accuracy, fusion_accuracy, \
           eeg_fscore, gsr_fscore, ppg_fscore, fusion_fscore

