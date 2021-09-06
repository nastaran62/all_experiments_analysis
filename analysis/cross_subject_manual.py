import numpy as np
import multiprocessing
from sklearn.model_selection import KFold
from classification import ModalityClassification, feature_selection_random_forest, svm_classification, random_forest
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
        eeg_queue = multiprocessing.Queue()
        eeg_classifier = \
            ModalityClassification(eeg_train,
                                   eeg_test,
                                   train_labels,
                                   test_labels,
                                   eeg_queue,
                                   type="random_forest",
                                   model_name="eeg.pickle")
        
        gsr_queue = multiprocessing.Queue()
        gsr_classifier = \
            ModalityClassification(gsr_train,
                                   gsr_test,
                                   train_labels,
                                   test_labels,
                                   gsr_queue,
                                   type="random_forest",
                                   model_name="gsr.pickle")
        
        ppg_queue = multiprocessing.Queue()
        ppg_classifier = \
            ModalityClassification(ppg_train,
                                   ppg_test,
                                   train_labels,
                                   test_labels,
                                   ppg_queue,
                                   type="random_forest",
                                   model_name="ppg.pickle")
        
        eeg_classifier.start()
        gsr_classifier.start()
        ppg_classifier.start()
        eeg_accuracy, eeg_fscore, eeg_preds, eeg_probabilities = eeg_queue.get()
        gsr_accuracy, gsr_fscore, gsr_preds, gsr_probabilities = gsr_queue.get()
        ppg_accuracy, ppg_fscore, ppg_preds, ppg_probabilities = ppg_queue.get()

        eeg_classifier.join()
        gsr_classifier.join()
        ppg_classifier.join()
        
        all_eeg_accuracy.append(eeg_accuracy)
        all_eeg_fscore.append(eeg_fscore) 
        
        all_gsr_accuracy.append(gsr_accuracy)
        all_gsr_fscore.append(gsr_fscore)
        

        all_ppg_accuracy.append(ppg_accuracy)
        all_ppg_fscore.append(ppg_fscore)
 
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


def lstm_kfold_evaluation(eeg, gsr, ppg, labels, k=5):
    # Shuffling
    permutation = np.random.permutation(labels.shape[0])
    eeg = eeg[permutation, :, :]
    gsr = gsr[permutation, :, :]
    ppg = ppg[permutation, :, :]
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
        eeg_queue = multiprocessing.Queue()
        eeg_classifier = \
            ModalityClassification(eeg_train,
                                   eeg_test,
                                   train_labels,
                                   test_labels,
                                   eeg_queue,
                                   type="lstm",
                                   model_name="eeg_lstm.h5")
        
        gsr_queue = multiprocessing.Queue()
        gsr_classifier = \
            ModalityClassification(gsr_train,
                                   gsr_test,
                                   train_labels,
                                   test_labels,
                                   gsr_queue,
                                   type="lstm",
                                   model_name="gsr_lstm.h5")
        
        ppg_queue = multiprocessing.Queue()
        ppg_classifier = \
            ModalityClassification(ppg_train,
                                   ppg_test,
                                   train_labels,
                                   test_labels,
                                   ppg_queue,
                                   type="lstm",
                                   model_name="ppg_lstm.h5")
        
        eeg_classifier.start()
        gsr_classifier.start()
        ppg_classifier.start()
        eeg_accuracy, eeg_fscore, eeg_preds, eeg_probabilities = eeg_queue.get()
        gsr_accuracy, gsr_fscore, gsr_preds, gsr_probabilities = gsr_queue.get()
        ppg_accuracy, ppg_fscore, ppg_preds, ppg_probabilities = ppg_queue.get()

        eeg_classifier.join()
        gsr_classifier.join()
        ppg_classifier.join()

        all_eeg_accuracy.append(eeg_accuracy)
        all_eeg_fscore.append(eeg_fscore) 
        
        all_gsr_accuracy.append(gsr_accuracy)
        all_gsr_fscore.append(gsr_fscore)
        

        all_ppg_accuracy.append(ppg_accuracy)
        all_ppg_fscore.append(ppg_fscore)
 
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

    return eeg_accuracy, gsr_accuracy, ppg_accuracy, fusion_accuracy, \
           eeg_fscore, gsr_fscore, ppg_fscore, fusion_fscore

