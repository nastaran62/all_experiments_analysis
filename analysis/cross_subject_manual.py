import numpy as np
import os
import pathlib
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from classification import multimodal_classification, voting_fusion, equal_fusion
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler, MinMaxScaler


def shuffled_kfold_evaluation(eeg, gsr, ppg, labels, k=5, model_path="."):
    eeg = eeg.reshape(-1, *eeg.shape[-1:])
    gsr = gsr.reshape(-1, *gsr.shape[-1:])
    ppg = ppg.reshape(-1, *ppg.shape[-1:])
    labels = labels.reshape(-1)
    if not os.path.exists(model_path):
        pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
    print(eeg.shape, gsr.shape, ppg.shape, labels.shape)
    # Shuffling
    permutation = np.random.permutation(labels.shape[0])
    eeg = eeg[permutation, :]
    gsr = gsr[permutation, :]
    ppg = ppg[permutation, :]
    labels = labels[permutation]

    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=100)
    all_eeg_accuracy = []
    all_gsr_accuracy = []
    all_ppg_accuracy = []
    all_fusion_accuracy = []
    all_efusion_accuracy = []
    all_eeg_fscore = []
    all_gsr_fscore = []
    all_ppg_fscore = []
    all_fusion_fscore = []
    all_efusion_fscore = []
    for train_index, test_index in kf.split(labels, y=labels):
        eeg_train, eeg_test = eeg[train_index, :], eeg[test_index, :]
        gsr_train, gsr_test = gsr[train_index, :], gsr[test_index, :]
        ppg_train, ppg_test = ppg[train_index, :], ppg[test_index, :]
        train_labels, test_labels = labels[train_index], labels[test_index]
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

        efusion_accuracy, efusion_fscore = \
            equal_fusion(eeg_probabilities, gsr_probabilities,
                          ppg_probabilities, test_labels)
        all_efusion_accuracy.append(fusion_accuracy)
        all_efusion_fscore.append(efusion_fscore)

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
    efusion_accuracy = np.mean(np.array(all_efusion_accuracy))
    efusion_fscore = np.mean(np.array(all_efusion_fscore))

    return eeg_accuracy, gsr_accuracy, ppg_accuracy, fusion_accuracy, efusion_accuracy, \
           eeg_fscore, gsr_fscore, ppg_fscore, fusion_fscore, efusion_fscore


def lstm_kfold_evaluation(eeg, gsr, ppg, labels, k=5, model_path="."):
    if not os.path.exists(model_path):
        pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
    # Shuffling
    permutation = np.random.permutation(labels.shape[0])
    eeg = eeg[permutation, :, :]
    gsr = gsr[permutation, :, :]
    ppg = ppg[permutation, :, :]
    labels = labels[permutation]

    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    all_eeg_accuracy = []
    all_gsr_accuracy = []
    all_ppg_accuracy = []
    all_fusion_accuracy = []
    all_efusion_accuracy = []
    all_eeg_fscore = []
    all_gsr_fscore = []
    all_ppg_fscore = []
    all_fusion_fscore = []
    all_fusion_fscore = []
    all_efusion_fscore = []
    for train_index, test_index in kf.split(labels, y=labels):
        eeg_train, eeg_test = eeg[train_index, :, :], eeg[test_index, :, :]
        gsr_train, gsr_test = gsr[train_index, :, :], gsr[test_index, :, :]
        ppg_train, ppg_test = ppg[train_index, :, :], ppg[test_index, :, :]
        train_labels, test_labels = labels[train_index], labels[test_index]

        eeg_parameters = eeg_train, eeg_test, "lstm", os.path.join(model_path, "eeg_lstm.h5")
        gsr_parameters = gsr_train, gsr_test, "lstm", os.path.join(model_path, "gsr_lstm.h5")
        ppg_parameters = ppg_train, ppg_test, "lstm", os.path.join(model_path, "ppg_lstm.h5")

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

        efusion_accuracy, efusion_fscore = \
            equal_fusion(eeg_probabilities, gsr_probabilities,
                          ppg_probabilities, test_labels)
        all_efusion_accuracy.append(fusion_accuracy)
        all_efusion_fscore.append(efusion_fscore)

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
    efusion_accuracy = np.mean(np.array(all_efusion_accuracy))
    efusion_fscore = np.mean(np.array(all_efusion_fscore))

    return eeg_accuracy, gsr_accuracy, ppg_accuracy, fusion_accuracy, efusion_accuracy, \
           eeg_fscore, gsr_fscore, ppg_fscore, fusion_fscore, efusion_fscore


def mixed_kfold_evaluation(eeg, gsr, ppg, labels, k=5, model_path="."):
    '''
    It is for using lstm or traditional classifiers for different modalities and fusng the result
    '''
    if not os.path.exists(model_path):
        pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
    # Shuffling
    permutation = np.random.permutation(labels.shape[0])
    eeg = eeg[permutation, :, :]
    gsr = gsr[permutation, :, :]
    ppg = ppg[permutation, :, :]
    labels = labels[permutation]
    print(eeg.shape)

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
        eeg_train, eeg_test = eeg[train_index, :, :], eeg[test_index, :, :]
        gsr_train, gsr_test = gsr[train_index, :, :], gsr[test_index, :, :]
        ppg_train, ppg_test = ppg[train_index, :, :], ppg[test_index, :, :]
        train_labels, test_labels = labels[train_index], labels[test_index]

        #eeg_parameters = eeg_train, eeg_test, "lstm", os.path.join(model_path, "eeg_lstm.h5")
        eeg_parameters = eeg_train, eeg_test, "mixed_random_forest", os.path.join(model_path, "eeg_rf.pickle")
        #gsr_parameters = gsr_train, gsr_test, "lstm", os.path.join(model_path, "gsr_lstm.h5")
        gsr_parameters = gsr_train, gsr_test, "mixed_random_forest", os.path.join(model_path, "gsr_rf.pickle")
        #ppg_parameters = ppg_train, ppg_test, "lstm", os.path.join(model_path, "ppg_lstm.h5")
        ppg_parameters = ppg_train, ppg_test, "mixed_random_forest", os.path.join(model_path, "ppg_rf.pickle")

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


def kfold_evaluation(eeg, gsr, ppg, labels, k=3, model_path="."):
    '''
    It is for using lstm or traditional classifiers for different modalities and fusng the result
    '''
    if not os.path.exists(model_path):
        pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
    # Shape is (participant*trials), windows, features

    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=100)
    all_eeg_accuracy = []
    all_gsr_accuracy = []
    all_ppg_accuracy = []
    all_fusion_accuracy = []
    all_efusion_accuracy = []
    all_eeg_fscore = []
    all_gsr_fscore = []
    all_ppg_fscore = []
    all_fusion_fscore = []
    all_efusion_fscore = []

    for train_index, test_index in kf.split(labels[:, 0], y=labels[:, 0]):
        eeg_train, eeg_test = eeg[train_index, :, :], eeg[test_index, :, :]
        gsr_train, gsr_test = gsr[train_index, :, :], gsr[test_index, :, :]
        ppg_train, ppg_test = ppg[train_index, :, :], ppg[test_index, :, :]
        train_labels, test_labels = labels[train_index, :], labels[test_index, :]
        print(eeg_train.shape, train_labels.shape)
        eeg_train = eeg_train.reshape(-1, eeg_train.shape[-1])
        eeg_test = eeg_test.reshape(-1, eeg_test.shape[-1])
        gsr_train = gsr_train.reshape(-1, gsr_train.shape[-1])
        gsr_test = gsr_test.reshape(-1, gsr_test.shape[-1])
        ppg_train = ppg_train.reshape(-1, ppg_train.shape[-1])
        ppg_test = ppg_test.reshape(-1, ppg_test.shape[-1])
        train_labels = train_labels.reshape(-1)
        test_labels = test_labels.reshape(-1)
        print(eeg_train.shape, train_labels.shape)

        eeg_parameters = eeg_train, eeg_test, "random_forest", os.path.join(model_path, "eeg_rf.pickle")
        gsr_parameters = gsr_train, gsr_test, "random_forest", os.path.join(model_path, "gsr_rf.pickle")
        ppg_parameters = ppg_train, ppg_test, "random_forest", os.path.join(model_path, "ppg_rf.pickle")

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


        efusion_accuracy, efusion_fscore = \
            equal_fusion(eeg_probabilities, gsr_probabilities,
                          ppg_probabilities, test_labels)
        all_efusion_accuracy.append(fusion_accuracy)
        all_efusion_fscore.append(efusion_fscore)

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
    efusion_accuracy = np.mean(np.array(all_efusion_accuracy))
    efusion_fscore = np.mean(np.array(all_efusion_fscore))

    return eeg_accuracy, gsr_accuracy, ppg_accuracy, fusion_accuracy, efusion_accuracy, \
           eeg_fscore, gsr_fscore, ppg_fscore, fusion_fscore, efusion_fscore
