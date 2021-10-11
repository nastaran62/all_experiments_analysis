import numpy as np
import os
import csv
import pathlib
from classification import multimodal_classification, voting_fusion, equal_fusion
from cross_subject_manual import weighted_fusion

def subject_independent_cross_validation(all_eeg, all_gsr, all_ppg, all_labels,
                                         participants,
                                         make_train_test_set,
                                         fold=4,
                                         model_path=".",
                                         label_type="arousal"):
    if not os.path.exists(model_path):
        pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
    fold = fold
    start = 0
    end = start
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
    print(len(participants))
    while start < len(participants):
        end = end + fold
        if end > len(participants):
            end = len(participants)
        test_participants = participants[start:end]
        train_participants = []
        for item in participants:
            if item not in test_participants:
                train_participants.append(item)

        print(test_participants, train_participants,"*******************")
        eeg_train, eeg_test = \
            make_train_test_set(all_eeg, train_participants, test_participants)
        print(eeg_train.shape)

        gsr_train, gsr_test = \
            make_train_test_set(all_gsr, train_participants, test_participants)
        print(gsr_train.shape)

        ppg_train, ppg_test = \
            make_train_test_set(all_ppg, train_participants, test_participants)


        train_labels, test_labels = \
            make_train_test_set(all_labels, train_participants, test_participants)

        print(gsr_train.shape, eeg_train.shape, ppg_train.shape, train_labels.shape)
        print(gsr_test.shape, eeg_test.shape, ppg_test.shape, test_labels.shape)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$4")
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
        eeg_accuracy, eeg_fscore, eeg_preds, eeg_probabilities, eeg_model = eeg_result
        gsr_accuracy, gsr_fscore, gsr_preds, gsr_probabilities, gsr_model = gsr_result
        ppg_accuracy, ppg_fscore, ppg_preds, ppg_probabilities, ppg_model = ppg_result

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

        eeg_weight, gsr_weight, ppg_weight = \
            weighted_fusion(eeg_model, gsr_model, ppg_model, eeg_train, gsr_train, ppg_train,train_labels)
        efusion_accuracy, efusion_fscore = \
            equal_fusion(eeg_probabilities, gsr_probabilities,
                          ppg_probabilities, test_labels,
                          eeg_weight=eeg_weight,
                          gsr_weight=gsr_weight,
                          ppg_weight=ppg_weight)

        all_efusion_accuracy.append(efusion_accuracy)
        all_efusion_fscore.append(efusion_fscore)
        start = end
        row = [label_type, eeg_accuracy, gsr_accuracy, ppg_accuracy, fusion_accuracy, efusion_accuracy,
                eeg_fscore, gsr_fscore, ppg_fscore, fusion_fscore, efusion_fscore]
        with(open("result_subject_independent_for_alll.csv", "a")) as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(row)
            csv_file.flush()

    eeg_accuracy = np.mean(np.array(all_eeg_accuracy))
    eeg_fscore = np.mean(np.array(all_eeg_fscore))
    gsr_accuracy = np.mean(np.array(all_gsr_accuracy))
    gsr_fscore = np.mean(np.array(all_gsr_fscore))
    ppg_accuracy = np.mean(np.array(all_ppg_accuracy))
    ppg_fscore = np.mean(np.array(all_ppg_fscore))
    fusion_accuracy = np.mean(np.array(all_fusion_accuracy))
    fusion_fscore = np.mean(np.array(all_fusion_fscore))

    std = np.std(np.array(all_eeg_accuracy))
    max = np.max(np.array(all_eeg_accuracy))
    min = np.min(np.array(all_eeg_accuracy))
    #eeg_cfi = (1.96 * (std/np.sqrt(len(all_eeg_accuracy))))
    eeg_stat = (max, min, std)
    print("eeg", eeg_stat)

    std = np.std(np.array(all_eeg_fscore))
    max = np.max(np.array(all_eeg_fscore))
    min = np.min(np.array(all_eeg_fscore))
    eeg_fscore_cfi = (1.96 * (std/np.sqrt(len(all_eeg_fscore))))
    eeg_fscore_stat = (max, min, std)
    print("eeg", eeg_fscore_stat)

    std = np.std(np.array(all_gsr_accuracy))
    max = np.max(np.array(all_gsr_accuracy))
    min = np.min(np.array(all_gsr_accuracy))
    gsr_cfi = (1.96 * (std/np.sqrt(len(all_gsr_accuracy))))
    gsr_stat = (max, min, std)
    print("gsr", gsr_cfi)

    std = np.std(np.array(all_gsr_fscore))
    max = np.max(np.array(all_gsr_fscore))
    min = np.min(np.array(all_gsr_fscore))
    gsr_fscore_stat = (max, min, std)
    gsr_cfi = (1.96 * (std/np.sqrt(len(all_gsr_accuracy))))
    gsr_fscore_cfi = (1.96 * (std/np.sqrt(len(all_gsr_fscore))))
    print("gsr", gsr_fscore_cfi)

    std = np.std(np.array(all_ppg_accuracy))
    max = np.max(np.array(all_ppg_accuracy))
    min = np.min(np.array(all_ppg_accuracy))
    ppg_stat = (max, min, std)
    ppg_cfi = (1.96 * (std/np.sqrt(len(all_ppg_accuracy))))
    print("ppg", ppg_cfi)

    std = np.std(np.array(all_ppg_fscore))
    max = np.max(np.array(all_ppg_fscore))
    min = np.min(np.array(all_ppg_fscore))
    ppg_fscore_stat = (max, min, std)
    ppg_fscore_cfi = (1.96 * (std/np.sqrt(len(all_ppg_fscore))))
    print("ppg", gsr_fscore_cfi)

    std = np.std(np.array(all_fusion_accuracy))
    max = np.max(np.array(all_fusion_accuracy))
    min = np.min(np.array(all_fusion_accuracy))
    fusion_stat = (max, min, std)
    fusion_cfi = (1.96 * (std/np.sqrt(len(all_fusion_accuracy))))
    print("fusion", fusion_cfi)

    std = np.std(np.array(all_fusion_fscore))
    max = np.max(np.array(all_fusion_fscore))
    min = np.min(np.array(all_fusion_fscore))
    fusion_fscore_stat = (max, min, std)
    fusion_fscore_cfi = (1.96 * (std/np.sqrt(len(all_fusion_fscore))))
    print("fusion", fusion_fscore_cfi)

    std = np.std(np.array(all_efusion_accuracy))
    max = np.max(np.array(all_efusion_accuracy))
    min = np.min(np.array(all_efusion_accuracy))
    efusion_stat = (max, min, std)
    efusion_cfi = (1.96 * (std/np.sqrt(len(all_efusion_accuracy))))
    print("efusion", efusion_cfi)

    std = np.std(np.array(all_efusion_fscore))
    max = np.max(np.array(all_efusion_fscore))
    min = np.min(np.array(all_efusion_fscore))
    efusion_fscore_stat = (max, min, std)
    efusion_fscore_cfi = (1.96 * (std/np.sqrt(len(all_efusion_fscore))))
    print("efusion", efusion_fscore_cfi)

    row = [label_type, eeg_accuracy, gsr_accuracy, ppg_accuracy, fusion_accuracy, efusion_accuracy,
            eeg_fscore, gsr_fscore, ppg_fscore, fusion_fscore, efusion_fscore,
            eeg_stat, gsr_stat, ppg_stat, fusion_stat, efusion_stat, eeg_fscore_stat,
            gsr_fscore_stat, ppg_fscore_stat, fusion_fscore_stat, efusion_fscore_stat]
    with(open("result_subject_independent.csv", "a")) as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(row)

    print("eeg_accuracy: ", eeg_accuracy, "eeg_fscore: ", eeg_fscore)
    print("gsr_accuracy: ", gsr_accuracy, "gsr_fscore: ", gsr_fscore)
    print("ppg_accuracy: ", ppg_accuracy, "ppg_fscore: ", ppg_fscore)
    print("fusion_accuracy: ", fusion_accuracy, "fusion_fscore: ", fusion_fscore)

def subject_independent_lstm_cross_validation(all_eeg, all_gsr, all_ppg, all_labels,
                                         participants,
                                         make_train_test_set,
                                         fold=4,
                                         model_path=".",
                                         label_type="arousal"):
    if not os.path.exists(model_path):
        pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
    fold = fold
    start = 0
    end = start
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

        eeg_ppg_train = np.concatenate((eeg_train, ppg_train), axis=2)
        eeg_ppg_test = np.concatenate((eeg_test, ppg_test), axis=2)
        eeg_ppg_parameters = \
            eeg_ppg_train, eeg_ppg_test, "lstm", os.path.join(model_path, "eeg_ppg_lstm.h5")


        fused_features_train = np.concatenate((eeg_train, ppg_train), axis=2)
        fused_features_test = np.concatenate((eeg_test, ppg_test), axis=2)
        fused_parameters = \
            fused_features_train, fused_features_test, "lstm", os.path.join(model_path, "fused_lstm.h5")

        eeg_result, gsr_result, ppg_result = \
            multimodal_classification(train_labels,
                                      test_labels,
                                      eeg=eeg_parameters,
                                      gsr=eeg_ppg_parameters,
                                      ppg=ppg_parameters)
        eeg_accuracy, eeg_fscore, eeg_preds, eeg_probabilities, eeg_model = eeg_result
        gsr_accuracy, gsr_fscore, gsr_preds, gsr_probabilities, gsr_model = gsr_result
        ppg_accuracy, ppg_fscore, ppg_preds, ppg_probabilities, ppg_model = ppg_result

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

        eeg_weight, gsr_weight, ppg_weight = \
            weighted_fusion(eeg_model, gsr_model, ppg_model, eeg_train, gsr_train, ppg_train,train_labels)
        efusion_accuracy, efusion_fscore = \
            equal_fusion(eeg_probabilities, gsr_probabilities,
                          ppg_probabilities, test_labels,
                          eeg_weight=eeg_weight,
                          gsr_weight=gsr_weight,
                          ppg_weight=ppg_weight)
        start = end
        row = [label_type, eeg_accuracy, gsr_accuracy, ppg_accuracy, fusion_accuracy, efusion_accuracy,
                eeg_fscore, gsr_fscore, ppg_fscore, fusion_fscore, efusion_fscore]
        with(open("result_subject_independent_lstm.csv", "a")) as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(row)
            csv_file.flush()

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


    print("eeg_accuracy: ", eeg_accuracy, "eeg_fscore: ", eeg_fscore)
    print("gsr_accuracy: ", gsr_accuracy, "gsr_fscore: ", gsr_fscore)
    print("ppg_accuracy: ", ppg_accuracy, "ppg_fscore: ", ppg_fscore)
    print("fusion_accuracy: ", fusion_accuracy, "fusion_fscore: ", fusion_fscore)
    print("efusion_accuracy: ", efusion_accuracy, "efusion_fscore: ", efusion_fscore)
