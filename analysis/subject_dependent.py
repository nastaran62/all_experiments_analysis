import numpy as np
import csv
from cross_subject_manual import shuffled_kfold_evaluation, lstm_kfold_evaluation, kfold_evaluation

def subject_dependent_evaluation(all_eeg, all_gsr, all_ppg, all_labels,
                                 participants,
                                 prepare_data,
                                 fold=5,
                                 model_path=".",
                                 shuffle=True,
                                 label_type="arousal"):
    eeg_accuracy_all = []
    eeg_fscore_all = []

    gsr_accuracy_all = []
    gsr_fscore_all = []

    ppg_accuracy_all = []
    ppg_fscore_all = []

    fusion_accuracy_all = []
    fusion_fscore_all = []

    efusion_accuracy_all = []
    efusion_fscore_all = []
    for i in range(len(all_labels)):
        if i in participants:
            print(i, "**************************************")
            eeg = prepare_data(all_eeg[i])
            gsr = prepare_data(all_gsr[i])
            ppg = prepare_data(all_ppg[i])
            labels = prepare_data(all_labels[i], label=True)
            print(eeg.shape, gsr.shape, ppg.shape, labels.shape, "****************")
            if shuffle is True:
                eeg_accuracy, gsr_accuracy, ppg_accuracy, fusion_accuracy, efusion_accuracy, \
                    eeg_fscore, gsr_fscore, ppg_fscore, fusion_fscore, efusion_fscore = \
                        shuffled_kfold_evaluation(eeg, gsr, ppg, labels, k=fold, model_path=model_path, save=False)
            else:
                eeg_accuracy, gsr_accuracy, ppg_accuracy, fusion_accuracy, efusion_accuracy, \
                    eeg_fscore, gsr_fscore, ppg_fscore, fusion_fscore, efusion_fscore = \
                        kfold_evaluation(eeg, gsr, ppg, labels, k=fold, model_path=model_path, save=False)
            print("eeg_accuracy: ", eeg_accuracy, "eeg_fscore: ", eeg_fscore)
            print("gsr_accuracy: ", gsr_accuracy, "gsr_fscore: ", gsr_fscore)
            print("ppg_accuracy: ", ppg_accuracy, "ppg_fscore: ", ppg_fscore)
            print("fusion_accuracy: ", fusion_accuracy, "fusion_fscore: ", fusion_fscore)
            print("efusion_accuracy: ", efusion_accuracy, "efusion_fscore: ", efusion_fscore)
            eeg_accuracy_all.append(eeg_accuracy)
            eeg_fscore_all.append(eeg_fscore)

            gsr_accuracy_all.append(gsr_accuracy)
            gsr_fscore_all.append(gsr_fscore)

            ppg_accuracy_all.append(ppg_accuracy)
            ppg_fscore_all.append(ppg_fscore)

            fusion_accuracy_all.append(fusion_accuracy)
            fusion_fscore_all.append(fusion_fscore)

            efusion_accuracy_all.append(efusion_accuracy)
            efusion_fscore_all.append(efusion_fscore)

            row = [label_type, eeg_accuracy, gsr_accuracy, ppg_accuracy, fusion_accuracy, efusion_accuracy,
                    eeg_fscore, gsr_fscore, ppg_fscore, fusion_fscore, efusion_fscore]
            with(open("result_subject_dependent_for_alll.csv", "a")) as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(row)
                csv_file.flush()

    print("eeg accuracy: ", np.mean(np.array(eeg_accuracy_all)))
    print("eeg fscore: ", np.mean(np.array(eeg_fscore_all)))
    print("gsr accuracy: ", np.mean(np.array(gsr_accuracy_all)))
    print("gsr fscore: ", np.mean(np.array(gsr_fscore_all)))
    print("ppg accuracy: ", np.mean(np.array(ppg_accuracy_all)))
    print("ppg fscore: ", np.mean(np.array(ppg_fscore_all)))
    print("fusion accuracy: ", np.mean(np.array(fusion_accuracy_all)))
    print("fusion fscore: ", np.mean(np.array(fusion_fscore_all)))
    print("equal fusion accuracy: ", np.mean(np.array(efusion_accuracy_all)))
    print("equal fusion fscore: ", np.mean(np.array(efusion_fscore_all)))
    eeg_accuracy = np.mean(np.array(eeg_accuracy_all))
    eeg_fscore = np.mean(np.array(eeg_fscore_all))
    gsr_accuracy = np.mean(np.array(gsr_accuracy_all))
    gsr_fscore = np.mean(np.array(gsr_fscore_all))
    ppg_accuracy = np.mean(np.array(ppg_accuracy_all))
    ppg_fscore = np.mean(np.array(ppg_fscore_all))
    fusion_accuracy = np.mean(np.array(fusion_accuracy_all))
    fusion_fscore = np.mean(np.array(fusion_fscore_all))
    efusion_accuracy = np.mean(np.array(efusion_accuracy_all))
    efusion_fscore = np.mean(np.array(efusion_fscore_all))

    std = np.std(np.array(eeg_accuracy_all))
    eeg_cfi = (1.96 * (std/np.sqrt(len(eeg_accuracy_all))))
    print("eeg", eeg_cfi)

    std = np.std(np.array(eeg_fscore_all))
    eeg_fscore_cfi = (1.96 * (std/np.sqrt(len(eeg_fscore_all))))
    print("eeg", eeg_fscore_cfi)

    std = np.std(np.array(gsr_accuracy_all))
    gsr_cfi = (1.96 * (std/np.sqrt(len(gsr_accuracy_all))))
    print("gsr", gsr_cfi)

    std = np.std(np.array(gsr_fscore_all))
    gsr_fscore_cfi = (1.96 * (std/np.sqrt(len(gsr_fscore_all))))
    print("gsr", gsr_fscore_cfi)

    std = np.std(np.array(ppg_accuracy_all))
    ppg_cfi = (1.96 * (std/np.sqrt(len(ppg_accuracy_all))))
    print("ppg", ppg_cfi)

    std = np.std(np.array(ppg_fscore_all))
    ppg_fscore_cfi = (1.96 * (std/np.sqrt(len(ppg_fscore_all))))
    print("ppg", gsr_fscore_cfi)

    std = np.std(np.array(fusion_accuracy_all))
    fusion_cfi = (1.96 * (std/np.sqrt(len(fusion_accuracy_all))))
    print("fusion", fusion_cfi)

    std = np.std(np.array(fusion_fscore_all))
    fusion_fscore_cfi = (1.96 * (std/np.sqrt(len(fusion_fscore_all))))
    print("fusion", fusion_fscore_cfi)

    std = np.std(np.array(efusion_accuracy_all))
    efusion_cfi = (1.96 * (std/np.sqrt(len(efusion_accuracy_all))))
    print("efusion", efusion_cfi)

    std = np.std(np.array(efusion_fscore_all))
    efusion_fscore_cfi = (1.96 * (std/np.sqrt(len(efusion_fscore_all))))
    print("efusion", efusion_fscore_cfi)

    row = [label_type, shuffle, eeg_accuracy, gsr_accuracy, ppg_accuracy, fusion_accuracy, efusion_accuracy,
            eeg_fscore, gsr_fscore, ppg_fscore, fusion_fscore, efusion_fscore,
            eeg_cfi, gsr_cfi, ppg_cfi, fusion_cfi, efusion_cfi, eeg_fscore_cfi,
            gsr_fscore_cfi, ppg_fscore_cfi, fusion_fscore_cfi, efusion_fscore_cfi]
    with(open("result_subject_dep.csv", "a")) as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(row)


def subject_dependent_lstm_evaluation(all_eeg, all_gsr, all_ppg, all_labels,
                                 participants,
                                 prepare_data,
                                 fold=3,
                                 model_path=".",
                                 label_type="arousal"):
    eeg_accuracy_all = []
    eeg_fscore_all = []

    gsr_accuracy_all = []
    gsr_fscore_all = []

    ppg_accuracy_all = []
    ppg_fscore_all = []

    fusion_accuracy_all = []
    fusion_fscore_all = []

    efusion_accuracy_all = []
    efusion_fscore_all = []
    for i in range(len(all_labels)):
        if i in participants:
            print(i, "**************************************")
            eeg = prepare_data(all_eeg[i])
            gsr = prepare_data(all_gsr[i])
            ppg = prepare_data(all_ppg[i])
            labels = prepare_data(all_labels[i], label=True)
            print(eeg.shape, gsr.shape, ppg.shape, labels.shape)
            eeg_accuracy, gsr_accuracy, ppg_accuracy, fusion_accuracy, efusion_accuracy, \
                eeg_fscore, gsr_fscore, ppg_fscore, fusion_fscore, efusion_fscore = \
                    lstm_kfold_evaluation(eeg, gsr, ppg, labels, k=fold, model_path=model_path)
            print("eeg_accuracy: ", eeg_accuracy, "eeg_fscore: ", eeg_fscore)
            print("gsr_accuracy: ", gsr_accuracy, "gsr_fscore: ", gsr_fscore)
            print("ppg_accuracy: ", ppg_accuracy, "ppg_fscore: ", ppg_fscore)
            print("fusion_accuracy: ", fusion_accuracy, "fusion_fscore: ", fusion_fscore)
            eeg_accuracy_all.append(eeg_accuracy)
            eeg_fscore_all.append(eeg_fscore)

            gsr_accuracy_all.append(gsr_accuracy)
            gsr_fscore_all.append(gsr_fscore)

            ppg_accuracy_all.append(ppg_accuracy)
            ppg_fscore_all.append(ppg_fscore)

            fusion_accuracy_all.append(fusion_accuracy)
            fusion_fscore_all.append(fusion_fscore)

            efusion_accuracy_all.append(efusion_accuracy)
            efusion_fscore_all.append(efusion_fscore)

            row = [label_type, eeg_accuracy, gsr_accuracy, ppg_accuracy, fusion_accuracy, efusion_accuracy,
                    eeg_fscore, gsr_fscore, ppg_fscore, fusion_fscore, efusion_fscore]
            with(open("result_lstm_subject_dep.csv", "a")) as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(row)
                csv_file.flush()

    print("eeg accuracy: ", np.mean(np.array(eeg_accuracy_all)))
    print("eeg fscore: ", np.mean(np.array(eeg_fscore_all)))
    print("gsr accuracy: ", np.mean(np.array(gsr_accuracy_all)))
    print("gsr fscore: ", np.mean(np.array(gsr_fscore_all)))
    print("ppg accuracy: ", np.mean(np.array(ppg_accuracy_all)))
    print("ppg fscore: ", np.mean(np.array(ppg_fscore_all)))
    print("fusion accuracy: ", np.mean(np.array(fusion_accuracy_all)))
    print("fusion fscore: ", np.mean(np.array(fusion_fscore_all)))
    print("equal fusion accuracy: ", np.mean(np.array(efusion_accuracy_all)))
    print("equal fusion fscore: ", np.mean(np.array(efusion_fscore_all)))
