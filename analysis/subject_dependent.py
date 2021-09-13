import numpy as np
from cross_subject_manual import kfold_evaluation, lstm_kfold_evaluation

def subject_dependent_evaluation(all_eeg, all_gsr, all_ppg, all_labels, 
                                 participants,
                                 prepare_data,
                                 fold=3,
                                 model_path="."):
    eeg_accuracy_all = []
    eeg_fscore_all = []

    gsr_accuracy_all = []
    gsr_fscore_all = []

    ppg_accuracy_all = []
    ppg_fscore_all = []

    fusion_accuracy_all = []
    fusion_fscore_all = []
    for i in range(len(all_labels)):
        if i in participants:
            print(i, "**************************************")
            eeg = prepare_data(all_eeg[i])
            gsr = prepare_data(all_gsr[i])
            ppg = prepare_data(all_ppg[i])
            labels = prepare_data(all_labels[i], label=True)
            eeg_accuracy, gsr_accuracy, ppg_accuracy, fusion_accuracy, \
                eeg_fscore, gsr_fscore, ppg_fscore, fusion_fscore = \
                    kfold_evaluation(eeg, gsr, ppg, labels, k=fold, model_path=model_path)
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

    print("eeg accuracy: ", np.mean(np.array(eeg_accuracy_all)))
    print("eeg fscore: ", np.mean(np.array(eeg_fscore_all)))
    print("gsr accuracy: ", np.mean(np.array(gsr_accuracy_all)))
    print("gsr fscore: ", np.mean(np.array(gsr_fscore_all)))
    print("ppg accuracy: ", np.mean(np.array(ppg_accuracy_all)))
    print("ppg fscore: ", np.mean(np.array(ppg_fscore_all)))
    print("fusion accuracy: ", np.mean(np.array(fusion_accuracy_all)))
    print("fusion fscore: ", np.mean(np.array(fusion_fscore_all)))

def subject_dependent_lstm_evaluation(all_eeg, all_gsr, all_ppg, all_labels, 
                                 participants,
                                 prepare_data,
                                 fold=3,
                                 model_path="."):
    eeg_accuracy_all = []
    eeg_fscore_all = []

    gsr_accuracy_all = []
    gsr_fscore_all = []

    ppg_accuracy_all = []
    ppg_fscore_all = []

    fusion_accuracy_all = []
    fusion_fscore_all = []
    for i in range(len(all_labels)):
        if i in participants:
            print(i, "**************************************")
            eeg = prepare_data(all_eeg[i])
            gsr = prepare_data(all_gsr[i])
            ppg = prepare_data(all_ppg[i])
            labels = prepare_data(all_labels[i], label=True)
            print(eeg.shape, gsr.shape, ppg.shape, labels.shape)
            eeg_accuracy, gsr_accuracy, ppg_accuracy, fusion_accuracy, \
                eeg_fscore, gsr_fscore, ppg_fscore, fusion_fscore = \
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

    print("eeg accuracy: ", np.mean(np.array(eeg_accuracy_all)))
    print("eeg fscore: ", np.mean(np.array(eeg_fscore_all)))
    print("gsr accuracy: ", np.mean(np.array(gsr_accuracy_all)))
    print("gsr fscore: ", np.mean(np.array(gsr_fscore_all)))
    print("ppg accuracy: ", np.mean(np.array(ppg_accuracy_all)))
    print("ppg fscore: ", np.mean(np.array(ppg_fscore_all)))
    print("fusion accuracy: ", np.mean(np.array(fusion_accuracy_all)))
    print("fusion fscore: ", np.mean(np.array(fusion_fscore_all)))