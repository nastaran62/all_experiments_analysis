from exp1_1.feature_extraction import partitioning_and_getting_features, partitioning_and_getting_features_liwc


def prepare_data(label_type="arousal", window_size=0, calculate=False):

    input_path = "../experimental_data/exp1_1/preprocessed_data_same_length"
    label_path = "../experimental_data/exp1_1/prepared_labels"
    feature_path = "../experimental_data/exp1_1/features"
    all_eeg, all_gsr, all_ppg, all_emotions, all_arousals, all_valences, all_intensity = \
        partitioning_and_getting_features(input_path, label_path, feature_path, trial_size=60, window_size=window_size, calculate=calculate)


    if label_type == "arousal":
        labels = all_arousals
    elif label_type == "valence":
        labels = all_valences
    else:
        labels = all_emotions

    return all_eeg, all_gsr, all_ppg, labels