from analysis.exp2.eeg import measure_eeg_features
from analysis.exp2.gsr import measure_gsr_features
from analysis.exp2.ppg import measure_ppg_features

measure_eeg_features("/media/nastaran/HDD/projects/f2f_remote/preprocessed_data",
                     "analysis/exp2/extracted_data/normalized")
#measure_gsr_features("/media/nastaran/HDD/projects/f2f_remote/preprocessed_data",
#                     "analysis/exp2/extracted_data/not_normalized")
#measure_ppg_features("/media/nastaran/HDD/projects/f2f_remote/preprocessed_data",
#                     "analysis/exp2/extracted_data/not_normalized_with_nan")