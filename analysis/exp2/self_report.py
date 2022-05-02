import pandas as pd
import scipy.stats as stat
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def prepare_data_frame():
    all_f2f = pd.read_csv("../../experimental_data/exp2-0/all_f2f.csv")
    all_remote = pd.read_csv("../../experimental_data/exp2-0/all_remote.csv")

    #all_eeg = pd.read_csv("extracted_data/eeg_features.csv")
    #all_ppg = pd.read_csv("extracted_data/ppg_features_conv.csv")
    all_gsr = pd.read_csv("extracted_data/gsr_features_conv.csv")

    all_data = pd.concat([all_f2f, all_remote], ignore_index=True, keys=all_f2f.index)

    joined_gsr = all_data.set_index(["participant", "stimuli", "task"]).join(all_gsr.set_index(["participant", "stimuli", "task"]))

    print(joined_gsr.groupby(["task", "self_label"]).mean())

prepare_data_frame()

def category_label_analysis_within_subject():
    all_f2f = pd.read_csv("../../experimental_data/exp2-0/all_f2f.csv")
    all_remote = pd.read_csv("../../experimental_data/exp2-0/all_remote.csv")
    all_f2f_summary = all_f2f.groupby(["participant", "label"]).mean()
    all_f2f_summary.reset_index()
    all_f2f_summary.sort_values("participant")
    all_remote_summary = all_remote.groupby(["participant", "label"]).mean()
    all_remote_summary.reset_index()
    all_remote_summary.sort_values("participant")

    result = (all_f2f_summary.compare(all_remote_summary, keep_shape=True, keep_equal=True))
    result.to_csv("compare.csv")
    


def category_label_analysis_within_all():
    all_f2f = pd.read_csv("../../experimental_data/exp2-0/all_f2f.csv")
    all_remote = pd.read_csv("../../experimental_data/exp2-0/all_remote.csv")

    all_data = pd.concat([all_f2f, all_remote], ignore_index=True, keys=all_f2f.index)
    output_4class = all_data.groupby(["task", "label", "participant"]).mean()
    print(output_4class)
    output_valence = all_data.groupby(["task", "label_v"]).mean()
    output_arousal = all_data.groupby(["task", "label_a"]).mean()

    # y can be arousal, valence, dominance, emotion, intensity
    g = sns.catplot(data=all_data, kind="bar",
                    x="label", y="arousal", hue="task",
                    ci="sd", palette="dark", alpha=.6, height=6)
    g.despine(left=True)
    g.set_axis_labels("Label", "SAM rating")
    g.legend.set_title("Arousal")
    
    g = sns.catplot(data=all_data, kind="bar",
                    x="label", y="valence", hue="task",
                    ci="sd", palette="dark", alpha=.6, height=6)
    g.despine(left=True)
    g.set_axis_labels("Label", "SAM rating")
    g.legend.set_title("Valence")

    g = sns.catplot(data=all_data, kind="bar",
                    x="label", y="dominance", hue="task",
                    ci="sd", palette="dark", alpha=.6, height=6)
    g.despine(left=True)
    g.set_axis_labels("Label", "SAM rating")
    g.legend.set_title("Dominance")

    g = sns.catplot(data=all_data, kind="bar",
                    x="label", y="emotion", hue="task",
                    ci="sd", palette="dark", alpha=.6, height=6)
    g.despine(left=True)
    g.set_axis_labels("Label", "SAM rating")
    g.legend.set_title("Emotion")

    g = sns.catplot(data=all_data, kind="bar",
                    x="label", y="intensity", hue="task",
                    ci="sd", palette="dark", alpha=.6, height=6)
    g.despine(left=True)
    g.set_axis_labels("Label", "SAM rating")
    g.legend.set_title("Intensity")

    plt.show()

category_label_analysis_within_all()