import os
import pandas as pd
import numpy as np
import csv

F2F_PATH = "/home/zsaf419/Documents/project/all_experiments_analysis/analysis/exp2/face/f2f"
REMOTE_PATH = "/home/zsaf419/Documents/project/all_experiments_analysis/analysis/exp2/face/remote"
F2F_WITH_EXPRESSION = "f2f_with_expressions"
REMOTE_WITH_EXPRESSIONS = "remote_with_expressions"
EXPRESSIONS = ["happiness", "sadness", "surprise", "fear",
               "anger", "disgust", "neutral"]


FEATURES = ["AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
            "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
            "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU28_r", "AU45_r"
            "AU01_c", "AU02_c", "AU04_c", "AU05_c", "AU06_c", "AU07_c",
            "AU09_c", "AU10_c", "AU12_c", "AU14_c", "AU15_c", "AU17_c",
            "AU20_c", "AU23_c", "AU25_c", "AU26_c", "AU28_c", "AU45_c"]

def extract_emotions(in_folder_path, out_folder_path):
    if not os.path.exists(out_folder_path):
        os.mkdir(out_folder_path)

    all_files = os.listdir(in_folder_path)
    all_files.sort()
    for file in all_files:
        df = pd.read_csv(os.path.join(in_folder_path, file))
        df.reset_index()
        header = df.columns.values
        rows = []
        for index, row in df.iterrows():
            happiness = 0
            sadness = 0
            surprise = 0
            fear = 0
            anger = 0
            disgust = 0
            neutral = 0
            new_row = list(row.values)

            # happiness: 6+12
            if row[" AU06_c"] + row[" AU12_c"] == 2:
                happiness = 1


            # sadness: 1+4+15
            if row[" AU01_c"] + row[" AU04_c"] + row[" AU15_c"] == 3:
                sadness = 1

            # surprise: 1+2+5B+26
            if ((row[" AU01_c"] + row[" AU02_c"] + row[" AU05_c"] + row[" AU26_c"]) == 4
                    and row[" AU05_r"] > 0.2 and row[" AU05_r"] <= 0.4):
                surprise = 1

            # fear: 1+2+4+5+7+20+26
            if (row[" AU01_c"] + row[" AU02_c"] + row[" AU04_c"] + row[" AU05_c"] +
                    row[" AU07_c"] + row[" AU20_c"] + row[" AU26_c"] == 7):
                fear = 1

            # anger: 4+5+7+23
            if (row[" AU04_c"] + row[" AU05_c"] + row[" AU07_c"] + row[" AU23_c"]) == 4:
                anger = 1

            # disgust: 9+15+17
            if row[" AU09_c"] + row[" AU15_c"] + row[" AU17_c"] == 3:
                disgust = 1

            if row[" AU01_c"] + row[" AU02_c"] + row[" AU04_c"] + row[" AU05_c"] + \
               row[" AU06_c"] + row[" AU07_c"] + row[" AU09_c"] + row[" AU10_c"] + \
               row[" AU12_c"] + row[" AU14_c"] + row[" AU15_c"] + row[" AU17_c"] + \
               row[" AU20_c"] + row[" AU23_c"] + row[" AU25_c"] + row[" AU26_c"] + \
               row[" AU28_c"] + row[" AU45_c"] == 0:
                neutral = 1
            emotions = \
                [happiness, sadness, surprise, fear, anger, disgust, neutral]

            new_row.extend(emotions)
            rows.append(new_row)
        expressions = ["happiness", "sadness", "surprise", "fear",
                       "anger", "disgust", "neutral"]
        new_header = list(header) + EXPRESSIONS
        df = pd.DataFrame(np.array(rows), columns=new_header)
        print(file)
        df.to_csv(os.path.join(out_folder_path, file))


def extract_features(file_path, file_writer, task):
    happiness = 0
    sadness = 0
    surprise = 0
    fear = 0
    anger = 0
    disgust = 0
    neutral = 0
    all_files = os.listdir(file_path)
    all_files.sort()
    for file in all_files:
        participant_number = int(file[7:9])
        print(participant_number, "************************* ppp")
        stimuli = int(file[13:15])
        print(stimuli, "************************* sss")
        df = pd.read_csv(os.path.join(file_path, file))
        sum_of_frames = df.sum()
        row = [participant_number, stimuli, task]
        for item in sum_of_frames[6:]:
            row.extend([item])
        file_writer.writerow(row)


def main():
    #extract_emotions(F2F_PATH, F2F_WITH_EXPRESSION)
    #extract_emotions(REMOTE_PATH, REMOTE_WITH_EXPRESSIONS)
    with open("face_features_sum.csv", "a") as face_features_file:
        csv_writer = csv.writer(face_features_file)
        header = ["participant", "stimuli", "task"] + FEATURES + EXPRESSIONS
        csv_writer.writerow(header)
        extract_features(F2F_WITH_EXPRESSION, csv_writer, 0)
        extract_features(REMOTE_WITH_EXPRESSIONS, csv_writer, 1)
        face_features_file.flush()


main()
#extract_emotions(F2F_PATH, ".")
