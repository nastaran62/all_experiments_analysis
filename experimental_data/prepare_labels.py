import csv
import os
from operator import itemgetter

emotions = {"Anger":[0, "4", "2"],
             "Fear":[1, "4", "2"],
             "Happiness":[2, "4", "5"],
             "Neutral":[3, "3", "3"],
             "Sadness":[4, "2", "2"],
             "other":[5, "3", "3"]}
def prepare_exp1_1_labels(label_path, output_path):
    csv_file = open(label_path, 'r')
    csv_reader = csv.reader(csv_file)
    start = False
    for line in csv_reader:
        if start is False:
            start = True
            continue
        with open("{0}/p{1}.csv".format(output_path, line[1]), "w") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["stimuli",
                                 "emotion-1", "intensity-1", "valence-1","arousal-1",
                                 "emotion-2", "intensity-2", "valence-2","arousal-2",
                                 "emotion-3", "intensity-3", "valence-3","arousal-3"])
            stimulus = 0
            stimuli = [1, 3, 4, 5, 6]
            count = 12
            start = 2
            while stimulus < 5:
                row = [stimuli[stimulus]] + \
                      line[count * stimulus + start: count * (stimulus + 1) + start]
                csv_writer.writerow(row)
                stimulus += 1


def prepare_exp1_0_labels(label_path, output_path):
    participants = os.listdir(label_path)
    participants.sort()
    for participant in participants:
        participant_label_path = "{0}/{1}/self_report".format(label_path, participant)
        self_report_file = os.listdir(participant_label_path)[0]
        csv_file = open(os.path.join(participant_label_path, self_report_file), 'r')
        csv_reader = csv.reader(csv_file)
        start = False
        all_rows = []
        first_row = False
        for line in csv_reader:
            if first_row is False:
                first_row = True
                continue
            all_rows.append(line)
        all_rows.sort(key=itemgetter(0))
        stimuli = {'1': 0,
                   '3': 1,
                   '4': 2,
                   '5': 3,
                   '6': 4}
        with open("{0}/{1}.csv".format(output_path, participant), "w") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["stimulus",
                                "emotion", "intensity", "valence","arousal",
                                "dominance",])
            for row in all_rows:
                valence = int(row[4])
                arousal = int(row[5])
                dominance = int(row[6])
                print(row[1])
                emotion = stimuli[row[1]]
                '''
                if valence >= 5:
                    valence = 1
                else:
                    valence = 0

                if arousal > 5:
                    arousal = 1
                else:
                    arousal = 0

                if dominance > 5:
                    dominance = 1
                else:
                    dominance = 0
                '''
                row = [int(row[0]), emotion, int(row[2]), valence, arousal, dominance]
                csv_writer.writerow(row)


def prepare_high_intensity(label_path, output_path):
    all_p = os.listdir(label_path)
    all_p.sort()
    with(open(output_path, "w")) as output_file:
        csv_writer = csv.writer(output_file)
        for file in all_p:
            file_path = os.path.join(label_path, file)
            with(open(file_path, "r")) as csv_file:
                csv_reader = csv.reader(csv_file)
                i = 0
                for line in csv_reader:
                    if i == 0:
                        i += 1
                        continue

                    name = line[0]
                    first_intensity = line[2]
                    second_intensity = line[6]
                    third_intensity = line[10]
                    if second_intensity >= max(first_intensity, third_intensity):
                        emotion = line[5]
                        valence = line[7]
                        arousal = line[8]
                        intensity = second_intensity
                    elif first_intensity > max(second_intensity, third_intensity):
                        emotion = line[1]
                        valence = line[3]
                        arousal = line[4]
                        intensity = first_intensity
                    else:
                        emotion = line[9]
                        valence = line[11]
                        arousal = line[12]
                        intensity = third_intensity
                    i += 1
                    if valence == "":
                        valence = emotions[emotion][2]
                    if arousal == "":
                        arousal = emotions[emotion][1]
                    row = [file, name, emotion, intensity, arousal, valence]
                    csv_writer.writerow(row)
                    output_file.flush()

prepare_high_intensity("exp1_1/prepared_labels", "all_labels.csv")


#prepare_exp1_0_labels("exp1_0/raw_data",
#                "exp1_0/prepared_labels")


#prepare_exp1_1_labels("exp1_1/raw_labels/AfterConversation-V1.csv",
#                  "exp1_1/prepared_labels")
