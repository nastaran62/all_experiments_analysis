import csv
import os
from operator import itemgetter

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

                row = [int(row[0]), emotion, int(row[2]), valence, arousal, dominance]
                csv_writer.writerow(row)

prepare_exp1_0_labels("exp1_0/raw_data",
                "exp1_0/prepared_labels")

#prepare_exp1_1_labels("exp1_1/raw_labels/AfterConversation-V1.csv",
#                  "exp1_1/prepared_labels")

