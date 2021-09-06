import csv

def prepare_labels_v2(label_path, output_path):
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

prepare_labels_v2("exp1_1/raw_labels/AfterConversation-V1.csv",
                  "exp1_1/prepared_labels")