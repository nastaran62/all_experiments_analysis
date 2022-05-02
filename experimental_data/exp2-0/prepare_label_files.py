import os
import csv

EMOTIONS = {"Anger": 1,
            "Disgust": 2,
            "Fear": 3,
            "Happiness": 4,
            "Neutral": 5,
            "Sadness": 6,
            "Surprise": 7,
            "Other": 8}

# 0: HVHA,   1: HVLA,   2: LVHA,   3: LVLA
LABEL = {"11": 0, "12": 0, "13": 0, "14": 0,
          "20": 1, "22": 1, "23": 1, "24": 1,
          "31": 2, "33": 2, "34": 2, "35": 2,
          "40": 3, "41": 3, "42": 3, "44": 3}

IM_CONV = {"Image < Conversation": 1,
           "Image = Conversation": 2,
           "Image > Conversation": 3}
IM_CONV_V2 = {"ImagelessthanConversation": 1,
              "Thesame": 2,
              "ImagemorethanConversation": 3}

def __get_stimuli_list(stimuli_path, participant_id):
    stimuli_file_path = \
        os.path.join(stimuli_path, "p{0}_stimuli.csv".format(str(participant_id).zfill(2)))
    with(open(stimuli_file_path, "r")) as csv_file:
        csv_reader = csv.reader(csv_file)
        stimuli_order = []
        for line in csv_reader:
            stimuli_order.append(line[0][0:2])
    return stimuli_order
            
def prepare_label_files(file_path, output_path, stimuli_path):
    '''
    Converts csv file created by google doc to separated csv files for each participant
    '''
    with(open(file_path, "r")) as csv_file:
        csv_reader = csv.reader(csv_file)
        i = 0
        for line in csv_reader:
            if i == 0:
                i += 1
                continue
            header = ["stimuli", "label", "label_v", "label_a", "emotion", "intensity", 
                      "valence", "arousal", "dominance", "im_conv_pos", "im_conv_int"]
            participant_id = line[1]
            stimuli_order = __get_stimuli_list(stimuli_path, participant_id)
            file_name = "p{0}_labels.csv".format(str(participant_id).zfill(2))
            label_file_path = os.path.join(output_path, file_name)
            with(open(label_file_path, "w")) as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(header)
                for j in range(8):
                    stimuli = stimuli_order[j]
                    label = LABEL[stimuli]
                    if label == 0:
                        label_v = 1
                        label_a = 1
                    elif label == 1:
                        label_v = 1
                        label_a = 0
                    elif label == 2:
                        label_v = 0
                        label_a = 1
                    elif label == 3:
                        label_v = 0
                        label_a = 0
                    emotion = EMOTIONS[line[j*7+2]]
                    intensity = line[j*7+3]
                    valence = line[j*7+4]
                    arousal = line[j*7+5]
                    dominance = line[j*7+6]
                    im_conv_pos = IM_CONV[line[j*7+7]]
                    im_conv_int = IM_CONV[line[j*7+8]]
                    row = [stimuli, label, label_v, label_a, emotion, intensity,
                           valence, arousal, dominance, im_conv_pos, im_conv_int]
                    csv_writer.writerow(row)

def prepare_label_files_v2(file_path, output_path, stimuli_path):
    '''
    Converts csv file created by google doc to separated csv files for each participant
    '''
    with(open(file_path, "r")) as csv_file:
        csv_reader = csv.reader(csv_file)
        i = 0
        for line in csv_reader:
            if i == 0:
                i += 1
                continue
            header = ["stimuli", "label", "label_v", "label_a", "emotion", "intensity",
                      "valence", "arousal", "dominance", "im_conv_pos", "im_conv_int"]
            participant_id = line[1]
            stimuli_order = __get_stimuli_list(stimuli_path, participant_id)
            file_name = "p{0}_labels.csv".format(str(participant_id).zfill(2))
            label_file_path = os.path.join(output_path, file_name)
            with(open(label_file_path, "w")) as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(header)
                for j in range(8):
                    stimuli = stimuli_order[j]
                    label = LABEL[stimuli]
                    if label == 0:
                        label_v = 1
                        label_a = 1
                    elif label == 1:
                        label_v = 1
                        label_a = 0
                    elif label == 2:
                        label_v = 0
                        label_a = 1
                    elif label == 3:
                        label_v = 0
                        label_a = 0
                    valence = line[j*7+2]
                    arousal = line[j*7+3]
                    dominance = line[j*7+4]
                    emotion = EMOTIONS[line[j*7+5]]
                    intensity = line[j*7+6]
                    im_conv_pos = IM_CONV_V2[line[j*7+7].replace(" ", "")]
                    im_conv_int = IM_CONV_V2[line[j*7+8].replace(" ", "")]
                    row = [stimuli, label, label_v, label_a, emotion, intensity,
                           valence, arousal, dominance, im_conv_pos, im_conv_int]
                    csv_writer.writerow(row)

def prepare_category(input_path, participant_id, output_path="f2f"):
    stimuli_list = ["11", "12", "13", "14", "20", "22", "23", "24", 
                    "31", "33", "34", "35", "40", "41", "42", "44"]
    label_dict = {"HVHA":0, "HVLA":1, "LVHA":2, "LVLA":3}
    input_file = os.path.join(input_path, "p{0}_labels.csv".format(str(participant_id).zfill(2)))
    print(input_file)
    with(open(input_file, "r")) as csv_file:
        csv_reader = csv.reader(csv_file)
        i = 0
        for line in csv_reader:
            if i == 0:
                i += 1
                continue
            for stimuli in stimuli_list:
                if line[0] == stimuli:
                    file_name = stimuli + ".csv"
                    with(open(os.path.join("stimuli", file_name), "a")) as stimuli_file:
                        stimuli_file_writer = csv.writer(stimuli_file)
                        stimuli_file_writer.writerow([participant_id] +line)
            for name, label in label_dict.items():
                if line[1] == str(label):
                    file_name = name + ".csv"
                    with(open(os.path.join(output_path, file_name), "a")) as label_file:
                        label_file_writer = csv.writer(label_file)
                        label_file_writer.writerow([participant_id] + line)

def prepare_all_in_one(input_path, output_path="all_f2f.csv", task=0):
    with(open(output_path, "w")) as all_file:
        csv_writer = csv.writer(all_file)
        header = ["participant", "stimuli", "label", "label_v", "label_a", "emotion", "intensity",
                  "valence", "arousal", "dominance", "im_conv_pos", "im_conv_int", "task"]
        csv_writer.writerow(header)
        for participant_id in range(1, 17): 
            input_file = os.path.join(input_path, "p{0}_labels.csv".format(str(participant_id).zfill(2)))
            with(open(input_file, "r")) as csv_file:
                csv_reader = csv.reader(csv_file)
                i = 0
                for line in csv_reader:
                    if i == 0:
                        i += 1
                        continue
                    csv_writer.writerow([participant_id] + line + [task])


# Convert google doc file to separated label files for each participant
#prepare_label_files("after_f2f.csv", "f2f_labels", "/media/nastaran/HDD/projects/f2f_remote/stimuli/f2f")
#prepare_label_files("after_remote.csv", "remote_labels", "/media/nastaran/HDD/projects/f2f_remote/stimuli/remote")
#prepare_label_files_v2("after_f2f2.csv", "f2f_labels", "/media/nastaran/HDD/projects/f2f_remote/stimuli/f2f")
#prepare_label_files_v2("after_remote2.csv", "remote_labels", "/media/nastaran/HDD/projects/f2f_remote/stimuli/remote")


for i in range(1, 17):
    prepare_category("f2f_labels/", i, output_path="f2f")
    prepare_category("remote_labels/", i, output_path="remote")

# Put all participants labels in one file
#prepare_all_in_one("f2f_labels/", output_path="all_f2f.csv", task=0)
#prepare_all_in_one("remote_labels/", output_path="all_remote.csv", task=1)
