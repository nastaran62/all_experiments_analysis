import os

f2f_in_path = "/home/zsaf419/Documents/Study_2/output"
remote_in_path = "/home/zsaf419/Documents/Study_2/output_remote"

for p in ["p01", "p02", "p03", "p04", "p05", "p06", "p07", "p08", "p09", "p10"]:
    participant_f2f_in_path = f2f_in_path + "/{0}/webcam".format(p)
    participant_remote_in_path = remote_in_path + "/{0}/webcam".format(p)
    for file in os.listdir(participant_f2f_in_path):
        command = \
            "./FeatureExtraction -inroot {0} -aus -f {1} -out_dir processed/f2f/{2}"
        os.system(command.format(participant_f2f_in_path, file, p))
    for file in os.listdir(participant_remote_in_path):
        command = \
            "./FeatureExtraction -inroot {0} -aus -f {1} -out_dir processed/remote/{2}"
        os.system(command.format(participant_remote_in_path, file, p))
