# Reads data from prepared_data and preprocess them and save in preprocessed_data
# preprocessing modules come from processing.preprocessing package

import os
import sys
sys.path.append('../../octopus-sensing/')
from octopus_sensing.preprocessing.preprocess_devices import preprocess_devices_by_path


def prepare_exp1_1():
	path = "exp1_1/raw_data"
	output_path = "exp1_1/prepared_data"
	all_participants = os.listdir(path)
	for participant in all_participants:
	    participant_output_path = os.path.join(output_path, participant)
	    print(participant)
	    devices_path = {"openbci": os.path.join(path, participant, "eeg"),
			    "shimmer3": os.path.join(path, participant, "shimmer")}
	    preprocess_devices_by_path(devices_path, 
			               participant_output_path,
			               openbci_sampling_rate=128,
			               shimmer3_sampling_rate=128,
			               signal_preprocess=False)

def prepare_exp1_0():
	path = "exp1_0/raw_data"
	output_path = "exp1_0/prepared_data"
	all_participants = os.listdir(path)
	for participant in all_participants:
	    participant_output_path = os.path.join(output_path, participant)
	    print(participant)
	    devices_path = {"openbci": os.path.join(path, participant, "eeg"),
			    "shimmer3": os.path.join(path, participant, "shimmer")}
	    preprocess_devices_by_path(devices_path, 
			               participant_output_path,
			               openbci_sampling_rate=128,
			               shimmer3_sampling_rate=128,
			               signal_preprocess=False)

prepare_exp1_0()

prepare_exp1_1()


