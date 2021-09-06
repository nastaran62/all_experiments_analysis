from exp1_1.exp1_1 import cross_subject as exp1_1_cross_subject
from exp1_1.exp1_1 import subject_dependent as exp1_1_subject_dependent
from exp1_1.exp1_1 import subject_independent as exp1_1_subject_independent

from exp1_1.exp1_1_lstm import cross_subject as exp1_1_lstm_cross_subject

LABEL_TYPE = "valence"

exp1_1_lstm_cross_subject(label_type=LABEL_TYPE, window_size=5)