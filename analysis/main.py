
from exp1_1.exp1_1 import cross_subject as exp1_1_cross_subject

from exp1_1.exp1_1 import subject_dependent as exp1_1_subject_dependent
from exp1_1.exp1_1 import subject_independent as exp1_1_subject_independent
'''
from exp1_1.exp1_1_lstm import cross_subject as exp1_1_lstm_cross_subject
from exp1_1.exp1_1_lstm import subject_dependent as exp1_1_lstm_subject_dependent
from exp1_1.exp1_1_lstm import subject_independent as exp1_1_lstm_subject_independent

from exp1_0.exp1_0 import cross_subject as exp1_0_cross_subject

from exp1_0.exp1_0 import subject_dependent as exp1_0_subject_dependent
from exp1_0.exp1_0 import subject_independent as exp1_0_subject_independent

from exp1_0.exp1_0_lstm import cross_subject as exp1_0_lstm_cross_subject

from exp1_0.exp1_0_lstm import subject_independent as exp1_0_lstm_subject_independent
from exp1_0.exp1_0_lstm import subject_dependent as exp1_0_lstm_subject_dependent
'''

#from exp1_0.exp1_0_mixed import cross_subject as exp1_0_mixed_cross_subject
#from cross_exp.cross_exp import cross_dataset_evaluation
#cross_dataset_evaluation()


LABEL_TYPE = "valence"

from exp1_0.exp1_0 import cross_subject as exp1_0_cross_subject
exp1_0_cross_subject(label_type=LABEL_TYPE, window_size=10)
#exp1_0_lstm_cross_subject(label_type=LABEL_TYPE, window_size=1)
