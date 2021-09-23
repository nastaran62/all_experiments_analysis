
from sklearn.utils import shuffle

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


LABEL_TYPE = "arousal"

#from exp1_0.exp1_0 import cross_subject as exp1_0_cross_subject
#exp1_0_cross_subject(label_type=LABEL_TYPE, window_size=1, calculate=False, shuffle=False, fold=5)

#from exp1_0.exp1_0 import subject_independent as exp1_0_subject_independent
#exp1_0_subject_independent(label_type=LABEL_TYPE, window_size=1, calculate=False, shuffle=False, fold=3)

#from exp1_0.exp1_0 import subject_dependent as exp1_0_subject_dependent
#exp1_0_subject_dependent(label_type=LABEL_TYPE, window_size=1, calculate=False, shuffle=False, fold=2)

#from exp1_0.exp1_0_lstm import cross_subject as exp1_0_lstm_cross_subject
#exp1_0_lstm_cross_subject(label_type=LABEL_TYPE, window_size=1, calculate=False, fold=5)

#from exp1_0.exp1_0_lstm import subject_independent as exp1_0_lstm_subject_independent
#exp1_0_lstm_subject_independent(label_type=LABEL_TYPE, window_size=1, calculate=False, fold=4)

#from exp1_0.exp1_0_lstm import subject_dependent as exp1_0_lstm_subject_dependent
#exp1_0_lstm_subject_dependent(label_type=LABEL_TYPE, window_size=1, calculate=False, fold=2)

from exp1_1.exp1_1 import cross_subject as exp1_1_cross_subject
#exp1_1_cross_subject(label_type=LABEL_TYPE, window_size=1, calculate=False, shuffle=True, fold=5)
from exp1_1.exp1_1 import subject_dependent as exp1_1_subject_dependent
#exp1_1_subject_dependent(label_type=LABEL_TYPE, window_size=1, calculate=False, shuffle=False, fold=3)

from exp1_1.exp1_1 import subject_independent as exp1_1_subject_independent
#exp1_1_subject_independent(label_type=LABEL_TYPE, window_size=1, calculate=False, fold=5)

from exp1_1.exp1_1_lstm import subject_independent as exp1_1_lstm_subject_independent
exp1_1_lstm_subject_independent(label_type=LABEL_TYPE, window_size=1, calculate=False, fold=1)
