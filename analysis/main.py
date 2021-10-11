
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


LABEL_TYPE = "emotion"

#from exp1_0.exp1_0 import cross_subject as exp1_0#_cross_subject
#exp1_0_cross_subject(label_type=LABEL_TYPE, window_size=1, calculate=False, shuffle=False, fold=5)

#from exp1_0.exp1_0 import subject_independent as exp1_0_subject_independent
#exp1_0_subject_independent(label_type=LABEL_TYPE, window_size=1, calculate=False, fold=1)

#from exp1_0.exp1_0 import subject_dependent as exp1_0_subject_dependent
#exp1_0_subject_dependent(label_type=LABEL_TYPE, window_size=1, calculate=False, shuffle=False, fold=5)

#from exp1_0.exp1_0_lstm import cross_subject as exp1_0_lstm_cross_subject
#exp1_0_lstm_cross_subject(label_type=LABEL_TYPE, window_size=1, calculate=False, fold=5)

#from exp1_0.exp1_0_lstm import subject_independent as exp1_0_lstm_subject_independent
#exp1_0_lstm_subject_independent(label_type=LABEL_TYPE, window_size=1, calculate=False, fold=4)

#from exp1_0.exp1_0_lstm import subject_dependent as exp1_0_lstm_subject_dependent
#exp1_0_lstm_subject_dependent(label_type=LABEL_TYPE, window_size=1, calculate=False, fold=2)

from exp1_1.exp1_1 import cross_subject as exp1_1_cross_subject
'''
exp1_1_cross_subject(label_type="liwc_valence", window_size=5, calculate=True, shuffle=shuffle, fold=5)
label_types = ["arousal", "valence", "emotion"]
shuffles = [False]
for label_type in label_types:
    for shuffle in shuffles:
        exp1_1_cross_subject(label_type=label_type, window_size=0, calculate=True, shuffle=shuffle, fold=5)
'''
from exp1_1.exp1_1 import subject_dependent as exp1_1_subject_dependent

#exp1_1_subject_dependent(label_type="arousal", window_size=5, calculate=True, shuffle=False, fold=3)
'''
label_types = ["arousal", "valence", "emotion"]
shuffles = [False]
for label_type in label_types:
    for shuffle in shuffles:
        exp1_1_subject_dependent(label_type=label_type, window_size=5, calculate=False, shuffle=shuffle, fold=3)
'''

from exp1_1.exp1_1 import subject_independent as exp1_1_subject_independent
'''
label_types = ["arousal", "valence", "emotion"]
shuffles = [False]
for label_type in label_types:
    for shuffle in shuffles:
        exp1_1_subject_independent(label_type=label_type, window_size=5, calculate=False, fold=5)
'''
#exp1_1_subject_independent(label_type=LABEL_TYPE, window_size=5, calculate=False, fold=1)

from exp1_1.exp1_1_lstm import subject_independent as exp1_1_lstm_subject_independent
'''
label_types = ["arousal", "valence","emotion"]
shuffles = [False]
for label_type in label_types:
    for shuffle in shuffles:
        exp1_1_lstm_subject_independent(label_type=label_type, window_size=5, calculate=False, fold=5)
'''
#exp1_1_lstm_subject_independent(label_type=LABEL_TYPE, window_size=1, calculate=True, fold=1)

from exp1_1.exp1_1_lstm import cross_subject as exp1_1_lstm_cross_subject

label_types = ["arousal", "valence", "emotion"]
for label_type in label_types:
    exp1_1_lstm_cross_subject(label_type=label_type, window_size=5, calculate=False, fold=10)


from exp1_1.exp1_1_lstm import subject_dependent as exp1_1_lstm_subject_dependent
'''
label_types = ["emotion"]
shuffles = [False]
for label_type in label_types:
    for shuffle in shuffles:
        exp1_1_lstm_subject_dependent(label_type=label_type, window_size=5, calculate=False, fold=3)
'''
