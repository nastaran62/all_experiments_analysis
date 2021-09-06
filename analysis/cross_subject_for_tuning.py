import numpy as np
import pickle
from sklearn import pipeline
from sklearn.utils import shuffle
from feature_extraction import partitioning_and_getting_features_with_self_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
#from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle
from sklearn.feature_selection import RFE
from numpy import mean
from numpy import std
from sklearn.neighbors import KNeighborsClassifier

participants = [0, 1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19 ,20 ,21, 22, 23]
#participants = [9, 11, 12, 14, 15, 16, 17, 18, 19 ,20 ,21, 22, 23]
def main():
    '''
    input_path = "../experimental_data/exp1_1/preprocessed_data"
    label_path = "../experimental_data/exp1_1/prepared_labels"
    all_eeg, all_gsr, all_ppg, all_emotions, all_arousals, all_valences = \
        partitioning_and_getting_features_with_self_report(input_path, label_path, trial_size=60)
    
    pickle.dump(all_eeg, open("eeg.pickle", "wb"))
    pickle.dump(all_gsr, open("gsr.pickle", "wb"))
    pickle.dump(all_ppg, open("ppg.pickle", "wb"))
    pickle.dump((all_emotions, all_arousals, all_valences), open("labels.pickle", "wb"))
    '''

    all_eeg = pickle.load(open("eeg.pickle", "rb"))
    all_gsr = pickle.load(open("gsr.pickle", "rb"))
    all_ppg = pickle.load(open("ppg.pickle", "rb"))
    (all_emotions, all_arousals, all_valences) = pickle.load(open("labels.pickle", "rb"))
    
    eeg = make_np_array(all_eeg)
    gsr = make_np_array(all_gsr)
    ppg = make_np_array(all_ppg)
    emotion = make_np_array(all_emotions)
    arousal = make_np_array(all_arousals)
    valence = make_np_array(all_valences)

    print("arousal", list(arousal).count(0), list(arousal).count(1))
    print("valence", list(valence).count(0), list(valence).count(1))

    # Shuffling
    permutation = np.random.permutation(emotion.shape[0])
    eeg = eeg[permutation, :]
    gsr = gsr[permutation, :]
    ppg = ppg[permutation, :]
    arousal = arousal[permutation]
    valence = valence[permutation]
    emotion = emotion[permutation]
    
    labels = arousal
    pipeline = rf_pipeline()
    kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=100)
    n_scores = cross_val_score(pipeline, eeg, labels, scoring='f1', cv=kf, n_jobs=-1, error_score='raise')
    print('eeg: %.3f (%.3f)' % (mean(n_scores), std(n_scores))) 
    pipeline = rf_pipeline()
    n_scores = cross_val_score(pipeline, gsr, labels, scoring='f1', cv=kf, n_jobs=-1, error_score='raise')
    print('gsr: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
    n_scores = cross_val_score(pipeline, ppg, labels, scoring='f1', cv=kf, n_jobs=-1, error_score='raise')
    print('ppg: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

def make_np_array(data):
    trials = []
    p = 0
    for participant in data:
        for trial in participant:
            for part in trial:
                if p in participants:
                    trials.append(np.array(part))
        p += 1
    return np.array(trials)

def svm_pipeline():
    model = svm.SVC(probability=True,
                  class_weight='balanced',
                  C=500,
                  random_state=100,
                  kernel='rbf')
    pipeline = Pipeline(steps=[('m',model)])
    return pipeline

def rf_pipeline():
    model = RandomForestClassifier(n_estimators=200,
                                   max_features="auto",
                                   class_weight='balanced')
    rfe = RFE(model, 10)
    pipeline = Pipeline(steps=[('s',rfe),('m',model)])

    return pipeline

def knn_pipeline():
    model = KNeighborsClassifier(n_neighbors=5, metric='manhattan',weights='uniform')
    pipeline = Pipeline(steps=[('m',model)])
    return pipeline
                    
if __name__ == "__main__":
    main()