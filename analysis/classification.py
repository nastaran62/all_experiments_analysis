import pickle
import numpy as np
import os
import multiprocessing

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
#from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.linear_model import LogisticRegression
from sklearn.utils import class_weight
from sklearn.utils import shuffle

from tensorflow.keras.callbacks import Callback

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv1D, MaxPooling1D, MaxPooling1D, \
    TimeDistributed
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, \
    mean_squared_error, classification_report, accuracy_score, f1_score
from tensorflow.keras import backend as K

class Checkpoint(Callback):

    def __init__(self, test_data, filename):
        self.test_data = test_data
        self.filename = filename

    def on_train_begin(self, logs=None):
        self.fscore = 0.

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        pred_values = self.model.predict(x)
        y_pred = np.argmax(pred_values, axis=1)
        y = np.argmax(y, axis=1)
        precision, recall, f_score, support = \
            precision_recall_fscore_support(y,
                                            y_pred,
                                            average='weighted')


        # Save your model when a better trained model was found
        if f_score > self.fscore:
            self.fscore = f_score
            self.model.save(self.filename, overwrite=True)
            print('********************************************* Higher fscore', f_score, 'found. Save as %s' % self.filename)
        else:
            print("fscore did not improve for ", self.filename, "from ", self.fscore)
        return


class F1History(Callback):

    def __init__(self, train, validation=None):
        super(F1History, self).__init__()
        self.validation = validation
        self.train = train

    def on_epoch_end(self, epoch, logs={}):
        logs['F1_score_train'] = float('-inf')
        X_train, y_train = self.train[0], self.train[1]
        pred_values = self.model.predict(X_train)
        y_pred = np.argmax(pred_values, axis=1)
        y_train = np.argmax(y_train,axis=1)
        score = f1_score(y_train, y_pred)

        if (self.validation):
            logs['F1_score_val'] = float('-inf')
            X_valid, y_valid = self.validation[0], self.validation[1]
            pred_values = self.model.predict(X_valid)
            y_val_pred = np.argmax(pred_values, axis=1)
            y_valid = np.argmax(y_valid, axis=1)
            val_score = f1_score(y_valid, y_val_pred)
            logs['F1_score_train'] = np.round(score, 5)
            logs['F1_score_val'] = np.round(val_score, 5)
        else:
            logs['F1_score_train'] = np.round(score, 5)

class ModalityClassification(multiprocessing.Process):
    def __init__(self, train_x, test_x, train_y, test_y, queue, classes=[0, 1], type="simple", model_name="models/eeg_model.py"):
        super().__init__()
        self.queue = queue
        self.train_x = np.array(train_x)
        self.test_x = np.array(test_x)
        self.train_y = np.array(train_y)
        self.test_y = np.array(test_y)

        self._classification_type = type
        self._classes = list(np.unique(self.train_y))
        self._model_name = model_name

    def run(self):
        if self._classification_type == "random_forest":
            predictions = \
                self._random_forest()
        if self._classification_type == "mixed_random_forest":
            predictions = \
                self._mixed_random_forest()
        elif self._classification_type == "feature_selection_random_forest":
            predictions = \
                self._feature_selection_random_forest()
        elif self._classification_type == "svm":
            predictions = \
                self._svm()
        elif self._classification_type == "lstm":
            predictions = \
                self._lstm_classification()
        print("training done")
        self.queue.put(predictions)
        print("done")

    def _lstm_classification(self):
        try:
            mean = np.mean(self.train_x)
            std = np.std(self.train_x)
            self.train_x = (self.train_x - mean) / std
            self.test_x = (self.test_x - mean) / std
            class_weights = \
                class_weight.compute_class_weight('balanced',
                                                  self._classes,
                                                  self.train_y)
            class_weights = dict(enumerate(class_weights))
            self.train_y = to_categorical(self.train_y)
            self.test_y = to_categorical(self.test_y)
            print("Preparing classification")
            checkpoint = \
                ModelCheckpoint(self._model_name,
                                monitor='val_accuracy',
                                verbose=1,
                                save_weights_only=False,
                                mode='max',
                                period=1)
            checkpoint = Checkpoint((np.array(self.test_x),
                                     np.array(self.test_y)),
                                    self._model_name)
            early_stopping = \
                EarlyStopping(monitor='val_accuracy',
                              patience=50,
                              verbose=1,
                              mode='max')
            reduceLR = ReduceLROnPlateau(monitor='val_accuracy',
                                         factor=0.5,
                                         patience=10,
                                         min_lr=0.0001)
            print("creating model")
            model = simple_lstm((self.train_x.shape[1], self.train_x.shape[2]),
                                80,  # lstm layers
                                len(self._classes),  # number of classes
                                dropout=0.5)
            print("model.summary")
            model.summary()

            print("Start classification")

            history = \
                model.fit(np.array(self.train_x),
                          np.array(self.train_y),
                          batch_size=32,
                          epochs=1000,
                          class_weight=class_weights,
                          validation_data=(np.array(self.test_x),
                                           np.array(self.test_y)),
                          callbacks=[checkpoint, early_stopping, reduceLR])
            model = load_model(self._model_name)
            predictions = model.predict_proba(np.array(self.test_x))
            # plot history
            #pyplot.plot(history.history['loss'], label='train')
            #pyplot.plot(history.history['val_loss'], label='test')
            #pyplot.legend()
            #pyplot.show()
            pred_values = model.predict(self.test_x)

            predicted_labels = np.argmax(pred_values, axis=1)
            self.test_y = np.argmax(self.test_y, axis=1)

            acc = accuracy_score(predicted_labels, self.test_y)
            print(classification_report(self.test_y, predicted_labels))
            precision, recall, f_score, support = \
                precision_recall_fscore_support(self.test_y,
                                                predicted_labels,
                                                average='weighted')
        except Exception as error:
            acc=0
            f_score=0
            predicted_labels=0
            predictions=0
        return acc, f_score, predicted_labels, predictions


    def _random_forest(self):
        train_mean = np.mean(self.train_x)
        train_std = np.std(self.train_x)
        self.train_x = (self.train_x - train_mean) / train_std
        self.test_x = (self.test_x - train_mean) / train_std
        self.train_x, self.train_y = shuffle(self.train_x, self.train_y)
        #clf = svm.SVC(C=150, gamma="auto", probability=True)
        clf = RandomForestClassifier(n_estimators=200, max_features="auto", class_weight='balanced')
        #clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(self.train_x, self.train_y)
        pickle.dump(clf, open(self._model_name, "wb"))
        pred_values = clf.predict(self.test_x)
        acc = accuracy_score(pred_values, self.test_y)
        print(acc)
        #print(classification_report(self.test_y, pred_values))
        precision, recall, f_score, support = \
            precision_recall_fscore_support(self.test_y,
                                            pred_values,
                                            average='weighted')
        predictions = clf.predict_proba(self.test_x)
        return acc, f_score, pred_values, predictions

    def _mixed_random_forest(self):
        train_mean = np.mean(self.train_x)
        train_std = np.std(self.train_x)
        self.train_x = (self.train_x - train_mean) / train_std
        self.test_x = (self.test_x - train_mean) / train_std

        samples, windows, features = self.train_x.shape
        labels = []
        for i in range(samples):
            for j in range(windows):
                labels.append(self.train_y[i])

        print("************************", self.train_x.shape, self.train_y.shape)
        self.train_y = np.array(labels)
        self.train_x = self.train_x.reshape(-1, self.train_x.shape[-1])
        print("************************", self.train_x.shape, self.train_y.shape)

        self.train_x, self.train_y = shuffle(self.train_x, self.train_y)
        clf = RandomForestClassifier(n_estimators=200, max_features="auto", class_weight='balanced')
        clf.fit(self.train_x, self.train_y)
        pickle.dump(clf, open(self._model_name, "wb"))

        samples, windows, features = self.test_x.shape
        pred_values = []
        all_predictions = []

        for i in range(samples):
            predictions = clf.predict_proba(self.test_x[i, :, :])
            prediction = np.mean(np.array(predictions), axis=0)
            all_predictions.append(prediction)
            pred_values.append(np.argmax(prediction))

        pred_values = np.array(pred_values)
        acc = accuracy_score(pred_values, self.test_y)
        #print(classification_report(self.test_y, pred_values))
        precision, recall, f_score, support = \
            precision_recall_fscore_support(self.test_y,
                                            pred_values,
                                            average='weighted')
        print("acc, f_score, precision, recall", acc, f_score, precision, recall)
        predictions = np.array(all_predictions)
        return acc, f_score, pred_values, prediction


    def _svm(self):
        train_mean = np.mean(self.train_x)
        train_std = np.std(self.train_x)
        self.train_x = (self.train_x - train_mean) / train_std
        self.test_x = (self.test_x - train_mean) / train_std

        clf = svm.SVC(probability=True,
                class_weight='balanced',
                C=150,
                random_state=100,
                kernel='rbf')
        clf.fit(self.train_x, self.train_y)
        pickle.dump(clf, open(self._model_name, "wb"))
        pred_values = clf.predict(self.test_x)
        acc = accuracy_score(pred_values, self.test_y)
        #print(classification_report(self.test_y, pred_values))
        precision, recall, f_score, support = \
            precision_recall_fscore_support(self.test_y,
                                            pred_values,
                                            average='weighted')
        predictions = clf.predict_proba(self.test_x)
        return acc, f_score, pred_values, predictions

    def _knn(self):
        train_mean = np.mean(self.train_x)
        train_std = np.std(self.train_x)
        self.train_x = (self.train_x - train_mean) / train_std
        self.test_x = (self.test_x - train_mean) / train_std

        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(self.train_x, self.train_y)
        pickle.dump(clf, open(self._model_name, "wb"))
        pred_values = clf.predict(self.test_x)
        acc = accuracy_score(pred_values, self.test_y)
        #print(classification_report(self.test_y, pred_values))
        precision, recall, f_score, support = \
            precision_recall_fscore_support(self.test_y,
                                            pred_values,
                                            average='weighted')
        predictions = clf.predict_proba(self.test_x)
        return acc, f_score, pred_values, predictions

    def _feature_selection_random_forest(self, num_features=10):
        train_mean = np.mean(self.train_x)
        train_std = np.std(self.train_x)
        self.train_x = (self.train_x - train_mean) / train_std
        self.test_x = (self.test_x - train_mean) / train_std

        clf = RandomForestClassifier(n_estimators=200, max_features="auto", class_weight='balanced')
        rfe = RFE(clf, num_features)
        fit = rfe.fit(self.train_x, self.train_y)
        print("Num Features: %d" % fit.n_features_)
        print("Selected Features: %s" % fit.support_)
        print("Feature Ranking: %s" % fit.ranking_)
        self.train_x = fit.transform(self.train_x)
        self.test_x = fit.transform(self.test_x)
        clf.fit(self.train_x, self.train_y)
        pickle.dump(clf, open(self._model_name, "wb"))
        pred_values = clf.predict(self.test_x)
        acc = accuracy_score(pred_values, self.test_y)
        print(classification_report(self.test_y, pred_values))
        precision, recall, f_score, support = \
            precision_recall_fscore_support(self.test_y,
                                            pred_values,
                                            average='weighted')
        predictions = clf.predict_proba(self.test_x)
        return acc, f_score, pred_values, predictions


    def _cnn_lstm_classification(self):
        if not os.path.exists("models"):
            os.mkdir("models")
        class_weights = \
            class_weight.compute_class_weight('balanced',
                                              np.unique(self.train_y),
                                              self.train_y)
        class_weights = dict(enumerate(class_weights))
        #scaler = StandardScaler()
        # scaler.fit(self.train_x)
        #self.train_x = scaler.transform(self.train_x)
        #self.test_x = scaler.transform(self.test_x)

        self.train_y = to_categorical(self.train_y)
        self.test_y = to_categorical(self.test_y)

        n_timesteps, n_features, n_outputs = \
            self.train_x.shape[1], self.train_x.shape[2], self.train_y.shape[1]

        verbose, epochs, batch_size, n_steps = 1, 25, 32, 10

        # reshape into subsequences (samples, time steps, rows, cols, channels)
        n_length = int(n_timesteps/n_steps)
        self.train_x = self.train_x.reshape((self.train_x.shape[0], n_steps, n_length, n_features))
        self.test_x = self.test_x.reshape((self.test_x.shape[0], n_steps, n_length, n_features))
        # define model
        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'),
                                  input_shape=(None, n_length, n_features)))
        model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
        model.add(TimeDistributed(Dropout(0.5)))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(80))
        model.add(Dropout(0.5))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(n_outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy'])
        # fit network
        checkpoint = \
            ModelCheckpoint("models/eeg_cnn_model.h5",
                            monitor='val_accuracy',
                            verbose=1,
                            save_best_only=True,
                            save_weights_only=False,
                            mode='auto',
                            period=1)
        early_stopping = \
            EarlyStopping(monitor='val_loss',
                          patience=50,
                          verbose=1,
                          mode='auto')

        model.summary()
        model.fit(self.train_x, self.train_y, epochs=epochs, batch_size=batch_size, verbose=verbose,
                  class_weight=class_weights,
                  validation_data=(np.array(self.test_x),
                                   np.array(self.test_y)),
                  callbacks=[checkpoint, early_stopping])#, ReduceLROnPlateau])
        # evaluate model
        _, accuracy = model.evaluate(self.test_x, self.test_y, batch_size=batch_size, verbose=0)
        physiological_model = load_model("models/eeg_cnn_model.h5")
        preds_physiological = physiological_model.predict_proba(np.array(self.test_x))
        print(accuracy)
        return preds_physiological

def simple_lstm(input_shape, lstm_layers, num_classes, dropout=0.7):
    '''
    Model definition
    '''
    print("Input_shape:", input_shape, " lstm_layers:", lstm_layers, " num_classes: ", num_classes, " dropout:", dropout)
    model = Sequential()
    print("Mdel.add 1")
    model.add(LSTM(lstm_layers, input_shape=input_shape, return_sequences=True))
    #model.add(LSTM(10, return_sequences=True))
    print("model.add 2")
    model.add(LSTM(30))
    #model.add(Dropout(0.4))
    print("model.add 3")
    model.add(Dense(num_classes, activation='softmax'))
    print("compiling model")
    model.compile(loss="categorical_crossentropy",  # categorical_crossentropy
                  optimizer=optimizers.Adam(lr=0.01), metrics=["accuracy"])
    return model


def multimodal_classification(train_labels, test_labels, eeg, gsr, ppg):
    classifiers = []
    eeg_train, eeg_test, eeg_classification_method, eeg_model_name = eeg
    eeg_queue = multiprocessing.Queue()
    eeg_classifier = \
        ModalityClassification(eeg_train,
                                eeg_test,
                                train_labels,
                                test_labels,
                                eeg_queue,
                                type=eeg_classification_method,
                                model_name=eeg_model_name)
    classifiers.append(eeg_classifier)

    gsr_train, gsr_test, gsr_classification_method, gsr_model_name = gsr
    gsr_queue = multiprocessing.Queue()
    gsr_classifier = \
        ModalityClassification(gsr_train,
                                gsr_test,
                                train_labels,
                                test_labels,
                                gsr_queue,
                                type=gsr_classification_method,
                                model_name=gsr_model_name)
    classifiers.append(gsr_classifier)


    ppg_train, ppg_test, ppg_classification_method, ppg_model_name = ppg
    ppg_queue = multiprocessing.Queue()
    ppg_classifier = \
        ModalityClassification(ppg_train,
                                ppg_test,
                                train_labels,
                                test_labels,
                                ppg_queue,
                                type=ppg_classification_method,
                                model_name=ppg_model_name)
    classifiers.append(ppg_classifier)


    for classifier in classifiers:
        classifier.start()

    eeg_accuracy, eeg_fscore, eeg_preds, eeg_probabilities = eeg_queue.get()
    gsr_accuracy, gsr_fscore, gsr_preds, gsr_probabilities = gsr_queue.get()
    ppg_accuracy, ppg_fscore, ppg_preds, ppg_probabilities = ppg_queue.get()

    for classifier in classifiers:
        classifier.join()

    eeg = eeg_accuracy, eeg_fscore, eeg_preds, eeg_probabilities
    gsr = gsr_accuracy, gsr_fscore, gsr_preds, gsr_probabilities
    ppg = ppg_accuracy, ppg_fscore, ppg_preds, ppg_probabilities

    return eeg, gsr, ppg


def mixed_multimodal_classification(train_labels, test_labels, eeg, gsr, ppg):
    classifiers = []
    eeg_train, eeg_test, eeg_classification_method, eeg_model_name = eeg
    eeg_queue = multiprocessing.Queue()
    eeg_classifier = \
        ModalityClassification(eeg_train,
                                eeg_test,
                                train_labels,
                                test_labels,
                                eeg_queue,
                                type=eeg_classification_method,
                                model_name=eeg_model_name)
    classifiers.append(eeg_classifier)

    gsr_train, gsr_test, gsr_classification_method, gsr_model_name = gsr
    gsr_queue = multiprocessing.Queue()
    gsr_classifier = \
        ModalityClassification(gsr_train,
                                gsr_test,
                                train_labels,
                                test_labels,
                                gsr_queue,
                                type=gsr_classification_method,
                                model_name=gsr_model_name)
    classifiers.append(gsr_classifier)


    ppg_train, ppg_test, ppg_classification_method, ppg_model_name = ppg
    ppg_queue = multiprocessing.Queue()
    ppg_classifier = \
        ModalityClassification(ppg_train,
                                ppg_test,
                                train_labels,
                                test_labels,
                                ppg_queue,
                                type=ppg_classification_method,
                                model_name=ppg_model_name)
    classifiers.append(ppg_classifier)


    for classifier in classifiers:
        classifier.start()

    eeg_accuracy, eeg_fscore, eeg_preds, eeg_probabilities, eeg_model = eeg_queue.get()
    gsr_accuracy, gsr_fscore, gsr_preds, gsr_probabilities, gsr_model = gsr_queue.get()
    ppg_accuracy, ppg_fscore, ppg_preds, ppg_probabilities, ppg_model = ppg_queue.get()

    for classifier in classifiers:
        classifier.join()

    eeg = eeg_accuracy, eeg_fscore, eeg_preds, eeg_probabilities, eeg_mode
    gsr = gsr_accuracy, gsr_fscore, gsr_preds, gsr_probabilities, gsr_model
    ppg = ppg_accuracy, ppg_fscore, ppg_preds, ppg_probabilities, ppg_model

    return eeg, gsr, ppg

def voting_fusion(eeg_preds, gsr_preds, ppg_preds, test_labels):
    preds_fusion = []
    for i in range(len(eeg_preds)):
        if eeg_preds[i] + ppg_preds[i] + gsr_preds[i] > 1 :
            preds_fusion.append(1)
        else:
            preds_fusion.append(0)
    accuracy = accuracy_score(preds_fusion, test_labels)
    print(classification_report(test_labels, preds_fusion))
    precision, recall, fscore, support = \
        precision_recall_fscore_support(test_labels,
                                        preds_fusion,
                                        average='weighted')
    print(accuracy)
    return accuracy, fscore

def equal_fusion(eeg_preds, gsr_preds, ppg_preds, test_labels, eeg_weight=0.33, gsr_weight=0.33, ppg_weight=0.33):
    all = eeg_weight*eeg_preds + gsr_weight* gsr_preds + ppg_weight*ppg_preds
    #all = all / 3
    print(all.shape)
    preds_fusion = np.argmax(all, axis=1)
    accuracy = accuracy_score(preds_fusion, test_labels)
    print(classification_report(test_labels, preds_fusion))
    precision, recall, fscore, support = \
        precision_recall_fscore_support(test_labels,
                                        preds_fusion,
                                        average='weighted')
    print(accuracy)
    return accuracy, fscore

def weighted_fusion():
    pass


def random_forest(train_x, test_x, train_y, test_y):
    train_mean = np.mean(train_x)
    train_std = np.std(train_x)
    train_x = (train_x - train_mean) / train_std
    test_x = (test_x - train_mean) / train_std

    clf = RandomForestClassifier(n_estimators=200, max_features="auto", class_weight='balanced')
    clf.fit(train_x, train_y)
    pred_values = clf.predict(test_x)
    accuracy = accuracy_score(pred_values, test_y)
    #print(classification_report(test_y, pred_values))
    precision, recall, f_score, support = \
        precision_recall_fscore_support(test_y,
                                        pred_values,
                                        average='weighted',
                                        zero_division=0)
    print("accuracy:", accuracy, "f_score:", f_score, "\n")
    #predictions = clf.predict_proba(test_x)
    return accuracy, f_score
