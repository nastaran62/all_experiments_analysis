import pickle
import numpy as np
import os
import multiprocessing

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
#from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn import svm
#from sklearn.linear_model import LogisticRegression
from sklearn.utils import class_weight
from sklearn.utils import shuffle

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
    mean_squared_error, classification_report, accuracy_score

def feature_selection_random_forest(x_train, x_test, y_train, y_test, num_features=10):
    train_mean = np.mean(x_train)
    train_std = np.std(x_train)
    x_train = (x_train - train_mean) / train_std
    x_test = (x_test - train_mean) / train_std

    clf = RandomForestClassifier(n_estimators=200, max_features="auto", class_weight='balanced')
    rfe = RFE(clf, num_features)
    fit = rfe.fit(x_train, y_train)
    print("Num Features: %d" % fit.n_features_)
    print("Selected Features: %s" % fit.support_)
    print("Feature Ranking: %s" % fit.ranking_)
    x_train = fit.transform(x_train)
    x_test = fit.transform(x_test)
    clf.fit(x_train, y_train)
    pred_values = clf.predict(x_test)
    acc = accuracy_score(pred_values, y_test)
    print(classification_report(y_test, pred_values))
    precision, recall, f_score, support = \
        precision_recall_fscore_support(y_test,
                                        pred_values,
                                        average='weighted')
    return acc, f_score, pred_values

def random_forest(x_train, x_test, y_train, y_test):
    train_mean = np.mean(x_train)
    train_std = np.std(x_train)
    x_train = (x_train - train_mean) / train_std
    x_test = (x_test - train_mean) / train_std

    clf = RandomForestClassifier(n_estimators=200, max_features="auto", class_weight='balanced')
    clf.fit(x_train, y_train)
    pred_values = clf.predict(x_test)
    acc = accuracy_score(pred_values, y_test)
    print(classification_report(y_test, pred_values))
    precision, recall, f_score, support = \
        precision_recall_fscore_support(y_test,
                                        pred_values,
                                        average='weighted')
    return acc, f_score, pred_values

def svm_classification(x_train, x_test, y_train, y_test):
    train_mean = np.mean(x_train)
    train_std = np.std(x_train)
    x_train = (x_train - train_mean) / train_std
    x_test = (x_test - train_mean) / train_std
    clf = svm.SVC(probability=True,
                  class_weight='balanced',
                  C=500,
                  random_state=100,
                  kernel='rbf')
    clf.fit(x_train, y_train)
    pred_values = clf.predict(x_test)
    acc = accuracy_score(pred_values, y_test)
    print(classification_report(y_test, pred_values))
    precision, recall, f_score, support = \
        precision_recall_fscore_support(y_test,
                                        pred_values,
                                        average='weighted')
    print(acc)
    return acc, f_score, pred_values


class ModalityClassification(multiprocessing.Process):
    def __init__(self, train_x, test_x, train_y, test_y, queue, classes=[0, 1], type="simple", model_name="models/eeg_model.py"):
        super().__init__()
        self.queue = queue
        self.train_x = np.array(train_x)
        self.test_x = np.array(test_x)
        self.train_y = np.array(train_y)
        self.test_y = np.array(test_y)

        self._classification_type = type
        self._classes = classes
        self._model_name = model_name

    def run(self):
        if self._classification_type == "random_forest":
            predictions = \
                self._random_forest()
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

    def _random_forest(self):
        train_mean = np.mean(self.train_x)
        train_std = np.std(self.train_x)
        self.train_x = (self.train_x - train_mean) / train_std
        self.test_x = (self.test_x - train_mean) / train_std

        clf = RandomForestClassifier(n_estimators=200, max_features="auto", class_weight='balanced')
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

    def _svm(self):
        train_mean = np.mean(self.train_x)
        train_std = np.std(self.train_x)
        self.train_x = (self.train_x - train_mean) / train_std
        self.test_x = (self.test_x - train_mean) / train_std

        clf = svm.SVC(probability=True,
                class_weight='balanced',
                C=500,
                random_state=100,
                kernel='rbf')       
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

    def _lstm_classification(self):
        mean = np.mean(self.train_x)
        std = np.std(self.train_x)
        self.train_x = (self.train_x - mean) / std
        self.test_x = (self.test_x - mean) / std
        if not os.path.exists("models"):
            os.mkdir("models")
        class_weights = \
            class_weight.compute_class_weight('balanced',
                                              np.unique(self.train_y),
                                              self.train_y)
        class_weights = dict(enumerate(class_weights))
        self.train_y = to_categorical(self.train_y)
        self.test_y = to_categorical(self.test_y)
        print("Preparing classification")
        checkpoint = \
            ModelCheckpoint(self._model_name,
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
                      batch_size=128,
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
        return acc, f_score, predicted_labels, predictions 

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
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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
                  optimizer=optimizers.Adam(lr=0.01),
                  metrics=['accuracy'])
    return model
