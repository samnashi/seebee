import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM, GRU
from os import listdir
from os.path import isfile, join
import re as regex

import scipy.io as sio
import time


def parse_corpus(fileName):
    input = open(fileName, "r")
    labels = []
    features = []

    feature = []
    for line in input:
        parts = line.split(" ")
        if parts[0] == "label:":
            tokens = parts[1].split(",")
            label = []
            for e in tokens:
                label.append(int(e))
            labels.append(label)
        elif parts[0] == "feature:":
            tokens = parts[1].split(",")
            f = []
            for e in tokens:
                f.append(float(e))
            feature.append(f)
        else:  # END SEQUENCE
            features.append(feature)
            feature = []

    input.close()
    return np.array(features), np.array(labels)


#BasePath = '/home/efi/Riset/LSTM/'
BasePath = "./"
TrainPath = BasePath + "Train/"
TestPath = BasePath + "Test/"

# fileListTrain = [ff for ff in listdir(BasePath) if (isfile(join(BasePath, ff) and regex.match(r'.*DataTrain.txt', ff)))]
# fileListTest = [ff for ff in listdir(BasePath) if (isfile(join(BasePath, ff) and regex.match(r'.*DataTest.txt', ff)))]


# TODO: parameter layer
nLayer = 2

nEpoch = [100, 200, 400, 600, 800]
dataType = ['DATA_IMG', 'DATA_SKL', 'DATA_ALL']
# dataType = ['DASAR_IMG', 'DASAR_SKL', 'DASAR_ALL']
# dataType = ['AKHIRAN_ALL', 'AKHIRAN_IMG', 'AKHIRAN_SKL']
# dataType = ['AWALAN_ALL', 'AWALAN_IMG', 'AWALAN_SKL']
# rnnType = ['LSTM', 'GRU']
rnnType = ['LSTM']

for nE in nEpoch:
    for dT in dataType:
        for rT in rnnType:
            fL = TrainPath + dT + 'DataTrain.txt'
            fT = TestPath + dT + 'DataTest.txt'
            filenameResult = dT + '_e' + str(nE) + '_' + rT + str(nLayer) + 'Result.mat'

            print('fL: ', fL)
            print('fT: ', fT)

            (X_train, Y_train) = parse_corpus(fileName=fL)
            (X_test, Y_test) = parse_corpus(fileName=fT)

            print('X train shape: ', X_train.shape, 'X train ndim: ', X_train.ndim)
            print('Y train shape: ', Y_train.shape, 'Y train ndim: ', Y_train.ndim)

            print('X test shape: ', X_test.shape)
            print('Y test shape: ', Y_test.shape)

            #############################################
            (numInstances, numTimestep, numFeatures) = X_train.shape
            (_, numClasses) = Y_train.shape

            model = Sequential()

            # satu layer
            # model.add(LSTM(output_dim=64, input_dim=numFeatures, activation='sigmoid', inner_activation='hard_sigmoid'))

            if rT == 'GRU':
                if nLayer == 1:
                    print 'L1GRU'
                    model.add(GRU(output_dim=128, input_shape=X_train.shape, activation='sigmoid',
                                  inner_activation='hard_sigmoid'))

                    # kalau mau dua layer GRU
                if nLayer == 2:
                    print 'L2GRU'
                    model.add(GRU(output_dim=128, input_shape=X_train.shape, return_sequences=True, activation='sigmoid',
                            inner_activation='hard_sigmoid'))
                    model.add(GRU(output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid'))

            if rT == 'LSTM':
                if nLayer == 1:
                    print 'L1LSTM'
                    model.add(LSTM(output_dim=64, input_shape=X_train.shape, activation='sigmoid',
                                   inner_activation='hard_sigmoid'))

                    # kalau mau dua layer LSTM
                if nLayer == 2:
                    print 'L2LSTM'
                    model.add(LSTM(units=64, input_shape=X_train.shape, return_sequences=True, activation='sigmoid',inner_activation='hard_sigmoid', name='LSTM_0'))
                    model.add(LSTM(units=64, activation='sigmoid', inner_activation='hard_sigmoid',name='LSTM_1'))

            model.add(Dense(numClasses))
            model.add(Activation('softmax'))

            model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

            #4000 epoch, silakan diganti-ganti

            # TODO: Ganti parameter, misal epoch = 1000
            start_time_train = time.time()
            model.fit(X_train, Y_train, batch_size=16, epochs=nE)
            elapsed_time_train = time.time() - start_time_train

            score_train = model.evaluate(X_train, Y_train, batch_size=16, show_accuracy=True)
            print("accuracy di training data : " + str(score_train))

            score_testing = model.evaluate(X_test, Y_test, batch_size=16, show_accuracy=True)
            print("accuracy di testing data : " + str(score_testing))

            start_time_test = time.time()
            predict_result = model.predict(X_test, batch_size=16)
            elapsed_time_test = time.time() - start_time_test

            sio.savemat(file_name=filenameResult, mdict=dict(score_train=score_train, score_testing=score_testing,
                                                             predict_result=predict_result,
                                                             eta_train=elapsed_time_train,
                                                             eta_test=elapsed_time_test))

