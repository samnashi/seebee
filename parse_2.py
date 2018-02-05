import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Bidirectional
from keras.engine.topology import Input
from keras.layers.recurrent import LSTM, GRU
from keras.models import Model
from keras.regularizers import l1_l2, l1, l2
from os import listdir
from os.path import isfile, join
import re as regex

import scipy.io as sio
import time
import numpy as np

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
bidir = True

nEpoch = [100, 200, 400, 600, 800]
dataType = ['DATA_IMG', 'DATA_SKL', 'DATA_ALL']
# dataType = ['DASAR_IMG', 'DASAR_SKL', 'DASAR_ALL']
# dataType = ['AKHIRAN_ALL', 'AKHIRAN_IMG', 'AKHIRAN_SKL']
# dataType = ['AWALAN_ALL', 'AWALAN_IMG', 'AWALAN_SKL']
# rnnType = ['LSTM', 'GRU']
rnnType = ['LSTM']
regzr = l1_l2()

for nE in nEpoch:
    for dT in dataType:
        for rT in rnnType:
            fL = TrainPath + dT + 'DataTrain.txt'
            fT = TestPath + dT + 'DataTest.txt'
            if nLayer == 1 and bidir == True:
                filenameResult = dT + '_e' + str(nE) + '_bi' + rT + str(nLayer) + 'Result.mat'
            if nLayer == 1 and bidir == False:
                filenameResult = dT + '_e' + str(nE) + '_' + rT + str(nLayer) + 'Result.mat'
            if nLayer == 2 and bidir == True:
                filenameResult = dT + '_e' + str(nE) + '_bi_l1l2k_' + rT + str(nLayer) + 'Result.mat'
            else:
                filenameResult = dT + '_e' + str(nE) + '_' + rT + str(nLayer) + 'Result.mat'
            print("filename_result: {}".format(filenameResult))

            print('fL: ', fL)
            print('fT: ', fT)

            (X_train, Y_train) = parse_corpus(fileName=fL)
            (X_test, Y_test) = parse_corpus(fileName=fT)

            print('X train shape: {} X train ndim: {} '.format(X_train.shape,X_train.ndim))
            #X_train = np.reshape(X_train,newshape=)
            print('Y train shape: {} Y train ndim: {} '.format(Y_train.shape,Y_train.ndim))

            print('X test shape: {} X test ndim: {} '.format(X_test.shape,X_test.ndim))
            #X_test = np.reshape(X_test,newshape=)
            print('Y test shape: {} Y test ndim: {} '.format(Y_test.shape,Y_test.ndim))

            # num_timesteps = X_test.shape[1]
            # num_features = X_test.shape[2]
            # num_labels = Y_test.shape[1]

            #############################################
            (numInstances, numTimestep, numFeatures) = X_train.shape
            (_, numClasses) = Y_train.shape

            #don't use shape=X_train.shape. Keras pads the input dim if you have return_sequences=True.
            rnn_input = Input(shape=(numTimestep, numFeatures))

            # satu layer
            # model.add(LSTM(output_dim=64, input_dim=numFeatures, activation='sigmoid', inner_activation='hard_sigmoid'))

            if rT == 'GRU':
                if nLayer == 1:
                    print 'L1biGRU'
                    x = Bidirectional(GRU(units=128, activation='sigmoid',recurrent_activation='hard_sigmoid',
                                          kernel_regularizer=regzr, recurrent_regularizer=None,bias_regularizer=None,activity_regularizer=None))(rnn_input)

                    # kalau mau dua layer GRU
                if nLayer == 2 and bidir == False:
                    print 'L2GRU' #either way the final layer coming out of the rnn block is called 'x'
                    x1 = GRU(units=128, return_sequences=True, activation='sigmoid',recurrent_activation='hard_sigmoid',
                             kernel_regularizer=regzr, recurrent_regularizer=None,bias_regularizer=None,activity_regularizer=None)(rnn_input)
                    x = GRU(units=128, activation='sigmoid', recurrent_activation='hard_sigmoid',kernel_regularizer=regzr,
                            recurrent_regularizer=None,bias_regularizer=None,activity_regularizer=None)(x1)
                if nLayer == 2 and bidir == True:
                    print 'L2GRU' #either way the final layer coming out of the rnn block is called 'x'
                    x1 = Bidirectional(GRU(units=128, return_sequences=True, activation='sigmoid',recurrent_activation='hard_sigmoid'
                                           ,kernel_regularizer=regzr, recurrent_regularizer=None,bias_regularizer=None,activity_regularizer=None))(rnn_input)
                    x = Bidirectional(GRU(units=128, activation='sigmoid', recurrent_activation='hard_sigmoid',
                                          kernel_regularizer=regzr, recurrent_regularizer=None,bias_regularizer=None,activity_regularizer=None))(x1)

            if rT == 'LSTM':
                if nLayer == 1:
                    print 'L1biLSTM'
                    x = Bidirectional(LSTM(units=64, activation='sigmoid',recurrent_activation='hard_sigmoid',
                                           kernel_regularizer=regzr, recurrent_regularizer=None,bias_regularizer=None,activity_regularizer=None))(rnn_input)

                    # kalau mau dua layer LSTM
                if nLayer == 2 and bidir == False:
                    print 'L2LSTM'
                    x1 = LSTM(units=64, return_sequences=True, activation='sigmoid',recurrent_activation='hard_sigmoid',
                              name='LSTM_0',kernel_regularizer=regzr, recurrent_regularizer=None,bias_regularizer=None,
                              activity_regularizer=None)(rnn_input)
                    x = LSTM(units=64, activation='sigmoid', recurrent_activation='hard_sigmoid',name='LSTM_1',
                             kernel_regularizer=regzr, recurrent_regularizer=None,bias_regularizer=None,
                             activity_regularizer=None)(x1)
                if nLayer == 2 and bidir == True:
                    print 'L2LSTM'
                    x1 = Bidirectional(LSTM(units=64, return_sequences=True, activation='sigmoid',
                                            recurrent_activation='hard_sigmoid', name='LSTM_0',kernel_regularizer=regzr,
                                            recurrent_regularizer=None,bias_regularizer=None,activity_regularizer=None))(rnn_input)
                    x = Bidirectional(LSTM(units=64, activation='sigmoid', recurrent_activation='hard_sigmoid',name='LSTM_1',
                                           kernel_regularizer=regzr, recurrent_regularizer=None,bias_regularizer=None,
                                           activity_regularizer=None))(x1)
            final_layer = Dense(numClasses,activation='softmax')(x)
            model = Model(inputs=rnn_input,outputs=final_layer)

            model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['categorical_accuracy','cosine']) #kld doesn't work 2.0.6

            #4000 epoch, silakan diganti-ganti

            # TODO: Ganti parameter, misal epoch = 1000
            start_time_train = time.time()
            model.fit(X_train, Y_train, batch_size=16, epochs=nE,verbose=0)
            elapsed_time_train = time.time() - start_time_train
            print("model metrics names: {}".format(model.metrics_names))

            score_train = model.evaluate(x = X_train, y = Y_train, batch_size=16, verbose = 1)
            print("accuracy di training data : " + str(score_train))

            score_testing = model.evaluate(x = X_test, y = Y_test, batch_size=16, verbose = 1)
            print("accuracy di testing data : " + str(score_testing))

            start_time_test = time.time()
            predict_result = model.predict(X_test, batch_size=16,verbose=0)
            elapsed_time_test = time.time() - start_time_test

            sio.savemat(file_name=filenameResult, mdict=dict(score_train=score_train, score_testing=score_testing,
                                                             predict_result=predict_result,
                                                             eta_train=elapsed_time_train,
                                                             eta_test=elapsed_time_test))

