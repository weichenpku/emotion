#! /usr/bin/python3
# -*-coding:utf-8

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
#from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

ECG_FILE = 'ecg'
GSR_FILE = 'gsr'
PUPIL_FILE = 'pupil.data'
TYPES = 4

emotion_map = {0:'neutral', 1:'joyful', 2:'sad', 3:'fearful'}
emotion_inverse_map = {'ne':0, 'jo':1, 'sa':2, 'fe':3}

fea_map = {}
pupil_fea = {} #ecg & gsr data: m*8*n, m people in people_map{}, 8 line for 4 emotions

def parse_pupil(): # n * (6 + label)
    print("In parse_pupil()")
    pupil_data = numpy.array([])
    with open(PUPIL_FILE, 'r') as f:
        lines = f.readlines()
        cur_emo = None
        emoNum = 0
        for i in range(len(lines)):
            line = lines[i]
            words = line.strip().split()
            if len(words) == 1:
                cur_emo = words[0]
                emoNum += 1
                print('Parsing %s'%words[0])
            else:
                pupil_fea[i - emoNum] = words[0]
                words = words[1:]
                words.append(int(emotion_inverse_map[cur_emo]))
                
                pupil_data = numpy.insert(pupil_data, len(pupil_data), values=numpy.array(words), axis=0)
                
    pupil_data = pupil_data.reshape([int(len(pupil_data) / 7), 7])
    
    for j in range(pupil_data.shape[1] - 1):
        pupil_data[:,j] = (pupil_data[:,j] - numpy.mean(pupil_data[:,j])) / numpy.std(pupil_data[:, j])
    #    pupil_data[:,j] = pupil_data[:,j] / numpy.sum(pupil_data[:,j])
    
    return pupil_data

def gen_del(num):
    res = []
    a = 0
    b = 1
    while b < num:
        res.append(a)
        res.append(b)
        a += 8
        b += 8
    
    return res

def parse_npy_data(SIGNAL_TYPE):
    print('Parsing %s...'%SIGNAL_TYPE)
    global fea_map
    
    fea_map[SIGNAL_TYPE] = {}
    idx = numpy.load(SIGNAL_TYPE + '_file.npy')
    for i in range(idx.shape[0]):
        fea_map[SIGNAL_TYPE]['y' + str(idx[i])] = i
    
    data = numpy.load(SIGNAL_TYPE + '_feature.npy')
    for i in range(data.shape[0]): # substract each person's neutral values
        ne_mean = (data[i][0] + data[i][1]) / 2.0
        for j in range(data.shape[1]):
            data[i][j] = data[i][j] - ne_mean
    
    data = numpy.insert(data, data.shape[2], int(0), axis=2) # add labels
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i][j][data.shape[2] - 1] = int(j / 2)
    
    data = data.reshape(data.shape[0]*data.shape[1], data.shape[2])
    del_list = gen_del(data.shape[0])
    data = numpy.delete(data, del_list, axis=0)
    
    for i in range(data.shape[1] - 1):
        #data[:,i] = (data[:,i] - numpy.mean(data[:,i])) / numpy.std(data[:,i])
        data[:,i] = (data[:,i] - numpy.min(data[:,i])) / (numpy.max(data[:,i]) - numpy.min(data[:,i]))
        
    return data
 
def load_features():
    print('In load_features()')
    ecg_data = parse_npy_data(ECG_FILE)
    gsr_data = parse_npy_data(GSR_FILE)
    pupil_data = parse_pupil()
    
    return ecg_data, gsr_data, pupil_data

def predict(train, test, epoch_num, batch_num):
    dims = train.shape
    X = train[:,0:dims[1] - 1].astype(float)
    Y = train[:,dims[1] - 1]
    print (X.shape)
    print (Y.shape)
    
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)
    #print(Y)
    #print(dummy_y)
    
    model = Sequential()
    model.add(Dense(32, input_shape=(dims[1] - 1,), activation='relu'))
    model.add(Dense(16, activation='relu'))
    #model.add(Dense(8, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.fit(X, dummy_y, epochs=epoch_num, batch_size=batch_num)
    
    #test
    dim_test = test.shape
    X1 = test[:,0:dim_test[1] - 1].astype(float)
    Y1 = test[:,dim_test[1] - 1]
    print(X1.shape)
    print(Y1.shape)
    
    result = model.predict_classes(X1)
    #print(result)
    correct = 0
    for i in range(len(Y1)):
        print(result[i], ' ', Y1[i])
        if int(result[i]) == int(Y1[i] - 1):
            correct += 1
    
    print("Correct Results: %d/%d"%(correct, len(X1)))

if __name__ == "__main__":
    #parse_pupil()
    ecg, gsr, pupil = load_features()
    rd_ecg = numpy.random.permutation(ecg.shape[0])
    rd_gsr = numpy.random.permutation(gsr.shape[0])
    rd_pupil = numpy.random.permutation(pupil.shape[0])
    
    ecg = ecg[rd_ecg,:]
    gsr = gsr[rd_gsr,:]
    pupil = pupil[rd_pupil,:]
    
    """print(ecg)
    print('\n')
    print(gsr)
    print('\n')
    print(pupil)"""
    predict(ecg[0:130,:], ecg[130:, :], 1000, 5)
    predict(gsr[0:130,:], gsr[130:, :], 1000, 5)
    exit()
    #predict(pupil[0:100,:], pupil[100:, :], 500, 5)
    #predict()
    