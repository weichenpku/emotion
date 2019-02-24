#! /usr/bin/python
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

DATA_DIR = '../features/'
ECG_FILE = 'ecg.npy'
GSR_FILE = 'gsr.npy'
EEG_FILE = 'eeg.npy'
PUPIL_FILE = 'pupil.data'
TYPES = 4
USERS = 28

emotion_map = {0:'neutral', 1:'joyful', 2:'sad', 3:'fearful'}
emotion_inverse_map = {'ne':0, 'jo':1, 'sa':2, 'fe':3}

pupil_fea = {} #ecg & gsr data: m*8*n, m people in people_map{}, 8 line for 4 emotions

result = {} # [confidence1, type1], [confidence2, type2], [confidence3, type3]] of each user
data_map = {} # channel: index => user

def parse_pupil(): # n * (6 + label)
    print("In parse_pupil()")
    pupil_data = numpy.array([])
    with open(DATA_DIR+PUPIL_FILE, 'r') as f:
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
    result[SIGNAL_TYPE] = [[[-1.0,-1.0] for i in range(3)] for j in range(USERS)]
    data_map[SIGNAL_TYPE] = []
    del_list = []

    data = numpy.load(DATA_DIR+SIGNAL_TYPE)
    for i in range(data.shape[0]):
        if numpy.max(numpy.abs(data[i]))>0:
            result[SIGNAL_TYPE][i] = [[0,0] for i in range(3)] 
            data_map[SIGNAL_TYPE].append(i)
        else:
            del_list.append(i)

    data = numpy.delete(data, del_list, axis=0)

    for i in range(data.shape[0]): # substract each person's neutral values
        ne_mean = (data[i][0] + data[i][1]) / 2.0
        for j in range(data.shape[1]):
            data[i][j] = data[i][j] - ne_mean
    
    data = numpy.delete(data, [0,1], axis=1)

    data = numpy.insert(data, data.shape[2], int(0), axis=2) # add labels
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i][j][data.shape[2] - 1] = int(j / 2)
    
    

    data = data.reshape(data.shape[0]*data.shape[1], data.shape[2])
    
    # normalization
    for i in range(data.shape[1] - 1):
        #data[:,i] = (data[:,i] - numpy.mean(data[:,i])) / numpy.std(data[:,i])
        data[:,i] = (data[:,i] - numpy.min(data[:,i])) / (numpy.max(data[:,i]) - numpy.min(data[:,i]))
        
    return data
 
def load_features():
    print('In load_features()')
    ecg_data = parse_npy_data(ECG_FILE)
    gsr_data = parse_npy_data(GSR_FILE)
    eeg_data = parse_npy_data(EEG_FILE)
    pupil_data = parse_pupil()
    
    return ecg_data, gsr_data, eeg_data, pupil_data

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
        if int(result[i]) == int(Y1[i]):
            correct += 1
    
    print("Correct Results: %d/%d"%(correct, len(X1)))

if __name__ == "__main__":
    #parse_pupil()
    ecg, gsr, eeg, pupil = load_features()
    print(ecg.shape,gsr.shape,eeg.shape)

    rd_ecg = numpy.random.permutation(ecg.shape[0])
    rd_gsr = numpy.random.permutation(gsr.shape[0])
    rd_eeg = numpy.random.permutation(eeg.shape[0])
    rd_pupil = numpy.random.permutation(pupil.shape[0])
    
    ecg = ecg[rd_ecg,:]
    gsr = gsr[rd_gsr,:]
    eeg = eeg[rd_eeg,:]
    pupil = pupil[rd_pupil,:]
    
    #predict(ecg[0:130,:], ecg[130:, :], 1000, 5)
    #predict(gsr[0:130,:], gsr[130:, :], 1000, 5)
    predict(eeg[0:130,:], eeg[130:, :], 200, 5)
    #predict(pupil[0:100,:], pupil[100:, :], 500, 5)
    #predict()
    exit()
    