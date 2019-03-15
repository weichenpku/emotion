#! /usr/bin/python
# -*-coding:utf-8

import numpy as np
import pandas
import gc
from keras.models import Sequential
from keras.layers import Dense
from keras import backend
#from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import scipy
import scipy.io as sio

DATA_DIR = './'
EEG_FILE = 'eeg_norm2.npy'
TYPES = 4
USERS = 46

emotion_map = {0:'neutral', 1:'joyful', 2:'sad', 3:'fearful'}
emotion_inverse_map = {'ne':0, 'jo':1, 'sa':2, 'fe':3}

#flist = [1, 2, 4, 5, 6, 8, 10, 11, 12, 18, 19, 25, 27, 28, 30, 33, 36, 37, 38, 39, 41, 43, 45, 46, 48, 49]
#mlist = [3, 7, 9, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 26, 29, 31, 32, 34, 35, 40, 42, 44, 47]
female_idx = [1, 2, 3, 5, 7, 8, 14, 15, 21, 23, 24, 26, 29, 32, 33, 34, 35, 37, 39, 41, 42, 44, 45 ]
male_idx = [0, 4, 6, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 22, 25, 27, 28, 30, 31, 36, 38, 40, 43]
female_num = 23
male_num = 23
#black_idx = [0, 5, 13, 16, 25]
black_idx = [13,16,25,36]  #[5,6,13,16,25,29,36]  #
white_num = 46-len(black_idx)

data_map = {} # channel: index => user

eeg_f1_result = []
eeg_classify_result = []

flag_no_male = False # True
flag_no_female = False #True
flag_black = True
flag_use_loo = False
user_del_list = []

repeat_time = 10

def parse_npy_data(SIGNAL_TYPE):
    print('Parsing %s...'%SIGNAL_TYPE)
    data_map[SIGNAL_TYPE] = []

    data = np.load(DATA_DIR+SIGNAL_TYPE)
    data = np.delete(data, user_del_list, axis=0)
    print(data.shape)

    data = np.insert(data, data.shape[2], int(0), axis=2) # add labels
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i][j][data.shape[2] - 1] = int(j / 2)+1
    
    '''
    for i in range(data.shape[2]-1):
        data[:,:,i] = (data[:,:,i]- np.min(data[:,:,i])) / (np.max(data[:,:,i]) - np.min(data[:,:,i]))
    '''
    return data
 
def load_features():
    print('In load_features()')
    ecg_data = np.array([]) #parse_npy_data(ECG_FILE)
    gsr_data = np.array([]) #parse_npy_data(GSR_FILE)
    eeg_data = parse_npy_data(EEG_FILE)
    pupil_data = np.array([]) #parse_pupil()
    
    return ecg_data, gsr_data, eeg_data, pupil_data

def predict(train, test, epoch_num, batch_num, user):
    #print('start predicting')
    #print(train.shape)
    #print(test.shape)
    #print(user.shape)
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
    #model.add(Dense(16, activation='relu'))
    #model.add(Dense(8, activation='relu')) 
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.fit(X, dummy_y, epochs=epoch_num, batch_size=batch_num, verbose=0)
    
    #test
    dim_test = test.shape
    X1 = test[:,0:dim_test[1] - 1].astype(float)
    Y1 = test[:,dim_test[1] - 1]
    print(X1.shape)
    print(Y1.shape)
    
    result = model.predict_classes(X1)
    result_p = model.predict(X1)
    correct = 0
    for i in range(len(Y1)):
        eeg_classify_result[int(Y1[i] - 1)][int(result[i])] = eeg_classify_result[int(Y1[i] - 1)][int(result[i])] + 1   
        print(result[i]+1, ' ', Y1[i], ' ',user[int(i/6)], ' ', result_p[i])
        if int(result[i]) == int(Y1[i] - 1):
            correct += 1

    user_valid=[[0,0,0] for i in range(len(Y1))]
    for i in range(len(Y1)):
        user_valid[i]=[user[int(i/6)],int(Y1[i]-1),int(result[i])]
    
    print("Correct Results: %d/%d"%(correct, len(X1)))
    #print(user_valid)

    backend.clear_session()
    return user_valid
    
def LDA_predict(xtrain,ytrain,xtest,ytest):
    clf = LinearDiscriminantAnalysis(n_components=2)
    clf.fit(xtrain,ytrain)

    xtrain_new=clf.transform(xtrain)
    xtest_new=clf.transform(xtest)
    datatrain = np.insert(xtrain_new, 2, int(1), axis=1) # add labels
    datatest = np.insert(xtest_new, 2, int(1), axis=1)
    
    for i in range(datatrain.shape[0]):
        datatrain[i][2]=ytrain[i]
    for i in range(datatest.shape[0]):
        datatest[i][2]=ytest[i]    

    print(datatrain)
    print(datatest)
    predict(datatrain, datatest, 50, 5)
    '''
    ypredict=clf.predict(xtest)
    
    correct = 0
    for i in range(len(ytest)):
        print(int(ypredict[i]), ' ', int(ytest[i]))
        if int(ypredict[i]) == int(ytest[i]):
            correct += 1
    
    print("Correct Results: %d/%d"%(correct, len(ytest)))
    '''

if __name__ == "__main__":

    if (flag_no_female):
        user_del_list = female_idx
        USERS = male_num
    if (flag_no_male):
        user_del_list = male_idx
        USERS = female_num
    if (flag_black):
        user_del_list = black_idx
        USERS = white_num

    #parse_pupil()
    ecg, gsr, eeg, pupil = load_features()
    print(ecg.shape,gsr.shape,eeg.shape,pupil.shape)
    eeg_origin=eeg
    
    apply_pca = True
    pca_comp_num = 128
    if apply_pca:
        fea_dim = eeg_origin.shape[2]-1
        shape = eeg_origin.shape
        pca_origin = eeg_origin[:,:,0:fea_dim].reshape(shape[0]*shape[1],fea_dim)
        
        pca = PCA(n_components = pca_comp_num)
        eeg_new = pca.fit_transform(pca_origin)
        
        #t=pca.get_covariance()
        eeg_pca = np.append(eeg_new.reshape(shape[0],shape[1],pca_comp_num),eeg_origin,axis=2)
        eeg_pca = np.delete(eeg_pca,range(pca_comp_num,eeg_pca.shape[2]-1),axis=2)
        eeg_origin = eeg_pca
    
    feature_able = [0 for i in range(pca_comp_num)]
    if flag_use_loo:
        feature_del = np.array([  0,   1,   9,  12,  13,  14,  16,  19,  21,  25,  26,  27,  28,  30,
         33,  42,  43,  45,  46,  49,  50,  53,  56,  57,  60,  61,  66,  69,
          71,  72,  73,  74,  75,  77,  79,  81,  82,  83,  86,  91,  92,  95,
          98, 101, 103, 109, 111, 113, 114, 115, 116, 117, 118, 119, 121, 123,
         126])
    else:
        feature_del = []
    
    acc_result1=[0 for i in range(repeat_time*USERS)]
    acc_result2=[0 for i in range(repeat_time*USERS)]
    std_result1=[0 for i in range(repeat_time*USERS)]
    std_result2=[0 for i in range(repeat_time*USERS)]
    if (1): #for feaidx in range(pca_comp_num):
        feaidx=1
        tmp_del = feature_del
        eeg_data1 = np.delete(eeg_origin, feature_del, axis=2)
        eeg_data2 = np.delete(eeg_origin, [], axis=2)
        
        print('feature_dims', eeg_data1.shape, eeg_data2.shape)

        eeg_data = eeg_data1
        eeg_result=([[[0 for k in range(3)] for j in range(3)] for i in range(USERS)])
        eeg_classify_result = [[0 for i in range(3)] for j in range(3)]
        for idx in range(repeat_time*USERS):
            eeg=eeg_data
            #rd_ecg = np.random.permutation(ecg.shape[0])
            #rd_gsr = np.random.permutation(gsr.shape[0])
            #rd_eeg = np.random.permutation(eeg.shape[0])
            #rd_pupil = np.random.permutation(pupil.shape[0])
            rd_eeg = [i for i in range(eeg.shape[0])]
            #print('idx',rd_eeg)
            modidx = idx % USERS
            rd_eeg[modidx] = USERS-1
            rd_eeg[USERS-1] = modidx
            #print('idx', rd_eeg) 

            #ecg = ecg[rd_ecg,:]
            #gsr = gsr[rd_gsr,:]
            eeg = eeg[rd_eeg,:]
            #pupil = pupil[rd_pupil,:]
            eeg=eeg.reshape([eeg.shape[0]*eeg.shape[1], eeg.shape[2]])         
            #testnum = int(USERS*0.1)
            #trainnum = USERS - testnum   
            trainnum = USERS - 1
            testnum = 1
            ret = predict(eeg[0:trainnum*6 ,:], eeg[trainnum*6:, :], 25, 5, rd_eeg[trainnum:])   
            #ret = predict(eeg[0:222 ,:], eeg[222:, :], 50, 5, rd_eeg)
            #ret = predict(eeg[0:126,:], eeg[126:, :], 100, 5, rd_eeg)
            for i in range(len(ret)):
                userid=ret[i][0]
                num1=ret[i][1]
                num2=ret[i][2]
                #print(eeg_result[id])
                #print(num)
                eeg_result[userid][num1][num2] = eeg_result[userid][num1][num2]+1
                
            #for i in range(USERS):
            #    print(i,eeg_result[i])
            acc1 = (eeg_classify_result[0][0] + eeg_classify_result[1][1] + eeg_classify_result[2][2])/(6*(idx+1))
            acc_result1[idx]=acc1
            std_result1[idx]=np.std(acc_result1[:idx])
            print('right_num',eeg_classify_result[0][0] + eeg_classify_result[1][1] + eeg_classify_result[2][2])
            print('total_num',6*(idx+1))
            print('acc',acc1)

        eeg_result1=eeg_result
        eeg_classify_result1=eeg_classify_result
        np.save('classify_result.npy',eeg_classify_result1)
        np.save('eeg_result.npy',eeg_result1)

        print(eeg_classify_result1)
        print(eeg_result1)

        
    exit()


    '''
    [13, 16, 25, 36]  0-41
    0.6571428571428571
    0.6547619047619049

    [5,6,13,16,25,36] 0-39
    0.6537499999999999

    [5,6,13,16,25,29,36] 0-38
    0.6628205128205129
    0.6679487179487179

    '''