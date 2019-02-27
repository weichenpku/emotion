#! /usr/bin/python
# -*-coding:utf-8

import numpy
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

DATA_DIR = '../features/'
ECG_FILE = 'ecg.npy'
GSR_FILE = 'gsr.npy'
EEG_FILE = 'eeg2.npy'
PUPIL_FILE = 'pupil.data'
TYPES = 4
USERS = 28

emotion_map = {0:'neutral', 1:'joyful', 2:'sad', 3:'fearful'}
emotion_inverse_map = {'ne':0, 'jo':1, 'sa':2, 'fe':3}

pupil_fea = {} #ecg & gsr data: m*8*n, m people in people_map{}, 8 line for 4 emotions

result = {} # [confidence1, type1], [confidence2, type2], [confidence3, type3]] of each user
data_map = {} # channel: index => user

eeg_f1_result = []
eeg_classify_result = []

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
        #ne_mean = numpy.sqrt(data[i][0] * data[i][1])
        for j in range(data.shape[1]):
            data[i][j] = data[i][j] - ne_mean
            if (SIGNAL_TYPE==EEG_FILE):
                for k in range(data.shape[2]):
                    if ne_mean[k]==0:
                        data[i][j][k] = 1
                    else:
                        data[i][j][k] = data[i][j][k]/ne_mean[k]
            
    data = numpy.delete(data, [0,1], axis=1)


    data = numpy.insert(data, data.shape[2], int(0), axis=2) # add labels
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i][j][data.shape[2] - 1] = int(j / 2)+1
    
    
    for i in range(data.shape[2]-1):
        data[:,:,i] = (data[:,:,i]- numpy.min(data[:,:,i])) / (numpy.max(data[:,:,i]) - numpy.min(data[:,:,i]))
    
    return data
 
def load_features():
    print('In load_features()')
    ecg_data = parse_npy_data(ECG_FILE)
    gsr_data = parse_npy_data(GSR_FILE)
    eeg_data = parse_npy_data(EEG_FILE)
    pupil_data = parse_pupil()
    
    return ecg_data, gsr_data, eeg_data, pupil_data

def predict(train, test, epoch_num, batch_num, user):
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
    
    model.fit(X, dummy_y, epochs=epoch_num, batch_size=batch_num)
    
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

    user_valid=[[0,0] for i in range(int(len(Y1)/6))]
    for i in range(0,len(Y1),6):
        user_valid[int(i/6)]=([user[int(i/6)],result[i:i+6]==Y1[i:i+6]-1])
    
    print("Correct Results: %d/%d"%(correct, len(X1)))
    print(user_valid)

    backend.clear_session()
    return user_valid
    
def LDA_predict(xtrain,ytrain,xtest,ytest):
    clf = LinearDiscriminantAnalysis(n_components=2)
    clf.fit(xtrain,ytrain)

    xtrain_new=clf.transform(xtrain)
    xtest_new=clf.transform(xtest)
    datatrain = numpy.insert(xtrain_new, 2, int(1), axis=1) # add labels
    datatest = numpy.insert(xtest_new, 2, int(1), axis=1)
    
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
    #parse_pupil()
    ecg, gsr, eeg, pupil = load_features()
    print(ecg.shape,gsr.shape,eeg.shape,pupil.shape)
    eeg_origin=eeg
    
    apply_pca = True
    pca_comp_num = 32
    if apply_pca:
        fea_dim = eeg_origin.shape[2]-1
        shape = eeg_origin.shape
        pca_origin = eeg_origin[:,:,0:fea_dim].reshape(shape[0]*shape[1],fea_dim)
        
        pca = PCA(n_components = pca_comp_num)
        eeg_new = pca.fit_transform(pca_origin)
        
        #t=pca.get_covariance()
        eeg_pca = numpy.append(eeg_new.reshape(shape[0],shape[1],pca_comp_num),eeg_origin,axis=2)
        eeg_pca = numpy.delete(eeg_pca,range(pca_comp_num,eeg_pca.shape[2]-1),axis=2)
        eeg_origin = eeg_pca
    
    feature_able = [0 for i in range(pca_comp_num)]
    feature_del = numpy.array([])

    acc_result1=[0 for i in range(pca_comp_num)]
    acc_result2=[0 for i in range(pca_comp_num)]
    eeg_result=([[] for i in range(USERS)])
    eeg_f1_result=[0 for i in range(pca_comp_num)]
    for feaidx in range(pca_comp_num):
        
        tmp_del = numpy.append(feature_del,feaidx)
        eeg_data1 = numpy.delete(eeg_origin, feature_del, axis=2)
        eeg_data2 = numpy.delete(eeg_origin, tmp_del, axis=2)
        
        print('feature_dims', eeg_data1.shape, eeg_data2.shape)
        
        eeg_data = eeg_data1
        eeg_result=([[] for i in range(USERS)])
        eeg_classify_result = [[0 for i in range(3)] for j in range(3)]
        repeat_time = 100
        for idx in range(repeat_time):
            print('feature id:',feaidx,idx)
            eeg=eeg_data
            rd_ecg = numpy.random.permutation(ecg.shape[0])
            rd_gsr = numpy.random.permutation(gsr.shape[0])
            rd_eeg = numpy.random.permutation(eeg.shape[0])
            rd_pupil = numpy.random.permutation(pupil.shape[0])

            ecg = ecg[rd_ecg,:]
            gsr = gsr[rd_gsr,:]
            eeg = eeg[rd_eeg,:]
            pupil = pupil[rd_pupil,:]
            eeg=eeg.reshape([eeg.shape[0]*eeg.shape[1], eeg.shape[2]])            
            ret = predict(eeg[0:126,:], eeg[126:, :], 100, 5, rd_eeg)
            for i in range(len(ret)):
                id=ret[i][0]
                num=ret[i][1]
                #print(eeg_result[id])
                print(num)
                eeg_result[id] = numpy.append(eeg_result[id],num)
                
            for i in range(USERS):
                print(i,eeg_result[i])
        acc1 = (eeg_classify_result[0][0] + eeg_classify_result[1][1] + eeg_classify_result[2][2])/(4*6*repeat_time)
        acc_result1[feaidx]=acc1

        eeg_data = eeg_data2
        eeg_result=([[] for i in range(USERS)])
        eeg_classify_result = [[0 for i in range(3)] for j in range(3)]
        repeat_time = 100
        for idx in range(repeat_time):
            eeg=eeg_data
            rd_ecg = numpy.random.permutation(ecg.shape[0])
            rd_gsr = numpy.random.permutation(gsr.shape[0])
            rd_eeg = numpy.random.permutation(eeg.shape[0])
            rd_pupil = numpy.random.permutation(pupil.shape[0])

            ecg = ecg[rd_ecg,:]
            gsr = gsr[rd_gsr,:]
            eeg = eeg[rd_eeg,:]
            pupil = pupil[rd_pupil,:]
            eeg=eeg.reshape([eeg.shape[0]*eeg.shape[1], eeg.shape[2]])            
            ret = predict(eeg[0:126,:], eeg[126:, :], 100, 5, rd_eeg)
            for i in range(len(ret)):
                id=ret[i][0]
                num=ret[i][1]
                #print(eeg_result[id])
                print(num)
                eeg_result[id] = numpy.append(eeg_result[id],num)
                
            for i in range(USERS):
                print(i,eeg_result[i])
        acc2 = (eeg_classify_result[0][0] + eeg_classify_result[1][1] + eeg_classify_result[2][2])/(4*6*repeat_time)
        acc_result2[feaidx]=acc2

        print('acc',acc1,acc2)
        if acc1+0.001<acc2:
            feature_del = tmp_del
    
    
    print(acc_result1)
    print(acc_result2)
    print('del_idx',feature_del)

    #predict(pupil[0:100,:], pupil[100:, :], 500, 5)
    #predict()
    exit()
    
    '''
    [0.535, 0.5379166666666667, 0.5304166666666666, 0.53, 0.5295833333333333, 0.5383333333333333, 0.53375, 0.44416666666666665, 0.5345833333333333, 0.4725, 0.52625, 0.55625, 0.5329166666666667, 0.5154166666666666, 0.5504166666666667, 0.5495833333333333, 0.515, 0.5475, 0.5283333333333333, 0.5429166666666667, 0.5575, 0.51875, 0.53125, 0.5404166666666667, 0.5279166666666667, 0.5179166666666667, 0.5466666666666666, 0.5341666666666667, 0.545, 0.51875, 0.5295833333333333, 0.54625, 0.5291666666666667]
    '''