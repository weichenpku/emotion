import numpy as np

path='./'#'black_data/'
classify_matrix = np.load(path+'classify_result.npy')
precise=np.array([0.0 for i in range(3)])
recall=np.array([0.0 for i in range(3)])
f1=np.array([0.0 for i in range(3)])
for i in range(3):
    tp = classify_matrix[i][i]
    fp = np.sum(classify_matrix[:,i]) - classify_matrix[i][i]
    fn = np.sum(classify_matrix[i,:]) - classify_matrix[i][i] 
    print(tp,fp,fn)
    precise[i] = tp*1.0/(tp+fp)
    recall[i] = tp*1.0/(tp+fn)
    f1[i] = 2*precise[i]*recall[i]/(precise[i]+recall[i])
print(precise)
print(recall)
print(f1)



eeg_result = np.load(path+'eeg_result.npy')
print(eeg_result)
usernum = eeg_result.shape[0]
accuracy_single = [0 for i in range(usernum)]
num=0
acc70=0

precise_single=np.array([np.array([0.0 for i in range(3)]) for j in range(usernum)])
recall_single=np.array([np.array([0.0 for i in range(3)]) for j in range(usernum)])
f1_single=np.array([np.array([0.0 for i in range(3)]) for j in range(usernum)])
for i in range(usernum):
    accuracy_single[i] = (eeg_result[i][0][0]+eeg_result[i][1][1]+eeg_result[i][2][2])*1.0/np.sum(np.sum(eeg_result[i]))
    print(accuracy_single[i])
    if (accuracy_single[i]<0.4):
        num=num+1
        acc70+=accuracy_single[i]
    for j in range(3):
        tp = eeg_result[i][j][j]
        fp = np.sum(eeg_result[i][:,j]) - eeg_result[i][j][j]
        fn = np.sum(eeg_result[i][j,:]) - eeg_result[i][j][j]
        # print(tp,fp,fn)
        if tp==0:
            precise_single[i][j] = 0
            recall_single[i][j] = 0
            f1_single[i][j] = 0
        else:
            precise_single[i][j] = tp*1.0/(tp+fp)
            recall_single[i][j] = tp*1.0/(tp+fn)
            f1_single[i][j] = 2*precise_single[i][j]*recall_single[i][j]/(precise_single[i][j]+recall_single[i][j])
    #print(i,precise_single[i],recall_single[i],f1_single[i])

print(f1_single)
print(np.mean(accuracy_single))
'''
print(num)
print(acc70/num)
'''
 
