import numpy as np

def load_samples(filename):
    EMG=np.load(filename)['EMG']
    labels=np.load(filename)['labels']
    print('Load datas sucessfully!')
    samples=(EMG,labels)
    return samples

def mode(array):
    count = np.count_nonzero(array == 'IDLE')
    if count>=int(array.shape[0]):
        return 'IDLE'
    else:
        vals, counts = np.unique(array[array!='IDLE'], return_counts=True)
        index = np.argmax(counts)
        return vals[index]

def reshape_samples(samples,length=50):
    EMGs,labels=samples
    reshape_EMGs=np.array([EMGs[:,0:length]])
    reshape_labels=np.array([mode(labels[0:length])])
    for i in range(labels.shape[0]-int(length/10)):
        label=np.array([mode(labels[i+1:i+1+int(length/10)])])
        EMG=np.array([EMGs[:,i*10+10:i*10+10+length]])
        reshape_labels=np.concatenate((reshape_labels,label),axis=0)
        reshape_EMGs=np.concatenate((reshape_EMGs,EMG),axis=0)
    reshape_samples=(reshape_EMGs,reshape_labels)
    return reshape_samples