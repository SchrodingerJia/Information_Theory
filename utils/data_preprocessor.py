from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def preparing(samples,test_size=0.2):
    X,y=samples
    label_set=set(y)
    label_dic={}
    for i in range(len(label_set)):
        label_dic[list(label_set)[i]]=i
    for j in range(len(y)):
        y[j]=label_dic[y[j]]
    y_categorical = to_categorical(y)  # one-hot 编码标签
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=test_size, random_state=28)
    return X_train, X_test, y_train, y_test, label_dic