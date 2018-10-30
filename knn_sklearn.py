from sklearn.model_selection import train_test_split
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
def scaling_data(X):
    minX = np.amin(X)
    maxX = np.amax(X)
    mean = np.mean(X)
    scale = (X-minX)/(maxX-minX)
    return scale, minX, maxX
def preprocessing_data(x_raw_data,y_data):
    print ('---------------preprocessing data-------------')
    print (x_raw_data[0])
    normalize_x_data = []
    for i in range(len(x_raw_data)):
        scalei,minXi,maxXi = scaling_data(x_raw_data[i])
        normalize_x_data.append(scalei)
    normalize_x_data = np.array(normalize_x_data)
    normalize_x_data = normalize_x_data.transpose()
    y_data = np.array(y_data[0])
    print (y_data)
    print (normalize_x_data.shape)
    print (y_data.shape) 
    # Split dataset into training set and test set
    x_train, x_test, y_train, y_test = train_test_split(normalize_x_data, y_data, test_size=0.2) 
    print (x_train.shape)
    print (x_test.shape)
    print (y_train.shape)
    print (y_test.shape)
    return x_train, x_test, y_train, y_test
def knn_model(x_data,y_data,num_neighbors):
    print ('---------------knn model--------------------')
    x_train, x_test, y_train, y_test = preprocessing_data(x_data,y_data)
    knn = KNeighborsClassifier(n_neighbors = num_neighbors)
    knn.fit(x_train, y_train)
    #Predict the response for test dataset
    y_pred = knn.predict(x_test)
    print("Accuracy KNN:",metrics.accuracy_score(y_test, y_pred))
def naive_bayes_model(x_data,y_data,method):
    x_train, x_test, y_train, y_test = preprocessing_data(x_data,y_data)
    if (method = 'GaussianNB'):
        gnb = GaussianNB()
        gnb.fit(
            x_train,y_train
        )
        y_pred = gnb.predict(x_test)
        print("Accuracy GaussianNB:",metrics.accuracy_score(y_test, y_pred))
    elif (method = 'BernoulliNB'):
        bnb = BernoulliNB()
        bnb.fit(
            x_train,y_train
        )
        y_pred = bnb.predict(x_test)
        print("Accuracy BernoulliNB:",metrics.accuracy_score(y_test, y_pred))
    else:
        mnb = MultinomialNB()
        mnb.fit(
            x_train,y_train
        )
        y_pred = mnb.predict(x_test)
        print("Accuracy MultinomialNB:",metrics.accuracy_score(y_test, y_pred))
def random_forest()
if __name__ == '__main__':
    data_file = './data/MHEALTHDATASET/data3.txt'
    colnames = ['accX','accY','accZ','gyroX','gyroY','gyroZ','labels'] 
    df = read_csv(data_file, header=None, index_col=False, names=colnames, usecols=[5,6,7,8,9,10,23], engine='python')
    scaler = MinMaxScaler(feature_range=(0, 1))
    accX = df['accX'].values
    accY = df['accY'].values
    accZ = df['accZ'].values
    gyroX = df['gyroX'].values
    gyroY = df['gyroY'].values
    gyroZ = df['gyroZ'].values
    labels = df['labels'].values
    x_raw_data = [accX,accY,accZ,gyroX,gyroY,gyroZ]
    y_data = [labels]
    print (accX)
    num_neighbors = 3
    knn_model(x_raw_data,y_data,num_neighbors)
    method = 'GaussianNB'
    naive_bayes_model(x_raw_data, y_data, method)