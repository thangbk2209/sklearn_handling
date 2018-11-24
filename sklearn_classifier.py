from sklearn.model_selection import train_test_split
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics, svm
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
import sys
import pickle as pk
import matplotlib.pyplot as plt 
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
    min_arr = []
    max_arr = []
    for i in range(len(x_raw_data)):
        scalei,minXi,maxXi = scaling_data(x_raw_data[i])
        normalize_x_data.append(scalei)
        min_arr.append(minXi)
        max_arr.append(maxXi)
    normalize_x_data = np.array(normalize_x_data)
    normalize_x_data = normalize_x_data.transpose()
    y_data = np.array(y_data[0])
    print (y_data)
    print (normalize_x_data[0])
    print (normalize_x_data.shape)
    print (y_data.shape) 
    # Split dataset into training set and test set
    x_train, x_test, y_train, y_test = train_test_split(normalize_x_data, y_data, test_size=0.2) 
    print (x_train.shape)
    print (x_test.shape)
    print (y_train.shape)
    print (y_test.shape)

    with open('./results/min_max.pkl','wb') as output:
        pk.dump(min_arr,output,pk.HIGHEST_PROTOCOL)
        pk.dump(max_arr,output,pk.HIGHEST_PROTOCOL)
    return x_train, x_test, y_train, y_test
def knn_model(x_data,y_data,num_neighbors):
    print ('---------------knn model--------------------')
    x_train, x_test, y_train, y_test = preprocessing_data(x_data,y_data)
    knn = KNeighborsClassifier(n_neighbors = num_neighbors)
    knn.fit(x_train, y_train)
    #Predict the response for test dataset
    y_pred = knn.predict(x_test)
    print("Accuracy KNN:",metrics.accuracy_score(y_test, y_pred))
    return knn, metrics.accuracy_score(y_test, y_pred)
def naive_bayes_model(x_data,y_data,method):
    x_train, x_test, y_train, y_test = preprocessing_data(x_data,y_data)
    if (method == 'GaussianNB'):
        gnb = GaussianNB()
        gnb.fit(
            x_train,y_train
        )
        y_pred = gnb.predict(x_test)
        print("Accuracy GaussianNB:",metrics.accuracy_score(y_test, y_pred))
        return gnb, metrics.accuracy_score(y_test, y_pred)
    elif (method == 'BernoulliNB'):
        bnb = BernoulliNB()
        bnb.fit(
            x_train,y_train
        )
        y_pred = bnb.predict(x_test)
        print("Accuracy BernoulliNB:",metrics.accuracy_score(y_test, y_pred))
        return bnb, metrics.accuracy_score(y_test, y_pred)
    else:
        mnb = MultinomialNB()
        mnb.fit(
            x_train,y_train
        )
        y_pred = mnb.predict(x_test)
        print("Accuracy MultinomialNB:",metrics.accuracy_score(y_test, y_pred))
        return mnb, metrics.accuracy_score(y_test, y_pred)
def random_forest_model(x_data,y_data,number_of_trees):
    x_train, x_test, y_train, y_test = preprocessing_data(x_data,y_data)
    clf = RandomForestClassifier(n_estimators=number_of_trees, random_state=0)  
    clf.fit(x_train, y_train)  
    y_pred = clf.predict(x_test) 
    print("Accuracy random forest:",metrics.accuracy_score(y_test, y_pred))
    return clf, metrics.accuracy_score(y_test, y_pred)
def svm_model(x_data,y_data, kernel_type):
    print ('---------svm model-------------')
    x_train, x_test, y_train, y_test = preprocessing_data(x_data,y_data)
    clf = svm.SVC(gamma='scale', kernel = kernel_type)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("Accuracy support vector machine:",metrics.accuracy_score(y_test, y_pred))
    return clf, metrics.accuracy_score(y_test, y_pred)
def mlp_model(x_data,y_data):
    x_train, x_test, y_train, y_test = preprocessing_data(x_data,y_data)
    
    clf = MLPClassifier(activation = 'tanh', learning_rate = "adaptive", max_iter = 2000 , solver='sgd', alpha=1e-5, hidden_layer_sizes=(32), random_state=1)
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    print("Accuracy mlp:",metrics.accuracy_score(y_test, y_pred))
    return clf, metrics.accuracy_score(y_test, y_pred)
def kmean_model(x_data,y_data):
    x_train, x_test, y_train, y_test = preprocessing_data(x_data,y_data)
    label = set(y_train)
    print (label)
    clf = KMeans(n_clusters = len(label))
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    print("Accuracy kmean:",metrics.accuracy_score(y_test, y_pred))
    return clf, metrics.accuracy_score(y_test, y_pred)
def save_model(model, file_name):
    pk.dump(model, open(file_name, 'wb'))
if __name__ == '__main__':
    data_file = './data/MHEALTHDATASET/data3.txt'
    colnames = ['accX','accY','accZ','gyroX','gyroY','gyroZ','labels'] 
    df = read_csv(data_file, header=None, index_col=False, names=colnames, usecols=[5,6,7,8,9,10,23], engine='python')
    scaler = MinMaxScaler(feature_range=(0, 1))
    accX = df['accX'].values
    accY = df['accY'].values
    accZ = df['accZ'].values
    print (accZ.shape)
    plt.plot(accZ) 
    plt.show()
    lol
    gyroX = df['gyroX'].values
    gyroY = df['gyroY'].values
    gyroZ = df['gyroZ'].values
    labels = df['labels'].values
    x_raw_data = [accX,accY,accZ,gyroX,gyroY,gyroZ]
    y_data = [labels]
    x_raw_data = np.asarray(x_raw_data)
    if (sys.argv[1] == 'knn'):
        num_neighbors = 3
        file_save_classifier_model = 'results/knn/knn_' +str(num_neighbors)+'_ver_' + sys.argv[2] + '.pkl'
        model, accuracy = knn_model(x_raw_data,y_data,num_neighbors)
        save_model (model, file_save_classifier_model)
    elif (sys.argv[1] == 'naive_bayes'):
        method = 'GaussianNB'
        file_save_classifier_model = 'results/naive_bayes/naive_bayes_' + method + '_ver_' + sys.argv[2] + '.pkl'
        model, accuracy = naive_bayes_model(x_raw_data, y_data, method)
        save_model (model, file_save_classifier_model)
    elif (sys.argv[1] == 'rf'):
        number_of_trees = 30
        file_save_classifier_model = 'results/random_forest/random_forest_' + str(number_of_trees) + '_ver_'  + sys.argv[2] + '.pkl'
        model, accuracy = random_forest_model(x_raw_data, y_data, number_of_trees)
        save_model (model, file_save_classifier_model)
    elif (sys.argv[1] == 'svm'):
        kernel_type = 'rbf'
        file_save_classifier_model = 'results/svm/svm_' + kernel_type + '_ver_'  + sys.argv[2] + '.pkl'
        model, accuracy = svm_model(x_raw_data, y_data, kernel_type)
        save_model (model, file_save_classifier_model)
    elif (sys.argv[1] == 'kmean'):
        file_save_classifier_model = 'results/kmean/kmean_' + '_ver_'  + sys.argv[2] + '.pkl'
        model, accuracy = kmean_model(x_raw_data, y_data)
        save_model (model, file_save_classifier_model)
    else:
        file_save_classifier_model = 'results/mlp/mlp' + '_ver_'  + sys.argv[2] + '.pkl'
        model, accuracy = mlp_model(x_raw_data, y_data)
        save_model (model, file_save_classifier_model)