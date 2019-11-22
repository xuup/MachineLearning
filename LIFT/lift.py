import math
import pandas as pd
import numpy as np
import scipy.io as sio

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

from sklearn.cluster import KMeans
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split


# load mat
mmat= sio.loadmat("sample_data.mat")

test_data= mmat['test_data']
test_target= mmat['test_target']
train_data= mmat['train_data']
train_target= mmat['train_target']

train_target= train_target.T
test_target= test_target.T


ratio = 0.1

def mLIFT(train_data, train_target, test_data, test_target, ratio):
    num_train, dim = train_data.shape
    num_test, num_class = test_target.shape

    P_Centers = []
    N_Centers = []

    ##### KMeans, and save the centers
    for i in range(num_class):
        print("Performing clustering:%d/%d" % (i + 1, num_class))

        p_data = train_data[train_target[:, i] == 1]
        n_data = train_data[train_target[:, i] == -1]

        k1 = int(min(math.ceil(p_data.shape[0] * ratio), math.ceil(n_data.shape[0] * ratio)))
        # print("k1= k2= %d" %k1)
        k2 = k1;

        if (k1 == 0):
            POS_C = []
            zero_kmeans = KMeans(n_clusters=min(50, num_train)).fit(train_data)
            NEG_C = zero_kmeans.cluster_centers_
        else:
            # Positive
            if (p_data.shape[0] == 1):
                POS_C = p_data
            else:
                p_kmeans = KMeans(n_clusters=k1).fit(p_data)
                POS_C = p_kmeans.cluster_centers_
            # Negative
            if (n_data.shape[0] == 1):
                NEG_C = n_data
            else:
                n_kmeans = KMeans(n_clusters=k2).fit(n_data)
                NEG_C = n_kmeans.cluster_centers_

        # Save the cluster centers
        P_Centers.append(POS_C)
        N_Centers.append(NEG_C)

    # print("The size of P_Canters is %d\n" %len(P_Centers))

    ##### Do the map and save the models
    Models = []
    for i in range(num_class):
        print("Building classifiers: :%d/%d" % (i + 1, num_class))
        centers = np.vstack((P_Centers[i], N_Centers[i]))
        num_center = centers.shape[0]
        # print(num_center)
        data = []

        if (num_center >= 5000):
            print("Too many cluster center!")
            break
        else:
            blocksize = 5000 - num_center
            num_block = int(math.ceil(num_train / blocksize))
            # print(num_block)

            mFirst = True
            for j in range(num_block - 1):
                print(j)
                low = j * blocksize
                high = (j + 1) * blocksize
                # Calculate the distance
                for k in range(num_center):
                    diff = train_data[low:high, :] - centers[k]
                    Eu_diff = np.linalg.norm(diff, axis=1)
                    if (mFirst == True):
                        mFirst = False
                        data_temp = Eu_diff
                    else:
                        data_temp = np.vstack((data_temp, Eu_diff))

            low = (num_block - 1) * blocksize
            high = num_train

            # Calculate the distance
            for j in range(num_center):
                diff = train_data[low:high, :] - centers[j]
                Eu_diff = np.linalg.norm(diff, axis=1)
                if (mFirst == True):
                    mFirst = False
                    data_temp = Eu_diff
                else:
                    data_temp = np.vstack((data_temp, Eu_diff))

            data = data_temp.T

        training_instance_matrix = data
        training_label_vector = train_target[:, i]

        model_this = SVC(C=10, probability=True).fit(training_instance_matrix, training_label_vector)
        # model_this= LogisticRegression(C= 0.03).fit(training_instance_matrix, training_label_vector)
        # model_this= DecisionTreeClassifier().fit(training_instance_matrix, training_label_vector)
        # model_this = AdaBoostClassifier(DecisionTreeClassifier(),
        # algorithm="SAMME",
        # n_estimators=50, learning_rate=0.8).fit(training_instance_matrix, training_label_vector)
        Models.append(model_this)

    ##### Predict
    for i in range(num_class):
        print("Predicting: :%d/%d" % (i + 1, num_class))
        centers = np.vstack((P_Centers[i], N_Centers[i]))
        num_center = centers.shape[0]
        # print(num_center)
        data = []

        if (num_center >= 5000):
            print("Too many cluster center!")
            break
        else:
            blocksize = 5000 - num_center
            num_block = int(math.ceil(num_test / blocksize))
            # print(num_block)

            mFirst = True
            for j in range(num_block - 1):
                print(j)
                low = j * blocksize
                high = (j + 1) * blocksize
                # Calculate the distance
                for k in range(num_center):
                    diff = test_data[low:high, :] - centers[k]
                    Eu_diff = np.linalg.norm(diff, axis=1)
                    if (mFirst == True):
                        mFirst = False
                        data_temp = Eu_diff
                    else:
                        data_temp = np.vstack((data_temp, Eu_diff))

            low = (num_block - 1) * blocksize
            high = num_train

            # Calculate the distance
            for j in range(num_center):
                diff = test_data[low:high, :] - centers[j]
                Eu_diff = np.linalg.norm(diff, axis=1)
                if (mFirst == True):
                    mFirst = False
                    data_temp = Eu_diff
                else:
                    data_temp = np.vstack((data_temp, Eu_diff))

            data = data_temp.T
            # print(data.shape)

        testing_instance_matrix = data;
        testing_label_vector = test_target[:, i]

        predicted_label = Models[i].predict(testing_instance_matrix)

        # print (predicted_label)

        print("The accuracy is: %f" % accuracy_score(testing_label_vector, predicted_label))
        # print(roc_auc_score(testing_label_vector, predicted_label))

    return 1

if __name__ == '__main__':
    mLIFT(train_data, train_target, test_data, test_target, ratio);





