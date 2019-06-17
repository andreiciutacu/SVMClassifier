import pandas as pd
import os
import numpy as np
from sklearn import svm
from sklearn import preprocessing
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import csv


#Normalizing data methods from the laboratory
def normalize_data(test_data,type):
    if type == 'standard':
        scaler = preprocessing.StandardScaler()
        scaler.fit(test_data)
        scaled_test_data = scaler.transform(test_data)
    elif type == 'min_max':
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(test_data)
        scaled_test_data = scaler.transform(test_data)
    elif type == "l1":
        scaled_test_data = test_data/np.expand_dims(np.sum(abs(test_data),axis = 0),axis = 0)
    elif type == "l2":
        scaled_test_data = test_data/np.expand_dims(np.sqrt(np.sum(test_data**2,axis = 0)), axis = 0 )
    return scaled_test_data


#Svm classifier from laboratory
def svm_classifier(train_data,train_labels,svm_model,test_data):
    print(type(train_data))
    print(type(svm_model))
    svm_model.fit(train_data,train_labels)
    predicted_labels = svm_model.predict(test_data)
    return (predicted_labels,svm_model)


# Loading all needed paths for reading/writing
resultPath = "G:/Facultate/ProiectIA/prezic.csv"
trainPath = 'G:/Facultate/ProiectIA/train/'
testPath = 'G:/Facultate/ProiectIA/test/'
trainLabels = np.array(pd.read_csv("G:/Facultate/ProiectIA/train_labels.csv", header=0))


# Empty lists needed for data
testFiles = []
trainProcessedData = []
testProcessedData = []

for root,dirs,files in os.walk(trainPath,topdown=True):
    for x in files:
        trainImage = (np.array(pd.read_csv(trainPath + x, header=0)))

        # Increasing range until 159 so it matches the test number of features
        # We're putting NaN since we don't care what data is there so it doesn't affect at all - v1
        n = trainImage.shape[0] - 1
        medX = medY = medZ = 0
        for i in range(trainImage.shape[0]):
            medX += trainImage[i][0]
            medY += trainImage[i][1]
            medZ += trainImage[i][2]
        medX = medX / (n+1)
        medY = medY / (n+1)
        medZ = medZ / (n+1)
        for i in range(trainImage.shape[0],159):
            trainImage = np.concatenate((trainImage,[[medX, medY, medZ]]), axis = 0)

        #Transforming our data into one-dimensional 3d-array with axis table
        trainImage = pd.Series(trainImage.flatten())
        #Making it an array
        trainImage = np.array(trainImage)
        #Transforming our array into a one-dimensional array
        trainImage = trainImage.flatten()
        trainProcessedData.append((trainImage))


# Repeating same process for the test data
for root,dirs,files in os.walk(testPath,topdown=True):
    testFiles = files
    for x in files:
        testImage = (np.array(pd.read_csv(testPath + x, header=0)))
        medX = medY = medZ = 0
        for i in range(testImage.shape[0]):
            medX += testImage[i][0]
            medY += testImage[i][1]
            medZ += testImage[i][2]
        medX = medX / (n + 1)
        medY = medY / (n + 1)
        medZ = medZ / (n + 1)
        for i in range(testImage.shape[0], 159):
            testImage = np.concatenate((testImage, [[medX, medY, medZ]]), axis=0)

        testImage = pd.Series(testImage.flatten())
        testImage  = np.array(testImage)
        testImage=  testImage.flatten()
        testProcessedData.append((testImage))


#Making sure both are arrays with 3x159 elements
testProcessedData = np.array(testProcessedData)
trainProcessedData = np.array(trainProcessedData)
print(trainProcessedData.shape)
print(testProcessedData.shape)

#Checking data before normalization
plt.plot(trainProcessedData[0],'C1o')
plt.plot(trainProcessedData[1], 'C2o')
plt.plot(trainProcessedData[2],'C3o')
plt.plot(trainProcessedData[3],'C4o')
plt.show()


#Normalizing our data
trainProcessedData = normalize_data(trainProcessedData,'standard')
testProcessedData = normalize_data(testProcessedData,'standard')

#Checking data after normalization
plt.plot(trainProcessedData[0],'C1o')
plt.plot(trainProcessedData[1], 'C2o')
plt.plot(trainProcessedData[2],'C3o')
plt.plot(trainProcessedData[3],'C4o')
plt.show()


# v1 bestC = [1e-8, 1, 5, 10, 15]
# v2 bestC = [10, 12.5, 15, 20, 30, 35, 40]
# v3 bestC = [15, 17.5, 20, 25, 27]
# v4 bestC = [27, 28, 29, 30]
# v5 bestC = [26, 26.5, 27, 27.5]

bestC = [ 1, 5, 10, 15, 25, 30]
for c in bestC:
    modelSVM =svm.SVC(C=c)
    print('C =', c)
    #1st cross-validation
    predicts, modelSVM = svm_classifier(trainProcessedData[0:6000], trainLabels[0:6000,[1]].flatten(), modelSVM,
                                        trainProcessedData)
    print(predicts)
    print('Accuracy on train labels: ', metrics.accuracy_score(trainLabels[0:6000,[1]], predicts[0:6000]))
    print('Accuracy on test labels: ', metrics.accuracy_score(trainLabels[6000:9000, [1]], predicts[6000:9000]))
    #Making confusion matrix
    confusionMatrix = metrics.confusion_matrix(trainLabels[0:6000,[1]],predicts[0:6000])
    print(confusionMatrix)

    #2nd cross-validation
    predicts, modelSVM = svm_classifier(trainProcessedData[3000:9000], trainLabels[3000:9000,[1]].flatten(), modelSVM,
                                         trainProcessedData)
    print(predicts)
    print('Accuracy on train labels: ', metrics.accuracy_score(trainLabels[3000:9000, [1]], predicts[3000:9000]))
    print('Accuracy on test labels: ', metrics.accuracy_score(trainLabels[0:3000, [1]], predicts[0:3000]))
    confusionMatrix = metrics.confusion_matrix(trainLabels[0:6000,[1]], predicts[0:6000])
    print(confusionMatrix)


#Getting our predicted labels for the test data
predictedLabels = modelSVM.predict(testProcessedData)


predictFileNames = []
for fileName in testFiles:
    predictFileNames.append(fileName.replace('.csv',''))

#Opening or creating the result file
with open(resultPath, mode='w', newline='') as result:
    writer = csv.writer(result, delimiter=',')
    writer.writerow(['id', 'class'])
    for i in range(len(predictedLabels)):
        writer.writerow([predictFileNames[i],predictedLabels[i]])

print("Done")