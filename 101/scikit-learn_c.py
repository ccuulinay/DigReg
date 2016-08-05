__author__ = 'ccuulinay'

import numpy as np
import operator
import csv
from os import listdir
from numpy import zeros
from numpy import tile
from numpy import array
from numpy import mat
from numpy import shape
from numpy import ravel

import pandas as pd

from timeit import Timer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB


def loadTempDataWithPandas():
    df = pd.read_csv('temp.csv')
    l = df.as_matrix()
    l = array(l)
    label = l[:, 0]  # The first column will be labels
    data = l[:, 1:]  # The rest will be data
    return normalize(toInt(data)), label



def loadTempData():
    l = []
    with open('temp.csv') as f:
        lines = csv.reader(f)
        for line in lines:
            l.append(line)  # in size of 42001*785 with 1st line of fields.

    l.remove(l[0])  # remove the 1st row which it's set of fields
    l = array(l)
    label = l[:, 0]  # The first column will be labels
    data = l[:, 1:]  # The rest will be data
    return normalize(toInt(data)), label


def loadTrainData():
    df = pd.read_csv('train.csv')
    l = df.as_matrix()
    """
    l = []
    with open('train.csv') as f:
        lines = csv.reader(f)
        for line in lines:
            l.append(line)  # in size of 42001*785 with 1st line of fields.

    l.remove(l[0])  # remove the 1st row which it's set of fields
    """
    l = array(l)
    label = l[:, 0]  # The first column will be labels
    data = l[:, 1:]  # The rest will be data
    return normalize(data), label


def toInt(array):
    array = mat(array)  # turn array to matrix
    m, n = shape(array)  # get row# as m and column# as n
    newArray = zeros((m, n))
    for i in xrange(m):
        for j in xrange(n):
            newArray[i, j] = int(array[i, j])
    return newArray


def normalize(array):
    #  To turn to a two value array with only 0 or 1
    #  For giving number, if 0 then 0, else then 1
    m, n = shape(array)
    for i in xrange(m):
        for j in xrange(n):
            if array[i, j]!= 0:
                array[i, j] = 1
    return array


def loadTestData():
    df = pd.read_csv('test.csv')
    l = df.as_matrix()
    """
    l = []
    with open('test.csv') as f:
        lines = csv.reader(f)
        for line in lines:
            l.append(line)

    l.remove(l[0])
    """
    data = array(l)
    return normalize(data)


def loadVerificationData():
    df = pd.read_csv('rf_benchmark.csv')
    l = df.as_matrix()
    label = array(l)
    return toInt(label[:, 1])


def saveResult(result, filename):
    with open(filename, 'wb') as rf:
        writer = csv.writer(rf)
        for i in result:
            content = []
            content.append(i)
            writer.writerow(content)


def knnClassify(trainData, trainLabel, testData):
    knnClassifier = KNeighborsClassifier()
    knnClassifier.fit(trainData, ravel(trainLabel))
    print 'knn predicting.'
    testLabel = knnClassifier.predict(testData)
    saveResult(testLabel, 'sklearn_knn_Result.csv')
    return testLabel


def svcClassify(trainData, trainLabel, testData):
    svcClassifier = svm.SVC(C=5.0)
    svcClassifier.fit(trainData, ravel(trainLabel))
    print 'svc predicting.'
    testLabel = svcClassifier.predict(testData)
    saveResult(testLabel, 'sklearn_SVC_C=5.0_Result.csv')
    return testLabel


def GaussianNBClassify(trainData,trainLabel,testData):
    nbClf=GaussianNB()
    nbClf.fit(trainData,ravel(trainLabel))
    print 'Gaussian NB predicting.'
    testLabel=nbClf.predict(testData)
    saveResult(testLabel,'sklearn_GaussianNB_Result.csv')
    return testLabel


def MultinomialNBClassify(trainData,trainLabel,testData):
    nbClf=MultinomialNB(alpha=0.1)      #default alpha=1.0,Setting alpha = 1 is called Laplace smoothing, while alpha < 1 is called Lidstone smoothing.
    nbClf.fit(trainData,ravel(trainLabel))
    testLabel=nbClf.predict(testData)
    saveResult(testLabel,'sklearn_MultinomialNB_alpha=0.1_Result.csv')
    return testLabel


def RandomForestClassify(trainData, trainLabel, testData):
    rfClassifier = RandomForestClassifier(n_estimators=10)
    rfClassifier.fit(trainData, ravel(trainLabel))
    print 'Random Forest predicting.'
    testLabel = rfClassifier.predict(testData)
    saveResult(testLabel, 'sklearn_RandomForest_Result.csv')


def digitRecognition():
    trainData, trainLabel = loadTrainData()
    testData = loadTestData()

    print 'Load data completed.'

    # result1=knnClassify(trainData,trainLabel,testData)
    # result2=svcClassify(trainData,trainLabel,testData)
    # result3=GaussianNBClassify(trainData,trainLabel,testData)
    result4=RandomForestClassify(trainData,trainLabel,testData)

    resultGiven = loadVerificationData()
    m, n = shape(testData)
    differenceCount = 0
    for i in xrange(m):
        if result4[i] != resultGiven[0, i]:
            differenceCount += 1
    print differenceCount


digitRecognition()
