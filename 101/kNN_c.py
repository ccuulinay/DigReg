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

from timeit import Timer


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
    return normalize(toInt(data)), toInt(label)


def loadTrainData():
    l = []
    with open('train.csv') as f:
        lines = csv.reader(f)
        for line in lines:
            l.append(line)  # in size of 42001*785 with 1st line of fields.

    l.remove(l[0])  # remove the 1st row which it's set of fields
    l = array(l)
    label = l[:, 0]  # The first column will be labels
    data = l[:, 1:]  # The rest will be data
    return normalize(toInt(data)), toInt(label)


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
    l = []
    with open('test.csv') as f:
        lines = csv.reader(f)
        for line in lines:
            l.append(line)

    l.remove(l[0])
    data = array(l)
    return normalize(toInt(data))


def loadVerificationData():
    l = []
    with open('rf_benchmark.csv') as f:
        lines = csv.reader(f)
        for line in lines:
            l.append(line)

    l.remove(l[0])
    label = array(l)
    return toInt(label[:, 1])


def knn_classify(testVector, trainDataSet, labels, k):
    testVector = mat(testVector)
    trainDataSet = mat(trainDataSet)
    labels = mat(labels)
    trainDataSetRowSize = trainDataSet.shape[0]
    differenceMatrix = tile(testVector, (trainDataSetRowSize, 1)) - trainDataSet

    #  Get euclidean distance and give r = 2.
    r = 2.0
    distancesMatrix = ((array(differenceMatrix)**r).sum(axis=1))**(1/r)
    sortedDistancesArgMatrix = distancesMatrix.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistancesArgMatrix[i], 0]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    print sortedClassCount
    return sortedClassCount[0][0]


def saveResult(result):
    with open('result.csv', 'wb') as rf:
        writer = csv.writer(rf)
        for i in result:
            content = []
            content.append(i)
            writer.writerow(content)


def handwritingClassTest():
    trainData, trainLabel = loadTrainData()
    testData = loadTestData()
    testLabel = loadVerificationData()
    m, n = shape(testData)
    errorCount=0
    resultList = []
    print "Load data complete."
    for i in range(m):
        classifierResult = knn_classify(testData[i], trainData, trainLabel.transpose(), 5)
        print i
        resultList.append(classifierResult)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, testLabel[0,i])
        if (classifierResult != testLabel[0, i]): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(m))
    saveResult(resultList)


def test():
    tempData, tempLabel = loadTempData()
    #print tempLabel
    #print mat(tempLabel)



handwritingClassTest()




