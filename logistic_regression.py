__author__ = 'ccuulinay'

import operator
from os import listdir
from numpy import zeros
from numpy import ones
from numpy import exp
from numpy import mat
from numpy import shape


# Load data and return two array
def loadData(fileroot):
    trainSetsList = listdir(fileroot)
    m = len(trainSetsList)
    dataArray = zeros((m, 1024))
    labelArray = zeros((m, 1))
    for i in range(m):
        returnArray = zeros((1, 1024))
        filename = trainSetsList[i]
        f = open('%s/%s' %(fileroot, filename))
        for j in range(32):
            line = f.readline()
            for k in range(32):
                returnArray[0,32*j+k] = int(line[k])
        dataArray[i,:]=returnArray

        label = filename.split('_')[0]
        labelArray[i] = int(label)
    return dataArray, labelArray


def sigmoid(inputX):
    return 1.0/(1+exp(-inputX))


def gradAscent(dataArray, labelArray, alpha, maxCycles):
    dataMatrix = mat(dataArray)
    labelMatrix = mat(labelArray)
    m, n = shape(dataMatrix)
    weigh = ones((n, 1))
    for i in range(maxCycles):
        h = sigmoid(dataMatrix*weigh)
        error = labelMatrix - h
        weigh = weigh + alpha*dataMatrix.transpose()*error
    return weigh


def classfy(testdir, weigh):
    dataArray, labelArray = loadData(testdir)
    dataMatrix = mat(dataArray)
    labelMatrix = mat(labelArray)
    h = sigmoid(dataMatrix*weigh)
    m = len(h)
    error = 0.0
    for i in range(m):
        if int(h[i]) > 0.5:
            print int(labelMatrix[i]), 'is classified as: 1'
            if int(labelMatrix[i]) != 1:
                error+=1
                print 'error'
        else:
            print int(labelMatrix[i]), 'is classified as: 0'
            if int(labelMatrix[i]) != 0:
                error+=1
                print 'error'
    print 'error rate is: ', '%.4f' %(error/m)


def digitRecognition(traindir, testdir, alpha=0.07, maxCycles=10):
    data, label = loadData(traindir)
    weigh = gradAscent(data, label, alpha, maxCycles)
    classfy(testdir, weigh)


digitRecognition("../FO_MachineLearning-master/logistic regression/use Python and NumPy/train",
                 "../FO_MachineLearning-master/logistic regression/use Python and NumPy/test",
                 )

