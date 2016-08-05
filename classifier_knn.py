__author__ = 'ccuulinay'

import numpy as np
import operator
from os import listdir
from numpy import zeros
from numpy import tile





# For eating a 32*32 file value to a 1*1024
def img2vector(filename):
    # to have a initialize 0 value with 1 row and 1024 fields v
    returnVector = zeros((1, 1024))
    f = open(filename)
    # with training sets are in 32*32 size
    for i in range(32):
        line = f.readline()
        for j in range(32):
            returnVector[0, 32*i+j] = int(line[j])
    return returnVector


def knn_classify(testVector, trainingSets, labels, k):
    dataSetSize = trainingSets.shape[0]
    r = 2.0
    diffMatrix = tile(testVector, (dataSetSize, 1)) - trainingSets

    # count euclidean distance

    distancesMatrix = ((diffMatrix**r).sum(axis=1))**(1/r)
    sortedDistancesMatrix = distancesMatrix.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistancesMatrix[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    print sortedClassCount
    return sortedClassCount[0][0]



def handwritingClassTest():

    hwlabels = []
    trainingFileList = listdir('trainSets')[1:]
    m = len(trainingFileList)
    trainingMatrix = zeros((m, 1024))
    for i in range(m):
        fileName = trainingFileList[i]
        classNum = int(fileName.split('_')[0])
        hwlabels.append(classNum)
        trainingMatrix[i, :] = img2vector('trainSets/%s' % fileName)

    testFileList = listdir('testSets')[1:]
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileName = testFileList[i]
        classNum = int(fileName.split('_')[0])
        testVector = img2vector('testSets/%s' % fileName)
        #print testVector
        vectorClassifier = knn_classify(testVector, trainingMatrix, hwlabels, 3)
        print "the classifier came back with : %d, the real answer is: %d" % (vectorClassifier, classNum)
        if (vectorClassifier != classNum): errorCount += 1.0

    print "\n the total number of errors is: %d" % errorCount
    print "\n the total error rate is: %f" % (errorCount/float(mTest))



handwritingClassTest()