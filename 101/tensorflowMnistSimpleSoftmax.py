from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


__author__ = 'ccuulinay'

"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import pandas as pd
import numpy as np
from numpy import array
from numpy import shape
import csv


def loadTrainData():
    df = pd.read_csv('train.csv')
    l = df.as_matrix()

    l = array(l)
    label = l[:, 0]  # The first column will be labels
    label = np.eye(10)[label]
    data = l[:, 1:]  # The rest will be data
    return normalize(data), label


def normalize(input_array):
    #  To turn to a two value array with only 0 or 1
    #  For giving number, if 0 then 0, else then 1
    """
    m, n = shape(array)
    for i in xrange(m):
        for j in xrange(n):
            if array[i, j] != 0:
                array[i, j] = 1

    m, n = shape(array)
    for i in xrange(m):
        for j in xrange(n):
            array[i, j] = array[i, j] / 255.0

    """
    input_array = np.multiply(input_array, 1.0 / 255.0)
    return input_array


def loadTestData():
    df = pd.read_csv('test.csv')
    l = df.as_matrix()

    data = array(l)
    return normalize(data)


def loadVerificationData():
    df = pd.read_csv('rf_benchmark.csv')
    l = df.as_matrix()
    label = array(l)
    label = np.eye(10)[label[:, 1]]
    return label


def saveResult(result, filename):
    with open(filename, 'wb') as rf:
        writer = csv.writer(rf)
        for i in result:
            content = []
            content.append(i)
            writer.writerow(content)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



train_data, train_label = loadTrainData()
test_data = loadTestData()


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
#mnist = input_data.read_data_sets(FLAGS.data_dir)
sess = tf.InteractiveSession()

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.7).minimize(cross_entropy)

# Train
tf.initialize_all_variables().run()
"""
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  train_step.run({x: batch_xs, y_: batch_ys})
"""

for i in range(1000):
    batch_begin = i * 40
    batch_end = (i + 1) * 40
    train_step.run({x: train_data[batch_begin:batch_end], y_: train_label[batch_begin:batch_end]})

#train_step.run({x:train_data, y_:train_label})


# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
prediction = sess.run(tf.argmax(y, 1), feed_dict={x: mnist.test.images[:100]})
saveResult(prediction, 'tensorflow_softmax.csv')

print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
#print(accuracy.eval({x: train_data[:100], y_: train_label[:100]}))


