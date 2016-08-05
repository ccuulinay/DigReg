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
import csv
import tensorflow as tf
import pandas as pd
import numpy as np
from numpy import array
from numpy import shape


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


def saveResult(result, filename):
    with open(filename, 'wb') as rf:
        writer = csv.writer(rf)
        for i in result:
            content = []
            content.append(i)
            writer.writerow(content)


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
#mnist = input_data.read_data_sets(FLAGS.data_dir)
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# Create the Model
# For CNN first layer

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])



x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# For CNN second layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# For Densely Connected Layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# For the dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)



train_data, train_label = loadTrainData()
test_data = loadTestData()





# Define loss and optimizer


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

for i in range(1000):
  batch_begin = i * 40
  batch_end = (i + 1) * 40
  if i%40 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:train_data[batch_begin:batch_end], y_: train_label[batch_begin:batch_end], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: train_data[batch_begin:batch_end], y_: train_label[batch_begin:batch_end], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

prediction = sess.run(tf.argmax(y_conv, 1), feed_dict={x: test_data, keep_prob:1.0})

saveResult(prediction, 'tensorflow_cnn.csv')

"""
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
"""

