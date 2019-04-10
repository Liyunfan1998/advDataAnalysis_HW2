# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]))
    Wx_plus_b = tf.matmul(inputs, Weights)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets("/Users/liyunfan/targetDirectory/Fashion-MNIST/data/fashion", one_hot=True)

sess = tf.InteractiveSession()
x = tf.placeholder('float', shape=[None, 784])
y_ = tf.placeholder('float', shape=[None, 10])
W=tf.Variable(tf.zeros[784,10])
b=tf.Variable(tf.zeros([10]))
sess.run(tf.global_variables_initializer())

