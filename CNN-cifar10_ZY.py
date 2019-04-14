# -*- coding:utf-8 -*-
import tensorflow as tf
import pickle
import numpy as np
#import matplotlib.pyplot as plt
import random
import os

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32,32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)#使变成行向量
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

def get_batch(batch_size, image, label):
    batch_image = list()
    batch_label = list()
    indexs = list()
    for i in range(batch_size):
        index = random.randint(0, len(image) - 1)
        while index in indexs:
            index = random.randint(0, len(image) - 1)
        d = list(image[index])
        batch_image.append(d)
        z = label[index]
        batch_label.append(z)
        indexs.append(index)
    return batch_image, batch_label

#print(Xtr.shape) #(50000, 32, 32, 3)
#print(Ytr.shape) #(50000,)
#print(Xte.shape) #(10000, 32, 32, 3)
#print(Yte.shape) #(10000,)

def weight_variable(shape):#初始化过滤器
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def l2_weight(shape, w1):#带有L2正则化的
    weight = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(weight), w1, name="weight_loss")
        tf.add_to_collection("losses", weight_loss)
    return weight
def bias_variable(shape): #始化偏置，初始化时，所有值是0.1
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
def conv2d(x,w): #卷积函数
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')
def max_pool_2x2(x): #池化函数
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#创建输入数据的placeholder
with tf.name_scope('input_holder'):
    x_image = tf.placeholder(tf.float32, [None, 32,32,3])
    y_ = tf.placeholder(tf.float32, [None])


# 第一个conv层
# 5*5的卷积核大小，3个channel ，32个卷积核
with tf.name_scope('conv1'):
    with tf.name_scope('weight1'):  # 权重
        weight1 = weight_variable(shape=[5, 5, 3, 32])
    with tf.name_scope('bias1'):  # 偏置
        bias1 = bias_variable([32])
    conv1 = tf.nn.relu(conv2d(x_image, weight1) + bias1)

# 第一个最大池化层
with tf.name_scope('pool1'):
    pool1 = max_pool_2x2(conv1)

# 第二层conv层 input: 32   size = 5*5   64个卷积核
with tf.name_scope('conv2'):
    with tf.name_scope('weight2'):
        weight2 = weight_variable(shape=[5, 5, 32, 64])
    with tf.name_scope('bias2'):
        bias2 = bias_variable([64])
    conv2 = tf.nn.relu(conv2d(pool1, weight2) + bias2)

# 第二个最大池化层
with tf.name_scope('pool2'):
    pool2 = max_pool_2x2(conv2)

# 全连接网络
with tf.name_scope('f-c-n'):
    with tf.name_scope('weight3'):
        weight3 = l2_weight(shape=[8* 8 * 64, 1024], w1=0.001)
        #weight3 = weight_variable(shape=[8* 8 * 64, 1024])
    with tf.name_scope('bias3'):
        bias3 = bias_variable([1024])
    with tf.name_scope('pool2_flat'):
        pool2_flat = tf.reshape(pool2, [-1, 8 *8 * 64])
    local3 = tf.nn.relu(tf.matmul(pool2_flat, weight3) + bias3)

# 为了减少过拟合，在输出层之前加入dropout
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    local3_drop = tf.nn.dropout(local3, keep_prob)

# 输出层
with tf.name_scope('inference'):
    with tf.name_scope('weight5'):
        weight4 = weight_variable(shape=[1024, 10])
    with tf.name_scope('bias5'):
        bias4 = bias_variable([10])
    logits = tf.add(tf.matmul(local3_drop, weight4), bias4)
    y_conv = tf.nn.softmax(logits)

# 计算loss function
with tf.name_scope('losses'):
    y_ = tf.cast(y_, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.add_to_collection('losses', cross_entropy_mean)
    loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    tf.summary.scalar('losses',loss)

# 使用adam优化，取代梯度下降
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

top_k_op = tf.nn.in_top_k(logits, y_, 1)


# 开启进程
sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

#保存模型
saver = tf.train.Saver()
saveFile='./CNNmodles/cifar10/CNN_cifar10.ckpt'

# 使用tensorboard ，保存至LOG文件夹
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./LOG2' , sess.graph)

# 每批次的大小
batch_size = 100
# 总共有多少个批次
n_batch = 50000 // batch_size

Xtr, Ytr, Xte, Yte=load_CIFAR10('/Users/liyunfan/targetDirectory/cifar_10/cifar-10-batches-py/')

test_a=list()
epoch_num=41
#epoch：迭代次数
for epoch in range(epoch_num):
    for batch in range(n_batch):
        #batch_ys_onehot = np.eye(10, dtype=float)[batch_labels]
        batch_images, batch_labels = get_batch(batch_size,  Xtr,Ytr)
        summary,_, cross_entropy = sess.run([merged,train_step, loss], feed_dict={x_image: batch_images, y_: batch_labels, keep_prob: 1})
        if batch % 100 == 0:
            print("epoch:" ,str(epoch) ," batch:",batch,'cross_entropy',cross_entropy)
        train_writer.add_summary(summary, epoch*50000+batch)

    #每个epoch后，计算训练出的模型在测试集上的准确率
    accuracy = sess.run([top_k_op], feed_dict={x_image: Xte, y_: Yte, keep_prob: 1})
    test_accuracy = float(np.sum(accuracy) / 10000)
    test_a.append(test_accuracy)
    print("epoch:%d  test accuracy %f" % (epoch, test_accuracy))
train_writer.close()
saver.save(sess, saveFile)

"""
plt.plot(range(epoch_num),test_a)
plt.xlabel("epoch")
plt.ylabel("test accuracy")
plt.show()
"""
