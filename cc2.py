#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import tensorflow as tf
import numpy as np
import  matplotlib.pyplot as plt
# cifa = input_data.read_data_sets("../cifar_10", one_hot=True)
data_dir = '../cifar_10/cifar-10-batches-py'


# In[2]:


def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f,encoding='latin1')
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32,32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y
def load_CIFAR10(ROOT):
  """ load all of cifar """
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


# In[3]:


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


# In[4]:


import sys, os, pickle
# sys.path.append("..")
# sys.path.append("./tutorials/")
batch_size = 128
XtrAll, YtrAll, XteAll, YteAll = load_CIFAR10(data_dir)


fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.scatter(XtrAll,YtrAll)

ax2 = fig.add_subplot(1,1,2)
ax2.scatter(XteAll,YteAll)

ax1.xlabel("trainX")
ax1.ylabel("trainY")

ax2.xlabel("testX")
ax2.ylabel("testY")

plt.show()
# In[6]:


def oneHotY(Y):
    Y_new=np.array([[0]*10]*len(Y))
    for i in range(len(Y)):
        Y_new[i][Y[i]]=1
    return Y_new

# YtrAll = oneHotY(YtrAll)
# YteAll = oneHotY(YteAll)


# In[23]:


def weight_variable(shape, stddev=5e-2, wl=0):#w1是L1正则中的系数
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        # 给weight加一个L2的loss，相当于做了一个L2的正则化处理
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        # 我们使用tf.add_to_collection把weight loss统一存到一个collection，这个collection名为"losses"
        # 它会在后面计算神经网络总体loss时被用上
        tf.add_to_collection("losses", weight_loss)
    return var
"""
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
"""

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# 卷积函数的四个参数分别是训练图像，卷积核，步长，填充
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 求最大值池化，长宽缩小一半
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')


# In[24]:


def add_layer(inputs, w_shape, b_shape, activation_function=None,mul_func = tf.matmul,pool=None,wl=0,stddev=5e-2,norm=False):
#     Weights = tf.Variable(tf.random_normal([in_size, out_size]))
#     biases = tf.Variable(tf.zeros([1, out_size]))
    with tf.name_scope('WB'):
        Weights = weight_variable(w_shape,stddev=5e-2,wl=wl)
        biases = bias_variable(b_shape)

    with tf.name_scope('W_mult_X_add_B'):
        Wx_plus_b = mul_func(inputs, Weights)+biases
    with tf.name_scope('activate'):
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
    with tf.name_scope('norm'):
        if norm:
            outputs= tf.nn.lrn(outputs, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    with tf.name_scope('pool'):
        if pool is not None:
            outputs = pool(outputs)
    print ('out:',outputs.shape)
    return outputs


# In[25]:


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)#真实标签0-9
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example'
    )#交叉熵
    cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                        name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)#加到最开始定义的集合这里最后又两部分交叉熵损失函数+L2损失函数

    return tf.add_n(tf.get_collection('losses'), name='total_loss')#拿到两部分损失


# In[46]:


sess = tf.InteractiveSession()
# x = tf.placeholder('float', shape=[batch_size, 32, 32, 3])
# y_ = tf.placeholder('float', shape=[batch_size])

x = tf.placeholder(tf.float32, [None, 32,32,3])
y_ = tf.placeholder(tf.float32, [None])

sess.run(tf.global_variables_initializer())


# In[47]:

with tf.name_scope('conv1'):
    w_shape1=[3,3,3,64]
    b_shape1=[64]
    x_image = tf.reshape(x,[-1, 32,32,3])
    out1 = add_layer(x_image,w_shape=w_shape1,
                     b_shape=b_shape1,mul_func=conv2d,
                     activation_function=tf.nn.relu, pool=max_pool_2x2)
    norm1 = tf.nn.lrn(out1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)


# In[48]:

with tf.name_scope('conv2'):
    w_shape2=[3,3,64,64]
    b_shape2=[64]

    out2 = add_layer(out1,w_shape=w_shape2,
                     b_shape=b_shape2,mul_func=conv2d,
                     activation_function=tf.nn.relu, pool=max_pool_2x2,norm=True)

    out2_flat=tf.reshape(out2,[batch_size,-1]) # flatten
    ofdim = out2_flat.get_shape()[1].value
# print ofdim


# In[58]:

with tf.name_scope('conv3'):
    w_shape3=[8*8*64,1024]
    b_shape3=[1024]

    # out2_flat = tf.reshape(out2,[-1,ofdim])
    out3 = add_layer(out2_flat,w_shape=w_shape3,
                     b_shape=b_shape3,
                     activation_function=tf.nn.relu,stddev=0.04, wl=0.004)

    # 随机关闭一些神经元防止过拟
    keep_prob = tf.placeholder("float")
    out3 = tf.nn.dropout(out3, keep_prob)


# In[59]:

with tf.name_scope('conv4'):
    w_shape4 =[1024, 512]
    b_shape4 = [512]

    out4 = add_layer(out3,w_shape4,b_shape4,activation_function=tf.nn.relu, stddev=0.04, wl=0.004)


# In[60]:

with tf.name_scope('conv_softmax'):
    # 从1024个神经元映射到10个神经元
    w_shape5 =[512, 10]
    b_shape5 = [10]

    pred = add_layer(out4,w_shape5,b_shape5,tf.nn.softmax, stddev=1/512.0)

# labels = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
# b=tf.cast(labels, tf.int64)
# sess = tf.Session()
# sess.run(tf.initialize_all_variables())
# print(sess.run(b))


# In[67]:


y_=tf.cast(y_,tf.int32)
with tf.name_scope('loss'):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out4, labels=y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.add_to_collection('losses', cross_entropy_mean)
    loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    tf.summary.scalar('losses',loss)
with tf.name_scope('train'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
    Xte = tf.placeholder(tf.float32, [None, 32,32,3])
    Yte = tf.placeholder(tf.int32, [None])
# top_k_op = tf.nn.in_top_k(Xte, Yte, 1)


# In[69]:


# 计算交叉熵损失
# cross_entropy = -tf.reduce_sum(y_*tf.log(pred))
# # 创建优化器
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 计算准确率， tf.argmax函数 在 label 中找出数值最大的那个元素的下标
# correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
with tf.name_scope('classify'):
    top_k_op = tf.nn.in_top_k(out4, y_, 1)
    sess.run(tf.global_variables_initializer())


# In[20]:


import math
import random
import time
# tf.train.start_queue_runners()

# In[70]:


# sess=tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
with tf.name_scope('train'):
    #保存模型
    saver = tf.train.Saver()
    saveFile='../cifar10/CNN_cifar10_myown_usingZYmethod.ckpt'
    # 使用tensorboard ，保存至LOG文件夹
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./LOG_myown' , sess.graph)
    import random
    test_a=list()
    epoch_num=41
    n_batch = 50000 // batch_size
    #epoch：迭代次数


# In[92]:

for epoch in range(epoch_num):
    for batch in range(50000//batch_size + 1):
    # for batch in range(101):
        #batch_ys_onehot = np.eye(10, dtype=float)[batch_labels]
        batch_images, batch_labels = get_batch(batch_size,  XtrAll,YtrAll)
        batch_images=np.array(batch_images,dtype=np.float)
        batch_labels=np.array(batch_labels,dtype=np.float)
        # print 'batch_images',batch_images.shape,'batch_labels',batch_labels.shape
        _, cross_entropy = sess.run([train_op, loss], feed_dict={x_image: batch_images, y_: batch_labels, keep_prob: 1})
        if batch % 100 == 0:
            print("epoch:" ,str(epoch) ," batch:",batch,'cross_entropy',cross_entropy)
#         train_writer.add_summary(summary, epoch*50000+batch)

    global_accuracy = 0
    for batch in range(10000//batch_size + 1):
        #batch_ys_onehot = np.eye(10, dtype=float)[batch_labels]
        batch_images_te, batch_labels_te = get_batch(batch_size,  XteAll,YteAll)
        batch_images_te=np.array(batch_images_te,dtype=np.float)
        batch_labels_te=np.array(batch_labels_te,dtype=np.float)
        #每个epoch后，计算训练出的模型在测试集上的准确率
        accuracy = sess.run([top_k_op], feed_dict={x_image: batch_images_te, y_: batch_labels_te, keep_prob: 1})
        # print accuracy
        global_accuracy+=np.sum(accuracy)
    print ('global_accuracy:',global_accuracy)
    test_accuracy = float(global_accuracy)/10000
    test_a.append(test_accuracy)
    print("epoch:%d  test accuracy %f" % (epoch, test_accuracy))
train_writer.close()
saver.save(sess, saveFile)
"""

for epoch in range(epoch_num):
    for batch in range(n_batch):
        #batch_ys_onehot = np.eye(10, dtype=float)[batch_labels]
        batch_images, batch_labels = get_batch(batch_size,  XtrAll,YtrAll)
        summary,_, cross_entropy = sess.run([merged,train_op, loss], feed_dict={x_image: batch_images, y_: batch_labels, keep_prob: 1})
        if batch % 100 == 0:
            print("epoch:" ,str(epoch) ," batch:",batch,'cross_entropy',cross_entropy)
        train_writer.add_summary(summary, epoch*50000+batch)

    #每个epoch后，计算训练出的模型在测试集上的准确率
    accuracy = sess.run([top_k_op], feed_dict={x_image: XteAll, y_: YteAll, keep_prob: 1})
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
