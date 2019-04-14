#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import tensorflow as tf
import numpy as np
import input_data
# cifa = input_data.read_data_sets("../cifar_10", one_hot=True)
data_dir = '../cifar_10/cifar-10-batches-py'


# In[ ]:


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


# In[ ]:


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


# In[ ]:


import sys, os, pickle
# sys.path.append("..")
# sys.path.append("./tutorials/")
max_steps = 3000
batch_size = 128
XtrAll, YtrAll, XteAll, YteAll = load_CIFAR10(data_dir)


# In[ ]:


YtrAll.shape


# In[ ]:


def oneHotY(Y):
    Y_new=np.array([[0]*10]*len(Y))
    for i in range(len(Y)):
        Y_new[i][Y[i]]=1
    return Y_new

# YtrAll = oneHotY(YtrAll)
# YteAll = oneHotY(YteAll)


# In[ ]:


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


# In[ ]:


def add_layer(inputs, w_shape, b_shape, activation_function=None,mul_func = tf.matmul,pool=None,wl=0,stddev=5e-2,norm=False):
#     Weights = tf.Variable(tf.random_normal([in_size, out_size]))
#     biases = tf.Variable(tf.zeros([1, out_size]))
    Weights = weight_variable(w_shape,stddev=5e-2,wl=wl)
    biases = bias_variable(b_shape)
    Wx_plus_b = mul_func(inputs, Weights)+biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    if norm:
        outputs= tf.nn.lrn(outputs, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    if pool is not None:
        outputs = pool(outputs)
        
    return outputs


# In[ ]:


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)#真实标签0-9
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example'
    )#交叉熵
    cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                        name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)#加到最开始定义的集合这里最后又两部分交叉熵损失函数+L2损失函数

    return tf.add_n(tf.get_collection('losses'), name='total_loss')#拿到两部分损失

y_=tf.cast(y_, tf.int64)
loss = loss(logits=pred, labels=y_)
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
top_k_op = tf.nn.in_top_k(pred, y_, 1)


# In[ ]:


sess = tf.InteractiveSession()
# x = tf.placeholder('float', shape=[batch_size, 32, 32, 3])
# y_ = tf.placeholder('float', shape=[batch_size])

x = tf.placeholder(tf.float64, [batch_size, 32,32,3])
y_ = tf.placeholder(tf.float64, [batch_size])

sess.run(tf.global_variables_initializer())


# In[ ]:


w_shape1=[5,5,3,64]
b_shape1=[64]

x_image = tf.reshape(x,[-1, 32,32,3])
x_image = x 
out1 = add_layer(x_image,w_shape=w_shape1,
                 b_shape=b_shape1,mul_func=conv2d,
                 activation_function=tf.nn.relu, pool=max_pool_2x2)
norm1 = tf.nn.lrn(out1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)


# In[ ]:


w_shape2=[5,5,64,64]
b_shape2=[64]

out2 = add_layer(out1,w_shape=w_shape2,
                 b_shape=b_shape2,mul_func=conv2d,
                 activation_function=tf.nn.relu, pool=max_pool_2x2,norm=True)

out2_flat=tf.reshape(out2,[batch_size,-1]) # flatten
ofdim = out2_flat.get_shape()[1].value


# In[ ]:


w_shape3=[ofdim,1024]
b_shape3=[1024]

# out2_flat = tf.reshape(out2,[-1,ofdim])
out3 = add_layer(out2_flat,w_shape=w_shape3,
                 b_shape=b_shape3,
                 activation_function=tf.nn.relu,stddev=0.04, wl=0.004)

# 随机关闭一些神经元防止过拟
keep_prob = tf.placeholder("float")
out3 = tf.nn.dropout(out3, keep_prob)


# In[ ]:


w_shape4 =[1024, 512]
b_shape4 = [512]

out4 = add_layer(out3,w_shape4,b_shape4,activation_function=tf.nn.relu, stddev=0.04, wl=0.004)


# In[ ]:


# 从1024个神经元映射到10个神经元
w_shape5 =[512, 10]
b_shape5 = [10]

pred = add_layer(out4,w_shape5,b_shape5,tf.nn.softmax, stddev=1/512.0)

pred.shape
# labels = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
# b=tf.cast(labels, tf.int64)
# sess = tf.Session()
# sess.run(tf.initialize_all_variables())
# print(sess.run(b))


# In[ ]:


import math
import random
import time
# tf.train.start_queue_runners()
summary_op =  tf.summary.merge_all()
sess.run(tf.global_variables_initializer())
for step in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = get_batch(batch_size=batch_size,image=XtrAll,label=YtrAll)
    image_batch= np.array(image_batch)
    label_batch=1.0*np.array(label_batch)
    print image_batch.dtype, label_batch.shape
    
#     image_batch, label_batch = sess.run([images_train, labels_train])#真正执行tensor逻辑 返回的是一批次的数据
    _, loss_value,summary_str = sess.run([train_op, loss,summary_op],feed_dict={x: image_batch, y_: label_batch})
    duration = time.time() - start_time
    if step % 10 == 0:
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)

        format_str = 'step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)'
        print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))


# In[ ]:


"""def next_batch(datasetX,datasetY,batch_size,nowAt):
    return datasetX[nowAt:nowAt+batch_size],datasetY[nowAt:nowAt+batch_size],nowAt+batch_size"""


# In[ ]:


"""
# 计算交叉熵损失
cross_entropy = -tf.reduce_sum(y_*tf.log(pred))
# 创建优化器
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#计算准确率， tf.argmax函数 在 label 中找出数值最大的那个元素的下标
correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
"""


# In[ ]:


"""for i in range(20000):
    nowAt=0
#     print nowAt
    Xtr,Ytr,a = next_batch(XtrAll,YtrAll,batch_size,nowAt)
    Xte,Yte,nowAt = next_batch(XteAll,YteAll,batch_size,nowAt)
    if (i%100 == 0):
        train_accuracy = accuracy.eval(feed_dict={
        x:Xtr, y_: Ytr, keep_prob: 1.0})
        print ("step %d, training accuracy %g"%(i, train_accuracy))
      # 运行训练模型
    train_step.run(feed_dict={x: Xtr, y_: Ytr, keep_prob: 0.5})
print ("test accuracy %g"%accuracy.eval(feed_dict={
x: Xte, y_: Yte, keep_prob: 1.0}))"""


# In[ ]:





# In[ ]:





# In[ ]:




