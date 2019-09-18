#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import math
from tensorflow.python.framework import ops
import warnings
import random
import os

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# In[2]:


logs_path = 'log_sigmoid/'
batch_size = 100
learning_rate = 0.003
training_epochs = 10
display_epoch = 1


# In[3]:


dataPath = "temp/"
if not os.path.exists(dataPath):
    os.makedirs(dataPath)

mnist = input_data.read_data_sets(dataPath,one_hot=True) # MNIST to be downloaded


# In[4]:


print(mnist.train.images.shape)
print(mnist.train.labels.shape)
print(mnist.test.images.shape)
print(mnist.test.labels.shape)


# In[5]:


image_0 = mnist.train.images[0]
image_0 = np.resize(image_0, (28,28))
label_0 = mnist.train.labels[0]
print(label_0)


# In[6]:


plt.imshow(image_0, cmap='Greys_r')
plt.show()


# In[7]:


# 图像大小 28*28=784。[None, 784]表示任意行，784列
X = tf.placeholder(tf.float32, [None, 784], name = 'InputData') 

#reshape中-1表示根据另一个维度自动匹配大小。为什么这里还要reshape，上面不是已经784列的任意行了吗？
XX = tf.reshape(X, [-1, 784]) 

# 数字0-9的10个分类。Y_中放置label数据, 任意行10列
Y_ = tf.placeholder(tf.float32, [None, 10], name = 'LabelData') 

L = 200 # layer 1 的neurons数量
M = 100 # layer 2 的neurons数量
N = 60 # layer 3 的neurons数量
O = 30 # layer 4 的neurons数量

# Hidden Layer 1

# 随机初始化hidden Layer 1 的Weights。从截断的正态分布中输出随机值。stddev：正态分布的标准差
W1 = tf.Variable(tf.truncated_normal([784, L], stddev=0.1))
# Layer 1 的bias，初始化为0
B1 = tf.Variable(tf.zeros([L])) 
# Layer 1 output
Y1 = tf.nn.sigmoid(tf.matmul(XX, W1) + B1) 

# Hidden Layer 2
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1)) 
B2 = tf.Variable(tf.ones([M])) 
Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + B2) 

# Hidden Layer 3
W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1)) 
B3 = tf.Variable(tf.ones([N])) 
Y3 = tf.nn.sigmoid(tf.matmul(Y2, W3) + B3) 

# Hidden Layer 4
W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1)) 
B4 = tf.Variable(tf.ones([O])) 
Y4 = tf.nn.sigmoid(tf.matmul(Y3, W4) + B4) 

# Layer 5 output layer . 激活函数使用softmax
W5 = tf.Variable(tf.truncated_normal([O, 10], stddev=0.1)) 
B5 = tf.Variable(tf.ones([10])) 

Ylogits = tf.matmul(Y4, W5) + B5 # logits ：未归一化的概率， 一般也就是 softmax层的输入
Y = tf.nn.softmax(Ylogits)


# In[8]:


# 交叉熵损失函数，BP算法

#下面函数的返回值类型与logits相同，shapes与labels相同
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=Ylogits, labels=Y_) 

#reduce_mean, 矩阵元素求平均值，或者在一个方向上的平均值。下面得到每个样本的交叉熵均值（0到1之间）
cost_op = tf.reduce_mean(cross_entropy)*100 

# tf.argmax() 返回矩阵在行方向（第二参数为1）或列方向（第二参数为0）的最大值的索引
# 在Y和Y_中，这个正好代表列one_hot编码中为1的位置
# 再使用equal()函数，对Y和Y_每一行的hot编码进行比较。相同为True，不同为False。
# 使用cast转换类型，True为1，False为0。所以，求均值正是得到了准确度
correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Y_, 1)) 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 

#使用优化器。Optimizer对损失函数cost_op进行最小化的优化，对Variable进行调整
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost_op)

init_op = tf.global_variables_initializer()

#创建summary用于在TensorBoard上进行监控
tf.summary.scalar("cost", cost_op)
tf.summary.scalar("accuracy", accuracy)
summary_op = tf.summary.merge_all()


# In[9]:


"""
# 理解argmax()
from numpy import *
xxx = array([
    [1,2,3],
    [2,3,4],
    [5,4,3],
    [8,7,2]])

print(np.argmax(xxx,0))
print(np.argmax(xxx,1))

# 理解tf.equal(), tf.cast(), tf.reduce_mean()

import tensorflow as tf
x = [1, 3, 0, 2]
y = [1, 4, 2, 2]
equal = tf.equal(x, y)
type = tf.cast(equal, tf.float32)
with tf.Session() as sess:
    print(sess.run(equal))
    print(sess.run(type))
    print(sess.run(tf.reduce_mean(type)))
    
"""


# In[10]:


# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init_op)
    
    avg_cost = 0.
    
    # op to write logs to TensorBoard
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    for epoch in range(training_epochs):
        batch_count = int(mnist.train.num_examples/batch_size)
        agg_cost = 0.
        for i in range(batch_count):
            batch_x, batch_y = mnist.train.next_batch(batch_size) #基于training set进行训练
            _, c, summary = sess.run([train_op, cost_op, summary_op], feed_dict={X: batch_x, Y_: batch_y})
            writer.add_summary(summary, epoch * batch_count + 1)
            #avg_cost += c / batch_count
            agg_cost += c
            avg_cost = agg_cost / (i+1)
            if (epoch+1) % display_epoch == 0:
                print("Epoch:", '%04d' % (epoch+1), "i=", '%d' % (i+1), "cost=", "{:.9f}".format(avg_cost))
                
    print("Optimization Finished!")    
    print("Accuracy: ", accuracy.eval(feed_dict={X: mnist.test.images, Y_: mnist.test.labels})) #最后，基于test set计算accuracy
            


# In[ ]:


### 训练完之后使用 TensorBoard 查看accuracy和cost的训练曲线
#
# 在终端上进入项目执行目录，键入下面的命令，启动tensorboard
# cd <ProjFolder>/log_sigmid 
# tensorboard --logdir='./'
#
# 使用浏览器打开 http://localhost:6006
#


# In[ ]:




