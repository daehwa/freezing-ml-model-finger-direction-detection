
# coding: utf-8

# In[14]:

import tensorflow as tf
import numpy as np
import os

# path where trained network to be saved
net_dir = '/home/daehwakim/freeze'

# define file name (!!do not change!!)
checkpoint_prefix = os.path.join(net_dir, "saved_checkpoint")
checkpoint_state_name = "checkpoint_state"
input_graph_name = "input_graph.pb"
output_graph_name = "output_graph.pb"
output_node_names="hypothesis"


def MinMaxScaler(data):
    print(data)
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    #print("n: ",numerator," d: ",denominator)
    print("min ",np.min(data,0))
    print("max ",np.max(data,0))
    print("return ",numerator / (denominator + 1e-7))
    return numerator / (denominator + 1e-7)

# classifying finger shape
xy_data = np.loadtxt('./shear_detection_daehwakim.csv', delimiter=',', dtype=np.float32)
x_data = xy_data[:,0:-1]
y_data = xy_data[:,[-1]]

x_data = MinMaxScaler(x_data)

xy_test = np.loadtxt('./shear_detection_daehwakim_test.csv', delimiter=',', dtype=np.float32)
x_test = xy_test[:,0:-1]
y_test = xy_test[:,[-1]]

x_test = MinMaxScaler(x_test)

X = tf.placeholder(tf.float32,shape=[None,49],name='input_node')
Y = tf.placeholder(tf.float32,shape=[None,1],name='output_node')

W = tf.Variable(tf.random_normal([49,1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')

hypothesis = tf.sigmoid(tf.matmul(X,W)+b, name='hypothesis')

cost = - tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y),dtype = tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    feed = {X: x_data, Y: y_data}
    feed_test = {X: x_test, Y: y_test}
    for step in range (10001):
        sess.run(train, feed_dict=feed)
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict=feed))
    
#    for n in tf.get_default_graph().as_graph_def().node:
#        print(n.name)

    h,c,a = sess.run([hypothesis,predicted,accuracy], feed_dict=feed_test)
    print("\nHypothesis:",h,"\nCorrect(Y):",c,"\nAccuracy:",a)

    checkpoint_path = tf.train.Saver().save(sess, checkpoint_prefix, global_step=0, latest_filename=checkpoint_state_name)
    tf.train.write_graph(sess.graph_def, net_dir, input_graph_name, as_text=False)
