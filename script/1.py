import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


tf.set_random_seed(1)

#x_data=np.random.rand(100).astype(np.float32)[:, np.newaxis]
x_data=np.linspace(-1, 1, 100)[:, np.newaxis] 

noise=np.random.rand(100).astype(np.float32)

y_data=np.power(x_data,2)+6


tf_x = tf.placeholder(tf.float32, x_data.shape)     # input x
tf_y = tf.placeholder(tf.float32, y_data.shape)     # input y

#structure

ll=tf.layers.dense(tf_x,10,tf.nn.relu)

output=tf.layers.dense(ll,1)

weight=tf.Variable(tf.random_uniform([1],-1.0,1.0))

biases=tf.Variable(1.0)    

y=weight*x_data+biases

loss=tf.reduce_mean(tf.square(output-tf_y))

optimizer=tf.train.GradientDescentOptimizer(0.2)

train=optimizer.minimize(loss)

#structure

sess = tf.Session()                                 # control training and others
sess.run(tf.global_variables_initializer())         # initialize var in graph




for step in range(500):
    _, l, pred=sess.run([train, loss, output], {tf_x: x_data, tf_y: y_data})
    if step%10==0:
        plt.cla()
        plt.scatter(x_data, y_data)
        plt.plot(x_data, pred,"r-")
        plt.pause(0.2)
    




#print(x_data)

