import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("mnist").train

x=tf.placeholder(dtype=tf.float32,shape=(None,784))
y=tf.placeholder(dtype=tf.float32,shape=(None,784))

w1=tf.Variable(tf.random_uniform(shape=(784,500),minval=-1,maxval=1))
b1=tf.Variable(tf.zeros(500))

encodelayer=tf.matmul(x,w1)+b1
sigmoid1=tf.sigmoid(encodelayer)

w2=tf.Variable(tf.random_uniform((500,784),-1,1))
b2=tf.Variable(tf.zeros(784))

decodelayer=tf.matmul(sigmoid1,w2)+b2
output=tf.sigmoid(decodelayer)

loss=tf.reduce_sum(tf.square(output-y))
train=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

tf.summary.scalar("loss",loss)
merge=tf.summary.merge_all()
filewriter=tf.summary.FileWriter("log")
sess=tf.Session()
sess.run(tf.global_variables_initializer())
saver=tf.train.Saver()
#saver.restore(sess=sess,save_path="save/")

for i in range(1000000):
    inputimage,labels=mnist.next_batch(32)
    sess.run(train,feed_dict={x:inputimage,y:inputimage})
    if i%100==0:
        saver.save(sess=sess,save_path="save/")
        yyy=sess.run(output,feed_dict={x:inputimage})
        aa=np.reshape(yyy[0],newshape=(28,28))*255
        a,b=(sess.run((loss,merge),feed_dict={x:inputimage,y:inputimage}))
        print(a)
        filewriter.add_summary(b,i)
