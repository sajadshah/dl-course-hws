import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("mnist", one_hot=True)

train_data = mnist.train;
test_data = mnist.test
val_data = mnist.validation

learning_rate = 0.5
batch_size=100
n_classes = 10
n_h1 = 10
steps = 2000
sigma = 2

x=tf.placeholder(dtype=tf.float32,shape=(None,784))
y=tf.placeholder(dtype=tf.float32,shape=(None,n_classes))

w1=tf.Variable(tf.random_normal([784,n_h1], stddev=sigma))
#w1=tf.Variable(tf.zeros([784,n_h1]))
b1=tf.Variable(tf.zeros([n_h1]))

z1=tf.add(tf.matmul(x,w1),b1)
h1=tf.sigmoid(z1)

w2=tf.Variable(tf.random_normal([n_h1,n_classes], stddev=sigma))
#w2=tf.Variable(tf.zeros([n_h1,n_classes]))
b2=tf.Variable(tf.zeros([n_classes]))

z2=tf.add(tf.matmul(h1,w2),b2)
yhat = z2

out=tf.argmax(yhat, axis=1)

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
train=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

correct_pred = tf.equal(out, tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.summary.scalar("loss",loss)
merge=tf.summary.merge_all()
filewriter=tf.summary.FileWriter("log")
sess=tf.Session()
sess.run(tf.global_variables_initializer())
saver=tf.train.Saver()
#saver.restore(sess=sess,save_path="save/")

batches = 0
for i in range(1, steps+1):
    inputimage,labels=train_data.next_batch(batch_size)
    sess.run(train,feed_dict={x:inputimage,y:labels})
    if i%100==0 or i==1:
        saver.save(sess=sess,save_path="save/")
        a,acc, b=(sess.run((loss,accuracy,merge),feed_dict={x:inputimage,y:labels}))
        print('{},{},{}'.format(i, a, acc))
        filewriter.add_summary(b,i)

train_loss, train_acc = sess.run((loss, accuracy), feed_dict={x: train_data.images, y: train_data.labels})
print('train loss:{} train acc:{}'.format(train_loss, train_acc))
test_loss, test_acc = sess.run((loss, accuracy), feed_dict={x: test_data.images, y: test_data.labels})
print('test loss:{} test acc:{}'.format(test_loss, test_acc))
