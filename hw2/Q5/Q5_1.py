import tensorflow as tf
import numpy as np
from load import *

learning_rate = 0.005
batch_size=50
n_classes = 10
n_h1 = 10
steps = (len(train_images) // batch_size) * 100
print(len(train_images))
print(steps)
sigma = 0.1

x=tf.placeholder(dtype=tf.float32,shape=(None,784))
y=tf.placeholder(dtype=tf.float32,shape=(None,n_classes))

w1=tf.Variable(tf.random_normal([784,n_h1], stddev=sigma))
b1=tf.Variable(tf.zeros([n_h1]))

z1=tf.add(tf.matmul(x,w1),b1)
h1=tf.sigmoid(z1)

w2=tf.Variable(tf.random_normal([n_h1,n_classes], stddev=sigma))
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

train_labels = tf.one_hot(train_labels, n_classes, dtype=tf.int32)
test_labels = tf.one_hot(test_labels, n_classes, dtype=tf.int32)
train_labels, test_labels = sess.run([train_labels, test_labels])

# train_mean = np.mean(train_images, axis=0)
# train_std = np.std(train_images, axis=0)
# for i in range(len(train_std)):
#     train_std[i] = max(train_std[i], 1.0/np.sqrt(784))
#
# train_images = (train_images - train_mean) / train_std
# test_images = (test_images - train_mean) / train_std

batches = 0
for i in range(steps):
    offset = (i * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_images[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]

    sess.run(train,feed_dict={x:batch_data,y:batch_labels})
    if i%100 == 0:
        saver.save(sess=sess,save_path="save/")
        a,acc, b=(sess.run((loss,accuracy,merge),feed_dict={x:batch_data,y:batch_labels}))
        print('{},{},{}'.format(i, a, acc))
        filewriter.add_summary(b,i)

train_loss, train_acc = sess.run((loss, accuracy), feed_dict={x: train_images, y: train_labels})
print('train loss:{} train acc:{}'.format(train_loss, train_acc))
test_loss, test_acc = sess.run((loss, accuracy), feed_dict={x: test_images, y: test_labels})
print('test loss:{} test acc:{}'.format(test_loss, test_acc))
