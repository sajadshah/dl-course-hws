import Q4_1
from config import *
import numpy as np
import tensorflow as tf


def output_size_no_pool(input_size, filter_size, padding, conv_stride):
    if padding == 'same':
        padding = -1.00
    elif padding == 'valid':
        padding = 0.00
    else:
        return None
    output_1 = float(((input_size - filter_size - 2 * padding) / conv_stride) + 1.00)
    output_2 = float(((output_1 - filter_size - 2 * padding) / conv_stride) + 1.00)
    return int(np.ceil(output_2))


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


batch_size = 50
graph = tf.Graph()

with graph.as_default():
    '''Input data'''
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, img_size, img_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_classes))

    '''Variables'''
    # Convolution 1 Layer
    # Input channels: num_channels = 1
    # Output channels: depth = 16
    depth = 64
    filter_size = 5

    '''Model'''
    layer1_weights = tf.Variable(tf.truncated_normal([filter_size, filter_size, num_channels, depth], stddev=0.1))

    conv1 = tf.nn.conv2d(tf_train_dataset, layer1_weights, strides=[1, 1, 1, 1], padding='SAME')

    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    layer1_biases = tf.Variable(tf.zeros([depth]))
    hidden1 = tf.nn.relu(pool1 + layer1_biases)

    # Convolution 2 Layer
    # Input channels: depth = 16
    # Output channels: depth = 16
    layer2_weights = tf.Variable(tf.truncated_normal([filter_size, filter_size, depth, depth], stddev=0.1))
    conv2 = tf.nn.conv2d(hidden1, layer2_weights, strides=[1, 1, 1, 1], padding='SAME')

    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    layer2_biases = tf.Variable(tf.zeros([depth]))
    hidden2 = tf.nn.relu(pool2 + layer2_biases)
    # Fully Connected Layer (Densely Connected Layer)
    # Use neurons to allow processing of entire image
    final_image_size = output_size_no_pool(img_size, filter_size, padding='same', conv_stride=2)

    num_hidden = 512
    layer3_weights = tf.Variable(
        tf.truncated_normal([final_image_size * final_image_size * depth, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.zeros([num_hidden]))
    shape = hidden2.get_shape().as_list()
    reshape = tf.reshape(hidden2, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden3 = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)

    # Readout layer: Softmax Layer
    layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1))
    layer4_biases = tf.Variable(tf.zeros([num_classes]))
    logits = tf.matmul(hidden3, layer4_weights) + layer4_biases


    '''Training computation'''
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))

    '''Optimizer'''
    # Learning rate of 0.05
    lobal_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.1
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               300, 0.90, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    '''Predictions for the training, validation, and test data'''
    train_prediction = tf.nn.softmax(logits)

    num_steps = 30000
    images, labels, test_images, test_labels = Q4_1.readData()

    test_batch_number = len(test_images) // batch_size

    with tf.Session(graph=graph) as session:
        filewriter = tf.summary.FileWriter("log")
        filewriter.add_graph(session.graph)
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        labels = tf.one_hot(labels, num_classes, dtype=tf.int32)
        test_labels = tf.one_hot(test_labels, num_classes, dtype=tf.int32)
        labels, test_labels = session.run([labels, test_labels])

        tf.initialize_all_variables().run()
        print('Initialized')
        for step in range(num_steps):
            offset = (step * batch_size) % (labels.shape[0] - batch_size)
            batch_data = images[offset:(offset + batch_size), :, :, :]
            batch_labels = labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            test_l = 0
            for i in range(test_batch_number):
                offset = (i * batch_size) % (test_labels.shape[0] - batch_size)
                test_batch_data = test_images[offset:(offset + batch_size), :, :, :]
                test_batch_labels = test_labels[offset:(offset + batch_size), :]
                feed_dict = {tf_train_dataset: test_batch_data, tf_train_labels: test_batch_labels}
                test_batch_l, test_bathc_predictions = session.run([loss, train_prediction], feed_dict=feed_dict)
                test_l += test_batch_l

            test_l /= test_batch_number
            print(step)
            if (step % 1 == 0):
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                print('Test loss at step %d: %f' % (step, test_l))

            if (step % 500 == 0):
                save_path = saver.save(session, "./models/m_{}.ckpt".format(step))
                #print('Test accuracy: %.1f%%' % accuracy(test_predictions, test_labels))
