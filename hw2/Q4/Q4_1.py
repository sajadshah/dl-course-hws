
from config import *
import numpy as np
import os
from PIL import Image


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def readData():
    batchesMeta = unpickle(os.path.join(datasetDir, 'batches.meta'))
    batches = [f for f in os.listdir(datasetDir) if f.startswith('data_batch_')]
    images = np.zeros(shape=[num_images_train, img_size, img_size, num_channels], dtype=np.uint8)
    labels = np.zeros(shape=[num_images_train], dtype=np.int)

    begin = 0
    for fb in batches:
        batch = unpickle(os.path.join(datasetDir, fb))
        raw_images = batch[b'data']
        cls = np.array(batch[b'labels'])

        images_reshaped = raw_images.reshape([-1, num_channels, img_size, img_size])
        images_reshaped = images_reshaped.transpose([0, 2, 3, 1])

        num_images = len(raw_images)
        end = begin + num_images
        images[begin:end, :] = images_reshaped
        labels[begin:end] = cls

        begin = end
    test_batch = unpickle(os.path.join(datasetDir, 'test_batch'))
    raw_images = test_batch[b'data']
    test_labels = np.array(test_batch[b'labels'])
    test_images = raw_images.reshape([-1, num_channels, img_size, img_size])
    test_images = test_images.transpose([0, 2, 3, 1])

    return images, labels, test_images, test_labels

if __name__ == '__main__':
    images, _ = readData()
    image = images[-1]
    img = Image.fromarray(image, 'RGB')
    img.show()

