#!/usr/bin/env python

"""
Description: Utilities for handling datasets.
Author: Dylan Elliott
Date: 01/31/2018
References:
    MNIST datasets from: http://yann.lecun.com/exdb/mnist/
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os

# MNIST datasets
MNIST_TRAIN_IMG = 'train-images-idx3-ubyte.gz'
MNIST_TRAIN_LAB = 'train-labels-idx1-ubyte.gz'
MNIST_TEST_IMG = 't10k-images-idx3-ubyte.gz'
MNIST_TEST_LAB = 't10k-labels-idx1-ubyte.gz'

# imports the MNIST dataset using TensorFlow's API
# Args:
#   - data_path: path to local directory where the mnist gz files are located
#   - vectorize: flag for vectorizing the mnist images if TRUE (default=FALSE)
# Returns:
#   - six numpy arrays of the mnist data (train, validate, test)
def import_mnist(data_path, vectorize=False):
    # import nnist training images
    with open(data_path + MNIST_TRAIN_IMG, 'rb') as fp:
        # import the MNIST dataset from its filename located at 'data_path' argument
        X = tf.contrib.learn.datasets.mnist.extract_images(fp)

    # import nnist training labels
    with open(data_path + MNIST_TRAIN_LAB, 'rb') as fp:
        # import the MNIST dataset from its filename located at 'data_path' argument
        Y = tf.contrib.learn.datasets.mnist.extract_labels(fp)

    # import nnist testing images
    with open(data_path + MNIST_TEST_IMG, 'rb') as fp:
        # import the MNIST dataset from its filename located at 'data_path' argument
        X_test = tf.contrib.learn.datasets.mnist.extract_images(fp)

    # import nnist testing labels
    with open(data_path + MNIST_TEST_LAB, 'rb') as fp:
        # import the MNIST dataset from its filename located at 'data_path' argument
        Y_test = tf.contrib.learn.datasets.mnist.extract_labels(fp)

    if (vectorize):
        # vectorize each image of the training set
        n, h, w, d = X.shape
        X = np.reshape(X, (n, h*w*d))

        # vectorize each image of the testing set
        n, h, w, d = X_test.shape
        X_test = np.reshape(X_test, (n, h*w*d))

    num_train = int(0.8 * X.shape[0])
    X_train = X[:num_train]
    Y_train = Y[:num_train]
    X_val = X[num_train:]
    Y_val = Y[num_train:]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


# imports the CIFAR dataset
# Args:
#   - data_path: path to local directory where the mnist batch files are located
#   - vectorize: flag for vectorizing the cifar images if TRUE (default=FALSE)
# Returns:
#   - six numpy arrays of the cifar data (train, validate, test)
def import_cifar(data_path, batches, vectorize=False):
    import cPickle
    for i in batches:
        # open the file for this batch of cifar data
        with open(data_path + 'data_batch_' + str(i), 'rb') as fp:
            # if this is the first batch, create a new data array
            if (i == 1):
                data_dict = cPickle.load(fp)
                X = np.asarray(data_dict['data'])
                Y = np.asarray(data_dict['labels'])
                if (not vectorize):
                    X_r = np.reshape(X[:, :1024], [X.shape[0], 32, 32, 1])
                    X_g = np.reshape(X[:, 1024:2*1024], [X.shape[0], 32, 32, 1])
                    X_b = np.reshape(X[:, 2*1024:], [X.shape[0], 32, 32, 1])
                    X = np.concatenate([X_b, X_g, X_r], axis=3)
            # if this isnt the first batch, concatenate to existing data array
            else:
                data_dict = cPickle.load(fp)
                X_c = np.asarray(data_dict['data'])
                Y_c = np.asarray(data_dict['labels'])
                if (not vectorize):
                    X_r = np.reshape(X_c[:, :1024], [X_c.shape[0], 32, 32, 1])
                    X_g = np.reshape(X_c[:, 1024:2*1024], [X_c.shape[0], 32, 32, 1])
                    X_b = np.reshape(X_c[:, 2*1024:], [X_c.shape[0], 32, 32, 1])
                    X_c = np.concatenate([X_b, X_g, X_r], axis=3)

                # concatenate the batch to the growing data array
                X = np.concatenate([X, X_c], axis=0)
                Y = np.concatenate([Y, Y_c], axis=0)

        with open(data_path + 'test_batch', 'rb') as fp:
            data_dict = cPickle.load(fp)
            X_test = np.asarray(data_dict['data'])
            Y_test = np.asarray(data_dict['labels'])
            if (not vectorize):
                X_r = np.reshape(X_test[:, :1024], [X_test.shape[0], 32, 32, 1])
                X_g = np.reshape(X_test[:, 1024:2*1024], [X_test.shape[0], 32, 32, 1])
                X_b = np.reshape(X_test[:, 2*1024:], [X_test.shape[0], 32, 32, 1])
                X_test = np.concatenate([X_b, X_g, X_r], axis=3)

    num_train = int(0.8 * X.shape[0])
    X_train = X[:num_train]
    Y_train = Y[:num_train]
    X_val = X[num_train:]
    Y_val = Y[num_train:]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


# Scales values of a dataset to within a specified range
# Args:
#   - data: numpy array of the dataset to scale
#   - scale: list of 2 ints for scale bounds [lower bound, upper bound]
#   - dtype: data type for scaled output array
# Returns:
#   - scaled and type-converted numpy array of dataset
def scale_data(data, scale=[0, 1], dtype=np.float32):
    min_data, max_data = [float(np.min(data)), float(np.max(data))]
    min_scale, max_scale = [float(scale[0]), float(scale[1])]
    data = ((max_scale - min_scale) * (data - min_data) / (max_data - min_data)) + min_scale
    return data.astype(dtype)


# General wrapper for importing common image datasets using predefined
# functions for importing each individual dataset
# Args:
#   - dataset: string specifying the name of the dataset (e.g. 'mnist')
#   - datapath: path to the dataset files needed for the unique dataset import
#       function (defined above)
#   - scale: list of 2 ints for scale bounds [lower bound, upper bound]
#   - dtype: data type for scaled output array
#   - vectorize: flag for vectorizing the dataset if TRUE (default=FALSE)
# Returns:
#   - list of scaled and type-converted numpy arrays of the dataset train,
#       validation and testing partitions
def import_image_dataset(dataset, datapath, scale=[-1, 1],
                            dtype=np.float32, vectorize=False):
    if (dataset=='mnist'):
        # import mnist images
        datas = import_mnist(datapath, vectorize=vectorize)
    elif (dataset=='cifar'):
        # import cifar images
        datas = import_cifar(datapath, [1, 2, 3, 4, 5], vectorize=vectorize)
    elif (dataset=='fashion'):
        # import fashion mnist images
        datas = import_mnist(datapath, vectorize=vectorize)
    else:
        print('[ERROR]: dataset \'{:s}\' not supported'.format(dataset))
        exit()

    # scale mnist images and delete old data
    scaled_datas = []
    for i, d in enumerate(datas):
        if (i % 2 == 0):
            scaled_datas.append(scale_data(d, scale=scale, dtype=dtype))
        else:
            scaled_datas.append(d)
    del datas

    return scaled_datas


# generates a random batch of sample-labels pairs from a dataset
# Args:
#   - X: dataset samples
#   - Y: dataset labels
#   - batch_size: size of batch to return
# Returns:
#   - X_batch: batch of samples
#   - Y_batch: batch of labels
def get_random_batch(X, Y, batch_size):
    # generate a random set of indices within the amount of samples in X
    indices = np.random.randint(0, X.shape[0], batch_size)

    # select these indices from the samples and labels
    X_batch = X[indices]
    Y_batch = Y[indices]

    return X_batch, Y_batch, indices


# generates a batch of images from an image directory
# Args:
#   - src_dir: path to image directory
#   - batch_size: number of samples to collect for the batch
# Returns:
#   - a batch array of images of size [bs, h, w, d]
def stream_random_batch(src_dir, batch_size):
    files = os.listdir(src_dir)
    num_files = len(files)
    indices = np.random.permutation(num_files)

    batch_indices = indices[:batch_size].tolist()
    batch = [files[i] for i in batch_indices]

    f = batch[0]
    filename = src_dir + '/' + f
    img = cv2.imread(filename)
    batch_array = np.repeat(np.expand_dims(np.zeros_like(img), axis=0), batch_size, axis=0)

    for i, f in enumerate(batch):
        filename = src_dir + '/' + f
        img = cv2.imread(filename)
        batch_array[i] = img

    return batch_array


# prints a tiling of image arrays as a single image
# Args:
#   - imgs: 4D numpy array of 3D image arrays (must be perfect square eg. 25)
#   - fname: name for output image file
#   - title: title for image
# Returns
#   None
def tile_imgs(imgs, fname, title=None):
    n, h, w, d = imgs.shape

    bh, bw = int(np.sqrt(n)), int(np.sqrt(n))

    img = np.zeros((bh * h, bw * w, d))
    for t in range(n):
        j = w * (t % bw)
        i = h * (t // bh)
        img[i:(i+h), j:(j+w), :] = imgs[t]

    img = cv2.resize(img.astype('uint8'), (1000, 1000))
    cv2.imwrite(fname, img)


# converts a numpy array of class scalars to a matrix of one-hot vectors
# Args:
#   - classes:
#   - fname: name for output image file
#   - title: title for image
# ReturnsL
#   None
def classes_2_onehot(classes, num_classes):
    num_samples = len(classes)

    classes_oh = np.zeros((num_samples, num_classes))

    classes_oh[np.arange(num_samples), classes] = 1

    return classes_oh


def sample_class_dist(num_classes, num_samples):
    classes = np.random.randint(0, num_classes, [num_samples])
    classes_oh = classes_2_onehot(classes, num_classes)

    return classes, classes_oh
