import tensorflow as tf
import sys
import matplotlib.pyplot as plt
import time
import random
import numpy as np
# updates the loading bar and accuracy in the console (useful for networks with many epochs)
# Args:
#   - time_taken: time taken to complete the most recent epoch
#   - acc: accuracy of the neural netowrk at this current epoch_loss
#   - epoch: the number epoch that was just completed
#   - num_epochs: the number of total epochs to complete
# Returns:
#   - None


def getUpdate(time_taken, acc, epoch, num_epochs):
    upstring = "|"
    numsigns = int(epoch / num_epochs * 50)
    for i in range(numsigns):
        upstring += "#"
    for i in range(50 - numsigns):
        upstring += " "
    upstring += "|  Accuracy: "
    upstring = upstring + str(acc) + "  "
    upstring += "Time left: "
    time_left_min = time_taken * (num_epochs - epoch) / 60
    time_left_sec = time_taken * (num_epochs - epoch) % 60
    upstring += str(int(time_left_min)) + " m   " + \
        str(int(time_left_sec)) + " s"
    sys.stdout.write("\r" + upstring)
    sys.stdout.flush()


# gets data for a batch from the entire dataset
# Args:
#   - batch_size: the size of the batch to fetch
#   - count: the current count within the epoch to continue from for this batch
#   - X_train: the training data
#   - Y_train: the training labels
# Returns:
#   - mini-batch of x's and y's of size batch_size

def get_mini_batch(batch_size, count, X_train, Y_train, Test=False):
    batch_x = []
    batch_y = []
    if Test and (batch_size * count + batch_size) == len(X_train):
        for i in range(batch_size - 1):
            batch_x.append(X_train[count * batch_size + i])
            batch_y.append(Y_train[count * batch_size + i])
            return batch_x, batch_y
    for i in range(batch_size):
        batch_x.append(X_train[count * batch_size + i])
        batch_y.append(Y_train[count * batch_size + i])
    return batch_x, batch_y


# creates a hidden_layer for use in a feedforward network
# Args:
#   - input_nodes: the number of input nodes for this layer
#   - output_nodes: the number of output nodes for this layer
#   - data: the data being passed into this layer
#   - relu: a boolean that decides whether or not to use a rectified linear unit on this layer, this hould only be false for the output layer
# Returns:
#   - a new hidden layer

def create_hidden_layer(input_nodes, output_nodes, data, relu=True):
    hidden_layer = {'weights': tf.Variable(tf.random_normal([input_nodes, output_nodes])),
                    'biases': tf.Variable(tf.random_normal([output_nodes]))}
    l = tf.add(
        tf.matmul(data, hidden_layer['weights']), hidden_layer['biases'])
    if relu:
        return tf.nn.relu(l)
    else:
        return l

# plots the accuracy data points for each epoch in this run
# Args:
#   - accpoints: the ordered accuracy points
#   - Title, the title of the graph
# Returns:
#   - None


def plot(accpoints, Title='Mnist Accuracy by Epoch'):
    plt.plot(accpoints)
    plt.ylabel('Accuracy %')
    plt.xlabel('Epoch #')
    plt.title(Title)
    plt.show()

# creates a convoluted layer for use in a convoluted network
# Args:
#   - input: the input data for the layer, use previous layer if there is one
#   - num_filters: the number fo features feeding into the layer
#   - filter_shape: a 2 element array where the first represents width of filer, and second represents height. ex. [5,5] a 5 x 5 filter
#   - pool_shape: a 2 element array where the first represents width of the pool, and second represents height. ex. [2,2] a 2 x 2 pool
#   - pool_stride: the size of the stride for the filer
# Returns:
#   - a new convoluted layer


def create_new_conv_layer(input, num_filters, filter_shape, pool_shape, stride, name):
    out_layer = tf.layers.conv2d(input, filters=num_filters, kernel_size=[filter_shape[0], filter_shape[1]],
                                 strides=[stride, stride], padding='SAME', activation=tf.nn.tanh)
    pool_size = [pool_shape[0], pool_shape[1]]
    strides = [2, 2]
    out_layer = tf.layers.max_pooling2d(
        out_layer, pool_size=pool_size, strides=strides, padding='SAME')
    return out_layer


# creates a dense layer for a convuluted netowrk
# Args:
#   - input: the vectorized input tensor
#   - units: the number of output nodes from this layer
#   - namespace: the namespace for this layer
# Returns:
#   - a new dense layer


def create_dense_layer(input, units, namespace):
    with tf.name_scope(namespace):
        return tf.layers.dense(input, units)
# print information about the current bunch of batches and total expected time of this run every specified batches and return an updated timestamp (useful for networks with few epochs)
# Args:
#   - num: the current bunch of batch number within this epoch
#   - size: How many batches are in this epoch
#   - epoch: current epoch number
#   - elapsed_time: the elapsed time since the last bunch of batches
#   - frequency: the frequency that controls how big a bunch of batches is. ex. 50 would mean a bunch is of size 50
#   - n_epochs: the total number of epochs in this run
#   - batch_size: the size of the batch
# Returns:
#   - The new start time if bunch of batches has completed, and the same start time otherwise


def loadString(num, size, epoch, elapsed_time, start_time, frequency, n_epochs, batch_size):
    if(num > 0 and num % frequency == 0):
        even_space = ""
        est_time_left = ((size - num) * frequency * .1 * elapsed_time / size) + \
            elapsed_time * (n_epochs - (epoch + 1)) * (size / batch_size)
        minutes = str(int(est_time_left / 60))
        seconds = str(int(est_time_left % 60))
        if(num < 100):
            even_space = "  "
        elif(num < 1000):
            even_space = " "
        Str = "Batch " + str(num) + even_space + " of " + str(int(size)) + " completed. Epoch: " + str(epoch + 1) + " out of " + str(n_epochs) + " Batch Size: " + \
            str(batch_size) + " Time Taken: " + str(round(elapsed_time, 2)) + \
            " s Time Remaining: " + str(minutes) + \
            " mins " + str(seconds) + " s"
        Strbar = ""
        for i in range(len(Str)):
            Strbar += "-"
        if(num == frequency):
            print(Strbar)
        sys.stdout.write("\r" + Str)
        sys.stdout.flush()
        return time.time()
    else:
        return start_time
def import_pokemon(data_path,color_dim=3):
    import os
    import cv2
    src=data_path
    dir_path = os.path.dirname(os.path.realpath(__file__))
    X_train=[]
    for each in os.listdir(src):
        path=os.path.join(src,each)
        img = cv2.imread(path, 1)
        if color_dim==1:
            img=np.expand_dims(img,axis=2)
        X_train.append(img)
    return np.asarray(X_train)
def import_audio(data_path):
    import os
    import wave
    import struct
    second_data=[]
    count=0
    for each in os.listdir(data_path):
        #print(os.path.join(data_path,each))
        waveFile=wave.open(os.path.join(data_path,each))
        length=waveFile.getnframes()
        alldata=[]
        for i in range(0,length):
            waveData=waveFile.readframes(1)
            binary_data=struct.unpack("<h",waveData)
            alldata.append(binary_data)
        waveFile.close()
        second_data.append(alldata)
        count+=1
        print(str(count) + " Files Scanned")
    return np.array(second_data)
def write_audio(data,path,iteration):
    import os
    import wave
    import struct
    for j in range(len(data)):
        x=[]
        #print(str(j) + " Out of: " + str(len(data)))
        signal=data[j]
        for i in range(len(signal)):
            x.append(wave.struct.pack('h',signal[i][0])) # transform to binary
        file_path=os.path.join(path,str(int((len(data)*iteration)+j)) + ".wav")
        file=wave.open(file_path, 'wb')
        file.setparams((1, 2, 16384, 44100, 'NONE', 'noncompressed'))
        x=np.array(x)
        file.writeframes(x)
        file.close()
        #print("File written!")
