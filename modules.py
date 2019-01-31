import tensorflow as tf
from utils import nnUtils
import numpy as np
import copy


def lrelu(x,leak=0.2):
    return tf.maximum(x, leak*x)
def generator(z,kernel_len=25, dim=64,reuse=False, name="generator", num_seconds=1):
    with tf.variable_scope(name, reuse=reuse):
        layer0=tf.nn.relu(tf.reshape(tf.layers.dense(z,4*4*dim*16),[-1,16,dim*16]))
        layer1=tf.nn.relu(deconv1d(layer0,dim*8,kernel_len))
        layer2=tf.nn.relu(deconv1d(layer1,dim*4,kernel_len))
        layer3=tf.nn.relu(deconv1d(layer2,dim*2,kernel_len))
        layer4=tf.nn.relu(deconv1d(layer3,dim,kernel_len))
        layer5=tf.nn.tanh(deconv1d(layer4,1,kernel_len, stride=4*num_seconds))
        return layer5
def discriminator(x, dfdim=64, kernel_len=25,reuse=False, name="discriminator"):
    with tf.variable_scope(name, reuse=reuse):
        layer0=lrelu(tf.layers.conv1d(x, dfdim, kernel_len, 4, padding='SAME'))
        layer0=apply_phaseshuffle(layer0,0)
        layer1=lrelu(tf.layers.conv1d(layer0, dfdim*2, kernel_len, 4, padding='SAME'))
        layer1=apply_phaseshuffle(layer1,0)
        layer2=lrelu(tf.layers.conv1d(layer1, dfdim*4, kernel_len, 4, padding='SAME'))
        layer2=apply_phaseshuffle(layer2,0)
        layer3=lrelu(tf.layers.conv1d(layer2, dfdim*8, kernel_len, 4, padding='SAME'))
        layer3=apply_phaseshuffle(layer3,0)
        layer4=lrelu(tf.layers.conv1d(layer3, dfdim*16, kernel_len, 4, padding='SAME'))
        layer4=apply_phaseshuffle(layer4,0)
        flattened=tf.contrib.layers.flatten(layer4)
        out=tf.layers.dense(flattened,1)
        return out

#Code for deconv1d and applyy_phaseshuffle from https://github.com/chrisdonahue/wavegan
def deconv1d(input,filters,kernel,stride=4):
    return tf.layers.conv2d_transpose(tf.expand_dims(input, axis=1),
        filters,
        (1, kernel),
        strides=(1, stride),
        padding='same'
        )[:, 0]
def apply_phaseshuffle(x, rad, pad_type='reflect'):
  b, x_len, nch = x.get_shape().as_list()

  phase = tf.random_uniform([], minval=-rad, maxval=rad + 1, dtype=tf.int32)
  pad_l = tf.maximum(phase, 0)
  pad_r = tf.maximum(-phase, 0)
  phase_start = pad_r
  x = tf.pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)

  x = x[:, phase_start:phase_start+x_len]
  x.set_shape([b, x_len, nch])

  return x
