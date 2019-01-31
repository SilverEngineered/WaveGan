import tensorflow as tf
import numpy as np
import random
from modules import*
from utils import nnUtils
from utils import data_utilsWDB as util
import cv2
class wavegan(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.dataset_dir = args.dataset_dir
        self.epoch=args.epoch
        self.discriminator = discriminator
        self.generator = generator
        self.x_data= nnUtils.import_audio(self.dataset_dir)
        self.x_data=util.scale_data(self.x_data)
        self.output_dir=args.output_dir
        self.a_len=args.a_len
        self.glr=args.glr
        self.dlr=args.dlr
        self.num_seconds=args.num_seconds
        self.z_dims=args.z_dims
        self.lamda=args.lamda
        self.num_critic_steps=args.num_critic_steps
        self.build()
    def build(self):
        self.x=tf.placeholder(tf.float32, [None, self.a_len, 1])
        self.z=tf.placeholder(tf.float32, [None, self.z_dims])

        self.genx = generator(self.z, num_seconds=self.num_seconds)

        self.rand = tf.random_uniform([tf.shape(self.x)[0]], minval=0, maxval=1)
        self.interp = tf.transpose((self.rand * tf.transpose(self.x, [2, 1, 0])), \
                                [2,1,0]) + tf.transpose(((1 - self.rand) * \
                                tf.transpose(self.genx, [2,1,0])), [2,1,0])
        self.Interpolator = discriminator(self.interp,name="GAN/discriminator")
        self.c_out_int = tf.reshape(self.Interpolator, [-1, 1])
        self.c_grad_int = tf.gradients(self.c_out_int, self.interp)[0]
        self.lag_int = tf.reduce_mean(tf.pow((tf.norm(self.c_grad_int, ord='euclidean', axis=(1, 2)) - 1), 2))
        self.dx = discriminator(self.x, name="GAN/discriminator", reuse=True)
        self.dg = discriminator(self.genx, name="GAN/discriminator",reuse=True)
        self.wd = tf.reduce_mean(self.dx-self.dg)
        self.d_loss = (self.lamda*self.lag_int)-self.wd
        self.g_loss=self.wd

        vars=tf.trainable_variables()
        self.d_vars=[v for v in vars if (v.name.startswith("GAN/discriminator"))]
        self.g_vars=[v for v in vars if (v.name.startswith("gen"))]
    def train(self,args):
        self.d_optim = tf.train.AdamOptimizer(self.dlr) \
            .minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.glr) \
            .minimize(self.g_loss, var_list=self.g_vars)
        iteration = tf.Variable(0, dtype=tf.int32)
        increment_iter = tf.assign(iteration, iteration + 1)
        init=tf.global_variables_initializer()
        self.sess.run(init)
        for i in range(self.epoch):
            for j in range(int(self.x_data.shape[0]/self.batch_size)):
                z_batch = np.random.randn(self.batch_size, self.z_dims)
                randlist_x = np.random.randint(0, self.x_data.shape[0], self.batch_size)
                real_x_sample=np.reshape(self.x_data,(self.x_data.shape[0],self.a_len,1))
                real_x_sample=real_x_sample[randlist_x]
                for k in range(self.num_critic_steps):
                    # Update D network

                    d_loss, _ = self.sess.run(
                        [self.d_loss,self.d_optim],
                        feed_dict={self.x: real_x_sample,
                        self.z: z_batch})

                #G Network
                fake_x, _ = self.sess.run(
                [self.genx, self.g_optim],
                feed_dict={self.z: z_batch, self.x: real_x_sample})
                wd=self.sess.run(self.wd, feed_dict={self.z: z_batch, self.x: real_x_sample})
                dx=self.sess.run(self.dx, feed_dict={self.z: z_batch, self.x: real_x_sample})
                dg=self.sess.run(self.dg, feed_dict={self.z: z_batch, self.x: real_x_sample})
                print("Wasserstein Disctance: " + str(wd))
            print("Iterations: %d\t" %(i))
            self.sess.run(increment_iter)
            if (i%1==0):
                g_batch = util.scale_data(fake_x, scale=[-32759, 32759], dtype=np.int32)
                nnUtils.write_audio(g_batch[:10],self.output_dir,i)
