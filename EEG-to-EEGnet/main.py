import tensorflow as tf
import numpy as np
# import input_data
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import os
from scipy.misc import imsave as ims
from utils import *
from ops import *
from load_img import load_img, Data

class LatentAttention():
    def __init__(self):
       # self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
       # self.n_samples = self.mnist.train.num_examples
        #self.mnist = Data("../EEGdata/train5/images/")
        self.mnist = Data(os.getcwd() + "/../EEGdata/train/images/")
        self.n_samples = self.mnist.len

        self.n_hidden = 500
        self.n_z = 20
        self.batchsize = 121
        self.w = 128

        self.images = tf.placeholder(tf.float32, [None, self.w ** 2])
        image_matrix = tf.reshape(self.images, [-1, self.w, self.w, 1])
        temp1, self.c1 = conv2d(image_matrix, 1, 16, "c1") # self.wxself.wx1 -> 64x64x16
        d_h1 = lrelu(temp1)
        temp2, self.c2 = conv2d(d_h1, 16, 32, "c2") # 64x64x16 -> 7x7x32
        d_h2 = lrelu(temp2)
        h2_flat = tf.reshape(d_h2,[self.batchsize, 32**3])

        z_mean, self.mean = dense(h2_flat, 32**3, self.n_z, "mean")
        z_stddev, self.stddev = dense(h2_flat, 32**3, self.n_z, "stddev")

        samples = tf.random_normal([self.batchsize,self.n_z],0,1,dtype=tf.float32)
        guessed_z = z_mean + (z_stddev * samples)

        self.generated_images = self.generation(guessed_z)
        generated_flat = tf.reshape(self.generated_images, [self.batchsize, self.w*self.w])

        self.generation_loss = -tf.reduce_sum(self.images * tf.log(1e-8 + generated_flat) + (1-self.images) * tf.log(1e-8 + 1 - generated_flat),1)

        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1,1)
        self.cost = tf.reduce_mean(self.generation_loss + self.latent_loss)
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)

    # decoder
    def generation(self, z):
        with tf.variable_scope("generation"):
            z_develop, _ = dense(z, self.n_z, 32**3, scope='z_matrix')
            z_matrix = tf.nn.relu(tf.reshape(z_develop, [self.batchsize, 32, 32, 32]))
            h1 = tf.nn.relu(conv_transpose(z_matrix, [self.batchsize, 64, 64, 16], "g_h1"))
            h2 = conv_transpose(h1, [self.batchsize, self.w, self.w, 1], "g_h2")
            h2 = tf.nn.sigmoid(h2)

        return h2

    def train(self):
        visualization = self.mnist.next_batch(self.batchsize)
        reshaped_vis = visualization.reshape(self.batchsize,self.w,self.w)
        ims("results/base.jpg",merge(reshaped_vis[:64],[8,8]))
        # train
        saver = tf.train.Saver(max_to_keep=2)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for epoch in range(1):
                for idx in range(int(self.n_samples / self.batchsize) -1):
                    batch = self.mnist.next_batch(self.batchsize)
                    #print("batch-size", batch)
                    _, gen_loss, lat_loss = sess.run((self.optimizer, self.generation_loss, self.latent_loss), feed_dict={self.images: batch})
                    # dumb hack to print cost every epoch
                    if idx % (self.n_samples) == 1:
                        print("epoch %d: genloss %f latloss %f" % (epoch, np.mean(gen_loss), np.mean(lat_loss)))
                        saver.save(sess, os.getcwd()+"/training/train")
                        self.c1.save(sess, os.getcwd()+"/training/c1")
                        self.c2.save(sess, os.getcwd()+"/training/c2")
                        self.stddev.save(sess, os.getcwd()+"/training/stddev")
                        self.mean.save(sess, os.getcwd()+"/training/mean")
                        generated_test = sess.run(self.generated_images, feed_dict={self.images: visualization})
                        generated_test = generated_test.reshape(self.batchsize,self.w,self.w)
                        ims("results/"+str(epoch)+".jpg",merge(generated_test[:64],[8,8]))


model = LatentAttention()
model.train()
