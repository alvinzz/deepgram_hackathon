import tensorflow as tf
import numpy as np
# import input_data
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import os
from scipy.misc import imsave as ims
from utils import *
from ops import *
from load_img_label import Data

class LatentAttention():
    def __init__(self):
       # self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
       # self.n_samples = self.mnist.train.num_examples
        #self.mnist = Data("../EEGdata/train5/images/")
        self.mnist = Data("../EEGdata/train/images/")
        self.n_samples = self.mnist.len

        self.c1_W = tf.get_variable("c1w", [5,5,1,16], trainable=False)
        self.c1_b = tf.get_variable("c1b", [16], trainable=False)
        with tf.Session() as sess:
            restore = tf.train.Saver([self.c1_W, self.c1_b])
            restore.restore(sess, "../EEGnet/training/c1")

        self.c2_W = tf.get_variable("c2w", [5,5,16,32], trainable=False)
        self.c2_b = tf.get_variable("c2b", [32], trainable=False)
        with tf.Session() as sess:
            restore = tf.train.Saver([self.c2_W, self.c2_b])
            restore.restore(sess, "../EEGnet/training/c2")

        self.mean_W = tf.get_variable("meanW", [32**3, 20], trainable=False)
        self.mean_b = tf.get_variable("meanb", [20], trainable=False)
        with tf.Session() as sess:
            restore = tf.train.Saver([self.mean_W, self.mean_b])
            restore.restore(sess, "../EEGnet/training/mean")

        self.stddev_W = tf.get_variable("stddevW", [32**3, 20], trainable=False)
        self.stddev_b = tf.get_variable("stddevb", [20], trainable=False)
        with tf.Session() as sess:
            restore = tf.train.Saver([self.stddev_W, self.stddev_b])
            restore.restore(sess, "../EEGnet/training/stddev")

        self.n_z = 20
        self.batchsize = 121
        self.w = 128

        self.images = tf.placeholder(tf.float32, [None, self.w ** 2])
        self.labels = tf.placeholder(tf.float32, [None, 72])
        image_matrix = tf.reshape(self.images,[-1, self.w, self.w, 1])
        z_mean, z_stddev = self.recognition(image_matrix)

        self.latent = tf.concat([z_mean, z_stddev], axis=1)
        self.preds = tf.add(tf.matmul(self.latent, tf.Variable(tf.random_normal([40, 72]))), tf.Variable(tf.random_normal([72])))

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.preds, labels=self.labels))
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)

    # encoder
    def recognition(self, input_images):
        with tf.variable_scope("recognition"):
            h1 = lrelu(tf.add(
                tf.nn.conv2d(input_images, self.c1_W, strides=[1, 2, 2, 1], padding='SAME'), self.c1_b)) # self.wxself.wx1 -> 64x64x16
            h2 = lrelu(tf.add(
                tf.nn.conv2d(h1, self.c2_W, strides=[1,2,2,1], padding='SAME'), self.c2_b)) # 64x64x16 -> 7x7x32
            h2_flat = tf.reshape(h2,[self.batchsize, 32**3])

        return tf.matmul(h2_flat, self.mean_W) + self.mean_b, tf.matmul(h2_flat, self.stddev_W) + self.stddev_b

    # decoder
    def generation(self, z):
        with tf.variable_scope("generation"):
            z_develop = dense(z, self.n_z, 32**3, scope='z_matrix')
            z_matrix = tf.nn.relu(tf.reshape(z_develop, [self.batchsize, 32, 32, 32]))
            h1 = tf.nn.relu(conv_transpose(z_matrix, [self.batchsize, 64, 64, 16], "g_h1"))
            h2 = conv_transpose(h1, [self.batchsize, self.w, self.w, 1], "g_h2")
            h2 = tf.nn.sigmoid(h2)

        return h2

    def train(self):
        #_,visualization = self.mnist.next_batch_not_modify(self.batchsize)
        #reshaped_vis = visualization.reshape(self.batchsize,self.w,self.w)
        #ims("results/base.jpg",merge(reshaped_vis[:64],[8,8]))
        # train
        saver = tf.train.Saver(max_to_keep=2)
        restore = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for epoch in range(100):
                for idx in range(int(self.n_samples / self.batchsize) -1):
                    im, label = self.mnist.next_batch(self.batchsize)
                    #print("batch-size", )
                    _, loss = sess.run((self.optimizer, self.cost), feed_dict={self.images: im, self.labels: label})
                    # dumb hack to print cost every epoch
                    if idx % (self.n_samples) == 1:
                        print("epoch %d: loss %f" % (epoch, np.mean(loss)))
                        saver.save(sess, os.getcwd()+"/training/train")

model = LatentAttention()
model.train()
