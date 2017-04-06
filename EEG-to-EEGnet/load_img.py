#READ IN IMAGES FOR LABELS

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
import os
import sys

IMDIR = "./img/"

def load_img(imdir=IMDIR):

    filenames = []
    for file in os.listdir(imdir):
        if file.endswith(".png"):
            filenames.append(imdir + file)

    img_tf = []
    for file in filenames:
        img = imresize(imread(file, flatten=True), (128, 128))
        img_tf.append(np.reshape(img, (128**2,)) / (255.))

    img_tf = np.array(img_tf)

    return img_tf

class Data:
    def __init__(self, imdir):
        self.imdir = imdir
        self.raw = load_img(imdir)
        self.len = len(self.raw)
        self.last = 0

    def next_batch(self, batch_size = 100):
        if self.last + batch_size >= self.len:
            self.last = self.len - batch_size
        ret = self.raw[self.last:self.last + batch_size]
        self.last += batch_size
        if self.last == self.len:
            self.last = 0
        return ret
