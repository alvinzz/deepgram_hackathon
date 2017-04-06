#READ IN IMAGES FOR LABELS

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
import os
import sys
import json
import codecs

def load_img(filenames):

    img_tf = []
    for file in filenames:
        img = imresize(imread(file, flatten=True), (128, 128))
        img_tf.append(np.reshape(img, (128**2,)) / (255.))

    img_tf = np.array(img_tf)

    return img_tf


def load_json():
    data = []
    with codecs.open('../EEGdata/train/data.jsonl','rU','utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    #print(data[0])
    dict = {}
    for line in data:
        dict[line['uuid']] = line['precise_label'] - 1
    return dict

class Data:
    def __init__(self, imdir):
        self.filenames = []
        for file in os.listdir(imdir):
            if file.endswith(".png"):
                self.filenames.append(imdir + file)
        self.dict = load_json()
        self.imdir = imdir
        self.len = len(self.filenames)
        self.last = 0

    def next_batch(self, batch_size = 100):
        if self.last + batch_size > self.len:
            self.last = self.len - batch_size
        to_load = self.filenames[self.last:self.last + batch_size]
        self.last += batch_size
        if self.last == self.len:
            self.last = 0
        #print(self.last)
        return load_img(to_load), load_img(["../EEGdata/img/img-" + str(self.dict[name[len(self.imdir):-4]]).zfill(3) + ".png" for name in to_load])

    def next_batch_not_modify(self, batch_size = 100):
        to_load = self.filenames[self.last:self.last + batch_size]
        return load_img(to_load), load_img(["../EEGdata/img/img-" + str(self.dict[name[len(self.imdir):-4]]).zfill(3) + ".png" for name in to_load])
