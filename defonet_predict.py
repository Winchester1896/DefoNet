# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 15:56:30 2021

@author: zhang.9325
"""


import tensorflow as tf
import cv2
import numpy as np
import os
import shutil
from keras.preprocessing.image import img_to_array



def get_prediction(file):
    model = tf.keras.models.load_model('models/nn_model_o')
    img = cv2.imread('images/' + file)
    img = cv2.resize(img, (108, 108))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float') / 255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    acc = model.predict(img)[0][1]
    return acc


def get_classification(folder):
    model = tf.keras.models.load_model('models/nn_model_o')
    fd = os.listdir('datasets/' + folder)
    outpath1 = 'datasets/' + folder[:-1] + '_1/'
    outpath0 = 'datasets/' + folder[:-1] + '_0/'
    if not os.path.exists(outpath1):
        os.mkdir(outpath1)
    if not os.path.exists(outpath0):
        os.mkdir(outpath0)
    c0 = 0
    c1 = 0
    for file in fd:
        img = cv2.imread('datasets/' + folder + file)
        img = cv2.resize(img, (108, 108))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float') / 255.0
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        acc = model.predict(img)[0][1]
        if acc >= 0.5:
            shutil.copy('datasets/' + folder + file, outpath1)
            c1 += 1
        else:
            shutil.copy('datasets/' + folder + file, outpath0)
            c0 += 1
    print("Total: ",c0 + c1, "defoliated: ", c1, "healthy: ", c0)


# filename = 'image.jpg'
# get_prediction(filename)

foldername = 'sample_images/'
get_classification(foldername)