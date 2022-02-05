# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 15:56:15 2021

@author: zhang.9325
"""


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adagrad
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import numpy as np
import argparse
import os
import keras
from keras import backend as K
from keras.optimizers import Adam
from model import DefoNet
import time
from tensorflow import set_random_seed
import random
import tensorflow as tf


K.tensorflow_backend._get_available_gpus()
seed = 21
random.seed(seed)
np.random.seed(seed)
set_random_seed(seed)
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to output model")
ap.add_argument("-d", "--dataset", type=str, required=True)
ap.add_argument("-e", "--epoches", type=int, default=150)
ap.add_argument("-o", "--output", type=str, default="output.txt")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output loss/accuracy plot")
args = vars(ap.parse_args())




def list_images(path):
    img_list = []
    f = os.listdir(path)
    for i in range(len(f)):
        subf_name = path + '/' + f[i]
        subf = os.listdir(subf_name)
        for j in range(len(subf)):
            img_path = subf_name + '/' + subf[j]
            img_list.append(img_path)
    print(img_list[0])     
    return img_list



# initialize our number of epochs, initial learning rate, and batch
# size
NUM_EPOCHS = args["epoches"]
INIT_LR = 1e-4
BS = 64
IMAGE_SIZE = (108, 108)
BASE_PATH = "datasets/" + args["dataset"]
print(BASE_PATH)
TRAIN_PATH = BASE_PATH + "/training"
VAL_PATH = BASE_PATH + "/validation"
TEST_PATH = BASE_PATH + "/testing"

trainPaths = list_images(TRAIN_PATH)
totalTrain = len(trainPaths)
totalVal = len(list_images(VAL_PATH))
totalTest = len(list_images(TEST_PATH))
         
trainLabels = [int(p.split('/')[-2]) for p in trainPaths]
trainLabels = np_utils.to_categorical(trainLabels)
print(trainLabels)
classTotals = trainLabels.sum(axis=0)
print(classTotals)
classWeight = classTotals.max() / classTotals
class_weight = 1
classWeight[1] = class_weight * classWeight[1]
# classWeight[2] = 64 * classWeight[2]
classWeight[0] = 1
classWeight[1] = 1
print(classWeight)

# initialize the training training data augmentation object
# trainAug = ImageDataGenerator(rescale=1 / 255.0)
trainAug = ImageDataGenerator(
    rescale=1 / 255.0,
    rotation_range=20,
    zoom_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05,
    horizontal_flip=True,
    vertical_flip=True,
    # featurewise_center=True,
    # featurewise_std_normalization=True,
    fill_mode="nearest")

# initialize the validation (and testing) data augmentation object
valAug = ImageDataGenerator(rescale=1 / 255.0)

# initialize the training generator
trainGen = trainAug.flow_from_directory(
    TRAIN_PATH,
    class_mode="categorical",
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    shuffle=True,
    seed=21,
    batch_size=BS)

# initialize the validation generator
valGen = valAug.flow_from_directory(
    VAL_PATH,
    class_mode="categorical",
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    shuffle=True,
    seed=21,
    batch_size=BS)

# initialize the testing generator
testGen = valAug.flow_from_directory(
    TEST_PATH,
    class_mode="categorical",
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    shuffle=False,
    batch_size=BS)


start = time.time()
callback = EarlyStopping(monitor='val_acc', patience=8)
model = DefoNet.build(
    width=IMAGE_SIZE[0], height=IMAGE_SIZE[1],
    depth=3, classes=2,
    finalAct="sigmoid")


print(model.summary())
decay_rate = 1
opt = Adam(lr=INIT_LR, decay=(INIT_LR * decay_rate) / (NUM_EPOCHS * 1), beta_1=0.9, beta_2=0.999, epsilon=1e-8)
model.compile(loss="binary_crossentropy", optimizer=opt, 
              metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
 
# fit the model
H = model.fit_generator(
    trainGen,
    steps_per_epoch=totalTrain // BS,
    validation_data=valGen, 
    validation_steps=totalVal // BS,
    class_weight=classWeight,
    workers=10,
    epochs=NUM_EPOCHS)


with open(args["output"], "a+")as f:
    f.write("==================================================\n")
    f.write("dataset: '{}'\n".format(args["dataset"]))

    # f.write("bad weight: {:.4f}\n".format(classWeight[1]))
    f.write("number of epoches: {:d}\n".format(NUM_EPOCHS))
    f.write("Learning rate: {:f}\n".format(INIT_LR))
    f.write("Decay rate: {:f}\n".format(decay_rate))
    f.write("Defo class weight: {:d}\n".format(class_weight))
    f.write("[INFO] evaluating network...\n")
    testGen.reset()

    if totalTest % BS == 0:
        STEPS = totalTest // BS
    else:
        STEPS = totalTest // BS + 1
    if totalTrain % BS == 0:
        TSTEPS = totalTrain // BS
    else:
        TSTEPS = totalTrain // BS + 1
    predIdxs = model.predict_generator(testGen, steps=STEPS)
    TpredIdxs = model.predict_generator(trainGen, steps=TSTEPS)
    
    # print(len(predIdxs))
    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    predIdxs = np.argmax(predIdxs, axis=1)
    TpredIdxs = np.argmax(TpredIdxs, axis=1)
    # show a nicely formatted classification report
    f.write(classification_report(testGen.classes, predIdxs,
                                target_names=testGen.class_indices.keys()))

    # save the network to disk
    f.write("[INFO] serializing network to '{}'...\n".format(args["model"]))
    model.save(args["model"])

    # compute the confusion matrix and and use it to derive the raw
    # accuracy, sensitivity, and specificity
    cm = confusion_matrix(testGen.classes, predIdxs)
    tcm = confusion_matrix(trainGen.classes, TpredIdxs)
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total
    Precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
    Recall = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    f.write("confusion matrix: \n")
    f.write("[[ {:d}     {:d}]\n".format(cm[0, 0], cm[0, 1]))
    f.write(" [ {:d}     {:d}]]\n".format(cm[1, 0], cm[1, 1]))
    # f.write(" [ {:d}     {:d}     {:d} ]]\n".format(cm[2, 0], cm[2, 1], cm[2, 2]))
    # show the confusion matrix, accuracy, sensitivity, and specificity
    print(cm)
    # print(tcm)
    f.write("acc: {:.4f}\n".format(acc))
    f.write("Precision: {:.4f}\n".format(Precision))
    f.write("Recall: {:.4f}\n".format(Recall))
    end = (time.time() - start) / 60.0
    f.write("training time: {:.4f}\n".format(end))
    f.write("[INFO] saving figure as '{}'\n".format(args["plot"]))
f.close()

# plot the training loss and accuracy
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure(figsize=(16,10))
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.plot(np.arange(0, N), H.history["precision"], label="train_precision")
plt.plot(np.arange(0, N), H.history["val_precision"], label="val_precision")
plt.plot(np.arange(0, N), H.history["recall"], label="train_recall")
plt.plot(np.arange(0, N), H.history["val_recall"], label="val_recall")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
plt.savefig(args["plot"])










