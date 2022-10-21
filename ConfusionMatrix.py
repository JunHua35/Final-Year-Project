import numpy as np
from classifiers import *
from pipeline import *
import tensorflow as tf
import os

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
import math

# 1 - Load the model and its pretrained weights
mesoInc4 = MesoInception4()
mesoInc4.load('weights/MesoInception_F2F.h5')

main_directory = 'faceforensics_Dataset_classified/skintone_balanced/fairs'

# 2 - Minimal image generator
# We did use it to read and compute the prediction by batchs on test videos
# but do as you please, the models were trained on 256x256 images in [0,1]^(n*n)
dataGenerator = ImageDataGenerator(rescale=1./255)
generator = dataGenerator.flow_from_directory(
    main_directory,
    target_size=(256, 256),
    batch_size=75,
    class_mode='binary',
    subset='training',
    shuffle= False)  # turn shuffle off so that the order of loaded files will not change

deepfake_directory = 'faceforensics_Dataset_classified/skintone_balanced/fairs/fair_manipulated'
nondeepfake_directory = 'faceforensics_Dataset_classified/skintone_balanced/fairs/fair_real'

total_length = len(os.listdir(deepfake_directory))+ len(os.listdir(nondeepfake_directory))

# 3 - Predict
# Properly reset the generator before predicting
generator.reset()
y_predict =[]
x_predict = []

range_number = math.ceil(total_length/75)  # range number changes depending on the batch_size * number ofimages = total images

for i in range(range_number):
    x_test, y_test = next(generator) # gets 1 batch of results
    x_predict.extend(x_test)
    y_predict.extend(y_test)

# turn the list into an array to make predictions
x_prediction= np.asarray(x_predict)
y_prediction = np.asarray(y_predict)
prediction = mesoInc4.predict(x_prediction)

# 4 - compute confusion matrix according to predicted and actual classes
conf_mat = confusion_matrix(prediction.round(), y_prediction)
try:
    if not total_length == np.sum(conf_mat):
        print("Confusion matrix does not add up to total samples")
    # accuracy is calculated using this formula: tp+tn/tp+tn+fp+fn
    acc = np.sum(conf_mat.diagonal()) / np.sum(conf_mat)
    precision = precision_score(prediction.round(), y_prediction)
    recall = recall_score(prediction.round(), y_prediction)
    print(conf_mat)
    print('Overall accuracy: {} %'.format(acc * 100))
    print('Precision: {} %'.format(precision * 100))
    print('Recall: {} %'.format(recall * 100))
except Exception as e:
    print(e)
    print("Confusion matrix is not successfully computed")


