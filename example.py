import numpy as np
from classifiers import *
from pipeline import *
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys

# 1 - Load the model and its pretrained weights
mesoInc4 = MesoInception4()
#mesoInc4.load('weights/MesoInception_DF.h5')
mesoInc4.load('weights/MesoInception_F2F.h5')

# 2 - Minimal image generator
# We did use it to read and compute the prediction by batchs on test videos
# but do as you please, the models were trained on 256x256 images in [0,1]^(n*n)
dataGenerator = ImageDataGenerator(rescale=1./255)
generator = dataGenerator.flow_from_directory(
    'faceforensics_Dataset/train_dataset/train_images',
    target_size=(256, 256),
    batch_size= 75 ,
    class_mode='binary',
    subset='training', 
    shuffle = True) # turn shuffle off so that the order of loaded files will not change

# 3 - Training/redicting
#Put the deepfake and real image datasets here
deepfake_directory = 'faceforensics_Dataset/train_dataset/train_images/manipulated'
nondeepfake_directory = 'faceforensics_Dataset/train_dataset/train_images/real'

total_length = len(os.listdir(deepfake_directory))+ len(os.listdir(nondeepfake_directory))

# Properly reset the generator before training 
generator.reset()
y_predict =[]
x_predict = []

# range number changes depending on the batch_size * number ofimages = total images
for i in range(total_length // 75):
    x_test, y_test = next(generator) # gets 1 batch of results
    x_predict.extend(x_test)
    y_predict.extend(y_test)

# turn the list into an array to make predictions
x_prediction= np.asarray(x_predict) 
y_prediction = np.asarray(y_predict)
prediction = mesoInc4.predict(x_prediction)

#run for a standard 20 epochs 
mesoInc4.model.fit(x_prediction,y_prediction,batch_size = None,epochs = 20, verbose ='auto')


