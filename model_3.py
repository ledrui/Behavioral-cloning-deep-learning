import os
import cv2
import csv
import h5py
import json
import math
import numpy as np
from pathlib import Path
import tensorflow as tf
from keras.models import Sequential
# from keras.utils.visualize_util import plot
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from img_tools import *

# tf.python.control_flow_ops = tf
np.random.seed(123) #For reproducability

FILE_TO_OPEN='udacity_data/driving_log.csv'
IMG_SIZE_X=320 #Input image size
IMG_SIZE_Y=160
IMG_PROC_X=128 #Scaled image size
IMG_PROC_Y=64
SHIFT_CUR = 0.28

def proc_img(img): # input is 160x320x3
    img = img[59:138:2, 0:-1:2, :] # select vertical region and take each second pixel to reduce image dimensions
    img = (img / 127.5) - 1.0 # normalize colors from 0-255 to -1.0 to 1.0
    return img # return 40x160x3 image

#Input .csv file read
col=[[] for x in range(7)]
with open(FILE_TO_OPEN) as f:
	reader = csv.reader(f, skipinitialspace=True)
	for row in reader:
		for (i,v) in enumerate(row):
			col[i].append(v)
train_img_center_paths=col[0]
train_img_left_paths=col[1]
train_img_right_paths=col[2]
y_train=[]
angles = col[3]
print(angles[:10])
for x in angles[1:]:

	if x != 0:
		y_train.append(float(x)) #Convert steering angles into float


print(y_train[:10])

#Batch generator of augmented images for training
def generator_train(batch_size = 256):
	X_batch = np.zeros((batch_size, IMG_PROC_Y, IMG_PROC_X, 6))
	y_batch = np.zeros(batch_size)
	while True:
		for i in range(batch_size):
			ind = np.random.randint(len(y_train)) #Random frame
			ind2 = ind-5 #index of the second, shifted back in time, frame
			if ind2<0:
				ind2 = 0
			y = y_train[ind]
			cam = np.random.randint(3) #Camera choose from the frame
			if cam==1:
				img_p = train_img_left_paths[ind]
				img_p2 = train_img_left_paths[ind2]
				y += SHIFT_CUR
			elif cam==2:
				img_p = train_img_right_paths[ind]
				img_p2 = train_img_right_paths[ind2]
				y -= SHIFT_CUR
			else:
				img_p = train_img_center_paths[ind]
				img_p2 = train_img_center_paths[ind2]
			f = np.random.randint(2) #if we need to flip the image
			if f==1:
				y = -y
			img = augment_img(img_p, IMG_PROC_X, IMG_PROC_Y, f)
			img2 = augment_img(img_p2, IMG_PROC_X, IMG_PROC_Y, f)
			img_conc=np.concatenate((img, img2), axis=2)
			X_batch[i] = img_conc
			y_batch[i] = y
		yield X_batch, y_batch

#Batch generator of images for validation
def generator_val(batch_size = 256):
	X_batch = np.zeros((batch_size, IMG_PROC_Y, IMG_PROC_X, 6))
	y_batch = np.zeros(batch_size)
	while True:
		for i in range(batch_size):
			ind = np.random.randint(len(y_train))
			y = y_train[ind]
			ind2 = ind-5
			if ind2<0:
				ind2 = 0
			img_p = train_img_center_paths[ind]
			img = val_img(img_p, IMG_PROC_X, IMG_PROC_Y)
			img_p2 = train_img_center_paths[ind2]
			img2 = val_img(img_p2, IMG_PROC_X, IMG_PROC_Y)
			img_conc=np.concatenate((img, img2), axis=2)
			X_batch[i] = img_conc
			y_batch[i] = y
		yield X_batch, y_batch

#Model saver
def saver(model_file, weight_file):
	if Path(model_file).is_file():
		os.remove(model_file)
	if Path(weight_file).is_file():
		os.remove(weight_file)
	with open(model_file,'w' ) as f:
		json.dump(model.to_json(), f)
	model.save_weights(weight_file)
	print("Model saved", model_file, weight_file)

#Model difinition
model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(IMG_PROC_Y, IMG_PROC_X, 6))) #64x128x6
model.add(Convolution2D(64, 7, 7, input_shape=(IMG_PROC_Y, IMG_PROC_X, 6), activation='relu', border_mode='same')) #64x128x64
model.add(Convolution2D(128, 3, 3, input_shape=(IMG_PROC_Y, IMG_PROC_X, 64), activation='relu', border_mode='same', subsample=(2,2))) #32x64x128
model.add(MaxPooling2D((2, 2))) #16x32x128
model.add(Dropout(0.5))
model.add(Convolution2D(256, 3, 3, input_shape=(IMG_PROC_Y/4, IMG_PROC_X/4, 128), activation='relu', border_mode='same')) #16x32x256
model.add(Convolution2D(64, 1, 1, input_shape=(IMG_PROC_Y/4, IMG_PROC_X/4, 256), activation='relu', border_mode='same')) #16x32x64
model.add(MaxPooling2D((4, 8))) #4x4x64
model.add(Dropout(0.5))
model.add(Flatten()) #1024
model.add(Dense(512, activation='relu')) #512
model.add(Dense(128, activation='relu')) #128
model.add(Dense(1))
#Model summary and plot output
model.summary()
# plot(model, to_file='model.png', show_shapes=True)
#Model compilation, training and saving
model.compile('Adam', loss='mean_squared_error')
model.fit_generator(generator_train(), samples_per_epoch=32768, nb_epoch=2, validation_data=generator_val(), nb_val_samples=4096)
saver("model.json", "model.h5")
