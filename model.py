import os
import csv
import matplotlib.image as mpimg

samples_ = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile,skipinitialspace=True)
    for line in reader:
        samples_.append(line)

# samples_2 = []
# with open('./udacity_data/driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile, skipinitialspace=True)
#     for line in reader:
#         samples_2.append(line)
#
# samples2 = []
# for line in samples_2[1:]:
#     samples2.append(line)

samples = []
for line in samples_[1:]:
    samples.append(line)




from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print("{} Training data and {} Val data".format(len(train_samples), len(validation_samples)))

# Crop image to remove the sky and driving deck, resize to 64x64 dimension
def crop_resize(image):
    cropped = cv2.resize(image[60:140, :], (64,64))
    return cropped

import cv2
import numpy as np
import sklearn
import random
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.activations import relu, softmax
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, ELU
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2
import math

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for line in batch_samples:
                center_angle = float(line[3])
                if center_angle == 0:
                    continue
                # for i in range(3):
                #     name = 'data/IMG/'+line[i].split('/')[-1]
                #     image = crop_resize(.imread(name))
                #     images.append(image)

                # For windows system
                for i in range(3):
                    token = line[i].split('\\')
                    # token.encode("string_escape").split('\\')
                    name = 'data/IMG/'+token[-1]
                    image = mpimg.imread(name)
                    images.append(image)

                correction = 0.2
                angles.append(center_angle)
                left_angle = center_angle + correction
                angles.append(left_angle)
                right_angle = center_angle - correction
                angles.append(right_angle)

                # Flip image on horizontal axis
                flip = random.randint(0,1)
                if flip == 1:
                    # For windows system
                    for i in range(3):
                        token = line[i].split('\\')
                        # token.encode("string_escape").split('\\')
                        name = 'data/IMG/'+token[-1]
                        image = mpimg.imread(name)
                        images.append(image)

                    # for i in range(3):
                    #     name = 'data/IMG/'+line[i].split('/')[-1]
                    #     image = crop_resize(mpimg.imread(name))
                    #     image = cv2.flip(image, 1)
                    #     images.append(image)

                    correction = 0.2
                    center_angle = float(line[3]) * -1
                    angles.append(center_angle)
                    left_angle = (center_angle + correction) * -1
                    angles.append(left_angle)
                    right_angle = (center_angle - correction) * -1
                    angles.append(right_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def generator_val(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for line in batch_samples:
                # for i in range(3):
                #     name = 'data/IMG/'+line[i].split('/')[-1]
                #     center_image = crop_resize(mpimg.imread(name))
                #     images.append(center_image)

                # For windows system
                for i in range(3):
                    token = line[i].split('\\')
                    # token.encode("string_escape").split('\\')
                    name = 'data/IMG/'+token[-1]
                    image = mpimg.imread(name)
                    # image = crop_resize(mpimg.imread(name))
                    images.append(image)

                correction = 0.25
                center_angle = float(line[3])
                angles.append(center_angle)
                angles.append(center_angle + correction)
                angles.append(center_angle - correction)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=128)
validation_generator = generator_val(validation_samples, batch_size=128)

for i in range(1):
    gen = next(train_generator)
    print(len(gen[0]))

def resize_normalize(image):
    from keras.backend import tf as ktf
    # resize to width 200 and high 66 liek recommended
    # in the nvidia paper for the used CNN
    # image = cv2.resize(image, (66, 200)) #first try
    resized = ktf.image.resize_images(image, (64, 64))
    #normalize 0-1
    resized = resized/255.0 - 0.5
    return resized

# input_shape = (64,64,3)  # Trimmed image format
nb_samples_per_epoch = len(train_samples)*3

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Cropping2D(cropping=((60,20), (1,1)), input_shape=(160,320,3)))
model.add(Lambda(resize_normalize, input_shape=(160, 320, 3)))
model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample =(2,2), W_regularizer = l2(0.001)))
model.add(ELU())
model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample =(2,2), W_regularizer = l2(0.001)))
model.add(ELU())
model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample = (2,2), W_regularizer = l2(0.001)))
model.add(ELU())
model.add(Convolution2D(64, 3, 3, border_mode='same', subsample = (2,2), W_regularizer = l2(0.001)))
model.add(ELU())
model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample = (2,2), W_regularizer = l2(0.001)))
model.add(ELU())
model.add(Flatten())
model.add(Dense(80, W_regularizer = l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(40, W_regularizer = l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(16, W_regularizer = l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(10, W_regularizer = l2(0.001)))
model.add(Dense(1, W_regularizer = l2(0.001)))
adam = Adam(lr = 0.0001)
model.compile(optimizer= adam, loss='mse', metrics=['accuracy'])
model.summary()

model_history = model.fit_generator(train_generator, samples_per_epoch=nb_samples_per_epoch, validation_data=validation_generator, \
            nb_val_samples=len(validation_samples), nb_epoch=10)

print("Done with training. ")

## Save the model and weights
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.save("model.h5")
print("Model save to disk")

# import matplotlib.pyplot as plt
# ### print the keys contained in the history object
# print(model_history.history.keys())
#
# ### plot the training and validation loss for each epoch
# plt.plot(model_history.history['loss'])
# plt.plot(model_history.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()
