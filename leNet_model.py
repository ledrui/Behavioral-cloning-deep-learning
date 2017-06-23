import csv
import cv2
import numpy as np
from scipy.misc import imresize



FILE_TO_OPEN = 'udacity_data/driving_log.csv'
SHIFT_CUR = .28

def flip_image(X_train, Y_train):
    # X_flipped = np.ndarray(shape=X_train_1.shape)
    X_flipped = []
    count = 0
    for i in range(len(X_train_1)):
        X_flipped.append(np.fliplr(X_train_1[i]))
        count += 1
    X_flipped = np.array(X_flipped)

    Y_flipped = Y_train_1 * -1

    return X_flipped, Y_flipped

lines = []
with open(FILE_TO_OPEN) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# image_data = []
#
# for line in lines[1:2]:
#     image_path_center = 'udacity_data/IMG/' + line[0].split('/')[-1]
#     image_path_left = 'udacity_data/IMG/' + line[1].split('/')[-1]
#     image_path_right = 'udacity_data/IMG/' + line[2].split('/')[-1]
#     print(image_path_center)
#     print(image_path_left)
#     print(image_path_right)
#
#     image_center = cv2.imread(image_path_center)
#     image_left = cv2.imread(image_path_left)
#     image_right = cv2.imread(image_path_right)
#
#     image_data.extend(image_center)
#     image_data.extend(image_left)
#     image_data.extend(image_right)
#
# print(image_data)

from sklearn.model_selection import train_test_split
training_samples, validation_samples = train_test_split(lines, test_size=.02)

def generator(samples, batch_size=32):
    samples_size = len(samples)
    while True:
        shuffle(samples)
        for offset in range(0, samples_size, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            mesurements = []
            for line in batch_samples[1:]:
                image_path_center = 'udacity_data/IMG/' + line[0].split('/')[-1]
                image_path_left = 'udacity_data/IMG/' + line[1].split('/')[-1]
                image_path_right = 'udacity_data/IMG/' + line[2].split('/')[-1]

                image_center = cv2.imread(image_path_center)
                # Resize images for faster training
                # image_center = imresize(image_center, (32,64,3))[12:,:,:]
                image_left = cv2.imread(image_path_left)
                # image_left = imresize(image_left, (32,64,3))[12:,:,:]
                image_right = cv2.imread(image_path_right)
                # image_right = imresize(image_right, (32,64,3))[12:,:,:]

                center_mesurement = float(line[3])
                left_mesurement = float(line[3]) + SHIFT_CUR
                right_mesurement = float(line[3]) - SHIFT_CUR

                image_data.extend(image_center)
                image_data.extend(image_left)
                image_data.extend(image_right)

                mesurements.extend(center_mesurement)
                mesurements.extend(left_mesurement)
                mesurements.extend(right_mesurement)

            X_train = np.array(images)
            y_train = np.array(mesurements)
            yield X_train, y_train

# compile and train the model using the generator function
train_generator = generator(training_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# X_flipped, Y_flipped = flip_image(X_train_1, Y_train_1)
# X_train = np.concatenate((X_train_1, X_flipped), axis=0)
# Y_train = np.concatenate((Y_train_1, Y_flipped), axis=0)


from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, MaxPooling2D, Cropping2D
from keras.layers.convolutional import Convolution2D

model = Sequential()
model.add(Lambda(lambda x:(x/255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50,20),(0,0)), input_shape=(160, 320, 3)))
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch = len(training_samples), validation_data = validation_generator, nb_val_samples = len(validation_samples) , nb_epoch=5)

model.save('leNet.h5')
