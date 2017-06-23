import csv
import cv2
import numpy as np

FILE_TO_OPEN = 'udacity_data/driving_log.csv'

lines = []
with open(FILE_TO_OPEN) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


images = []
mesurements = []
for line in lines[1:]:
    filename = line[0].split('/')[-1]
    current_path = 'udacity_data/IMG/' + filename
    image = cv2.imread(current_path)
    assert image is not None
    images.append(image)

    mesurement = float(line[3])
    mesurements.append(mesurement)

X_train = np.array(images)
Y_train = np.array(mesurements)

from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda

model = Sequential()
model.add(Lambda(lambda x:(x/255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('basic_model.h5')
