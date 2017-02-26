"""
Utility Functions and Model Building:

The following Python script contains the parts of:

Reading and Balancing Data

Functions for pre-processing images

Model Building - NVIDIA End to End CNN Model

Training the model on the data

Storing the trained model to an H5 file

"""
import math
import random

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda, ELU, Dropout
from keras.activations import relu, softmax
from keras.layers.convolutional import Convolution2D
from keras.optimizers import  Adam
from keras.regularizers import l2
from keras.layers.pooling import MaxPooling2D

import numpy as np
import pandas as pd

import cv2
import matplotlib.image as mpimg

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import tensorflow as tf

# Flag variables
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('adjustment_value', 0.23, "Apparent steering angle shift between right/left vs center camera")
flags.DEFINE_integer('epochs', 25, "Epochs for training")
flags.DEFINE_integer('batch_size', 200, "Batch size")

"""
Reading Data:

Here, I read the CSV file containing the image names (left, right and center)
with the additional data (throttle, speed, break and steering angle)

I also infuse another small data set that exclusively contains sharp turn samples.

The writeup explains why this is done through visualizations, and a reference to
the source of this data
"""
header_row = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
udacity_data = pd.read_csv('driving_log.csv', skiprows=[0], names=header_row)
sharp_turn_data = pd.read_csv('sharp_turn.csv')

"""
sharp_turn_data contains ~ 750 images. I tried using all of these images and
that resulted in very wobbly driving and going off the road. That could have
been avoided using multiple other ways, but taking only ~ 50% of the sharp turns
data solved the problem.
"""

# Combine the two data sources ...
data = pd.concat([udacity_data, sharp_turn_data])

# Separate the center image + steering angle from the rest of the data
center_img = data.center.tolist()
steering_angle = data.steering.tolist()

"""
Shuffle and split the data into 90% training, 10% validation:

This is not necessary since our real performance metric is in the simulator,
but this could be used for early stopping ...
"""

center_img, steering_angle = shuffle(center_img, steering_angle)
center_img, X_valid, steering_angle, y_valid = train_test_split(center_img
                                                                , steering_angle
                                                                , test_size=0.10
                                                                , random_state=1)

"""
Split the data into right, left and straight ...

A positive angle turns from left to right
A negative angle turns from right to left
All angles are normalized to be between -1.0 to 1.0

Here, we choose straight driving to be when the steering angle is
between -0.15 to 0.15 degrees.
"""
straight_img, left_img, right_img = [], [], []
straight_angle, left_angle, right_angle = [], [], []

for angle in steering_angle:
    index = steering_angle.index(angle)
    if angle > 0.15:
        right_img.append(center_img[index])
        right_angle.append(angle)
    if angle < -0.15:
        left_img.append(center_img[index])
        left_angle.append(angle)
    else:
        straight_img.append(center_img[index])
        straight_angle.append(angle)

"""
To balance the data set to a reasonable degree, and limit the impact of the dominance of
some values for the steering angle, such as straight driving or more positive angles than
negative ones, I limit the amount of images from each category (left, right and straight)
to some extent. The values below are what worked for me...
"""
# Keep left images as they are ...
right_img = right_img[0:500]
right_angle = right_angle[0:500]
straight_img = straight_img[0:700]
straight_angle = straight_angle[0:700]

"""
Data Recovery from Right and Left Cameras

Motivation: When we have much more data in one category than the others, this step
would introduce needed balance to the data. But in our case, after I limited the amount
of data from the dominant class in the previous step, and the difference between the number
of images between the dominant class w.r.t the others decreased, the resulting improvement
would be negligible...

I am only including it for future reference, in case I decide to train my model on more
data later.

"""

straight_size = len(straight_img)
left_size = len(left_img)
right_size = len(right_img)

recovery_angles = data.steering.tolist()
data_size = math.ceil(len(recovery_angles))

# Difference between (straight and right) and (straight and left)
extra_left_img = straight_size - left_size
extra_right_img = straight_size - right_size

# Generate random list of indices for left and right recovery images
# left_indexes_random = random.sample(range(data_size), extra_left_img)
# right_indexes_random = random.sample(range(data_size), extra_right_img)

# Get the left and right camera images from the original data
right_orig_img = data.right.tolist()
left_orig_img = data.left.tolist()

for i in range(data_size):
    if recovery_angles[i] < -0.15:
        left_img.append(right_orig_img[i])
        left_angle.append(recovery_angles[i] - FLAGS.adjustment_value)

# Since we have less images going to the left, I do this 3 times
for i in range(2):
    for i in range(data_size):
        if recovery_angles[i] > 0.25 and recovery_angles[i] < 0.90:
            right_img.append(left_orig_img[i])
            right_angle.append(recovery_angles[i] + FLAGS.adjustment_value)

# Combine the right, left and center images/angles to get the full training set
X_train = straight_img + left_img + right_img
y_train = np.float32(straight_angle + left_angle + right_angle)


"""
Pre-Processing Utility Functions:

Used by the Generator for Keras to generate images on the go. For each image, we introduce
a (1) random brightness change, (2) a flip (mirroring image along the y axis and multiplying the
steering angle by -1) and (3) cropping the image to contain only the pixels within the spatial
window necessary for detecting the lanes.

This will allow the model to adapt to different levels of brightness and increase the data.

"""


def random_brightness(img):
    """
    Convert the image to HSV, and multiply the brightness channel [:,:,2]
    by a random number in the range of [0.25 to 1.0] to get different levels of
    brightness.

    :param img: normalized image in RGB color space
    :return: new_img: Image in RGB Color space
    """
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    rand = random.uniform(0.25,1.0)
    hsv_img[:, :, 2] = rand*hsv_img[:, :, 2]
    new_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    return new_img


def flip(img, steer_angle):
    """

    :param img: camera input
    :param steer_angle: steering angle
    :return: new_img: Flipped, along the y axis.
             new_angle: steer_angle multiplied by -1
    """
    new_img = cv2.flip(img, 1)
    new_angle = steer_angle*(-1)
    return new_img, new_angle


def crop_resize(img):
    """

    :param img: original image
    :return: cropped: img - without the sky part, resized to fit the input
                        size requirement for the CNN
    """
    cropped = cv2.resize(img[60:140, :], (64, 64))
    return cropped


"""
Python Generators:

Training:

    Prepare a matrix to fit batch_size images of size (64,64,3)
    and a matrix to fit batch_size steering angles.

    Shuffle the training data

    Construct a batch on the fly by:
        batch_size times:
            randomly choose a training image.
            Apply a random brightness change
            Crop it
            Flip a coin to decide whether to flip it or not
        Return the batch!

Validation:

    The same as training, except do not apply random brightness change
    neither flip the image!

"""


def generator_data(batch_size):
    batch_train = np.zeros((batch_size, 64, 64, 3), dtype=np.float32)
    batch_angle = np.zeros((batch_size,), dtype=np.float32)

    while True:
        data, angle = shuffle(X_train, y_train)
        for i in range(batch_size):
            choice = int(np.random.choice(len(data), 1))
            batch_train[i] = crop_resize(random_brightness(mpimg.imread(data[choice].strip())))
            batch_angle[i] = angle[choice] * (1 + np.random.uniform(-0.05, 0.05))
            flip_coin = random.randint(0, 1)
            if flip_coin == 1:
                batch_train[i], batch_angle[i] = flip(batch_train[i], batch_angle[i])
        yield batch_train, batch_angle


def generator_valid(data, angle, batch_size):
    batch_train = np.zeros((batch_size, 64, 64, 3), dtype=np.float32)
    batch_angle = np.zeros((batch_size,), dtype=np.float32)
    while True:
        data, angle = shuffle(data, angle)
        for i in range(batch_size):
            rand = int(np.random.choice(len(data), 1))
            batch_train[i] = crop_resize(mpimg.imread(data[rand].strip()))
            batch_angle[i] = angle[rand]
        yield batch_train, batch_angle

"""
Model Building and Training
"""


def main(_):
    data_generator = generator_data(FLAGS.batch_size)
    valid_generator = generator_valid(X_valid, y_valid, FLAGS.batch_size)
    # Image size
    input_shape = (64, 64, 3)
    # Dropout probability
    dropout = 0.5

    model = Sequential()
    # Normalize image
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
    # 5 Colvolutional Layers with ReLu activations
    # The first 3 layers have [2,2] subsampling
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), input_shape=input_shape, activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    # Flatten the output of the last conv layer
    model.add(Flatten())
    model.add(Dropout(dropout))
    # 4 Fully connected layers, ReLu activation and Dropout
    model.add(Dense(1164, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    # Output layer
    model.add(Dense(1, activation='tanh'))
    learning_rate = 0.0001
    opt = Adam(lr=learning_rate)
    model.compile(optimizer=opt, loss='mean_squared_error')
    model.summary()

    # Run model training on the data generated in batches by data_generator method
    model.fit_generator(data_generator
                        , samples_per_epoch=math.ceil(len(X_train))
                        , nb_epoch=FLAGS.epochs
                        , validation_data=valid_generator
                        , nb_val_samples=len(X_valid)
                        )

    print('Done Training')

    # Save model to disk
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("Model Saved!")

if __name__ == '__main__':
    tf.app.run()
