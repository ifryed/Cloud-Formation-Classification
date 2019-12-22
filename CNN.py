from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from datetime import datetime
from sklearn.model_selection import train_test_split
import keras
from tqdm import tqdm
import time

NAME = "clouds recognition{}".format(int(time.time()))


def main():
    DATADIR = "data/mini_data"

    CATEGORIES = ["Fish", "Flower", "Gravel", "Sugar"]
    training_data = []
    for category in CATEGORIES:

        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat
        c = -100
        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                img_array = cv2.resize(img_array, (64, 64))
                img_h, img_w = img_array.shape
                img_array = img_array.reshape((img_h, img_w, 1))
                training_data.append([img_array, class_num])  # add this to our training_data
                c -= 1
                if c == 0:
                    break
            except IOError as e:  # in the interest in keeping the output clean...
                pass

    img_h, img_w = training_data[0][0].shape[:-1]
    np.random.shuffle(training_data)

    X = []
    y = []

    for features, label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(256, (3, 3), input_shape=(img_h, img_w, 1), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),  # this converts our 3D feature maps to 1D feature vectors
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(CATEGORIES), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    log_dir = os.path.join("tf_logs\\CNN\\", datetime.now().strftime("%Y%m%d-%H%M%S/"))
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch=0)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
    model.fit(x=train_x,
              y=train_y,
              epochs=50,
              validation_data=(test_x, test_y),
              callbacks=[tensorboard_callback])


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
main()
