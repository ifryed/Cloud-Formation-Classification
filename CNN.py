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
from tensorflow import keras
from tqdm import tqdm
import time

from utils import prepareData

NAME = "clouds recognition{}".format(int(time.time()))


def main():
    DATADIR = "data/mini_data"
    CATEGORIES = os.listdir(DATADIR)
    img_size = 256
    train_x, test_x, train_y, test_y = prepareData(img_folder=DATADIR, img_size=img_size, sample_size=10)
    epoch = len(train_x)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv3D(96, (5, 5, train_x[0].shape[2]), input_shape=train_x[0].shape, activation='relu',
                               padding='same'),
        # tf.keras.layers.Conv3D(96, (1, 1, train_x[0].shape[2]), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2)),

        tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2)),

        tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(len(CATEGORIES), activation='softmax')
    ])
    initial_learning_rate = 1e-5
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=epoch * 10,
        decay_rate=.1,
        staircase=True)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())

    log_dir = os.path.join("tf_logs\\CNN\\", datetime.now().strftime("%Y%m%d-%H%M%S/"))
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch=0)
    save_callback = keras.callbacks.ModelCheckpoint(log_dir, monitor='val_accuracy', verbose=True, save_best_only=True,
                                                    save_weights_only=False, mode='max', save_freq='epoch')

    model.fit(x=train_x,
              y=train_y,
              batch_size=8,
              epochs=100,
              validation_data=(test_x, test_y),
              callbacks=[tensorboard_callback,
                         save_callback])


if __name__ == "__main__":
    if 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
        config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    main()
