from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import cv2
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from datetime import datetime
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import time

NAME = "clouds recognition{}".format(int(time.time()))


def main():
    DATADIR = "data/mini_data"

    CATEGORIES = ["Fish", "Flower", "Gravel", "Sugar"]
    training_data = []
    img_size = 32
    for category in CATEGORIES:

        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat
        c = -30
        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                img_array = cv2.resize(img_array, (img_size, img_size))
                img_h, img_w = img_array.shape
                img_array = img_array.reshape((img_h, img_w, 1)).astype(np.float32)
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
    epoch = len(X)
    X = np.array(X)
    y = np.array(y)

    input_img = layers.Input(shape=(img_h, img_w, 1))
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Flatten()(x)
    encoder = layers.Dense(16, activation='relu')(x)

    x = layers.Dense(8192, activation='relu')(encoder)
    x = layers.Reshape((8, 8, 128))(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)
    decoder = layers.Conv2D(1, (3, 3), activation='relu', padding='same')(x)

    model = keras.Model(input_img, decoder)

    initial_learning_rate = 0.1
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=epoch * 5,
        decay_rate=.9,
        staircase=True)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss=keras.losses.mse)

    log_dir = os.path.join("tf_logs\\AE\\", datetime.now().strftime("%Y%m%d-%H%M%S/"))
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch=0)
    lr_callback = keras.callbacks.LearningRateScheduler(schedule=lr_schedule)

    save_callback = keras.callbacks.ModelCheckpoint(log_dir, monitor='val_loss', verbose=True, save_best_only=True,
                                                    save_weights_only=False, mode='min', save_freq='epoch')

    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
    print(model.summary())

    model.fit(x=train_x,
              y=train_x,
              batch_size=512,
              epochs=100,
              use_multiprocessing=True,
              validation_data=(test_x, test_x),
              callbacks=[tensorboard_callback,
                         save_callback,
                         # lr_callback
                         ])

    pred = model.predict(test_x[:1, :, :])
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(test_x[0, :, :].squeeze())
    axs[1].imshow(pred.squeeze())
    plt.show()

    pred = model.predict(train_x[:1, :, :])
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(train_x[0, :, :].squeeze())
    axs[1].imshow(pred.squeeze())
    plt.show()


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    main()
