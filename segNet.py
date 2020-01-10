from __future__ import division
from __future__ import print_function

import datetime
import os
import io

from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from datetime import datetime
import time

from utils import prepareSegData

NAME = "clouds recognition{}".format(int(time.time()))


def main():
    DATA_DIR = "data/train_images"
    kLABEL_NUM = 4
    img_size = img_h = img_w = 128
    train_x, test_x, train_y, test_y = \
        prepareSegData(
            img_list_file='data/train.csv',
            img_folder=DATA_DIR,
            img_size=img_size,
            sample_size=-1000,
            normalize=True)
    epoch = len(train_x)

    input_img = layers.Input(shape=[img_h, img_w, 3])
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(input_img)     # 64
    x = layers.Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)             # 32
    x = layers.Conv2D(128, (3, 3), strides=1, activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)             # 16
    x = layers.Conv2D(128, (3, 3), strides=1, activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)             # 8

    x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)   # 16
    x = layers.Conv2D(128, (3, 3), strides=1, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)   # 32
    x = layers.Conv2D(128, (3, 3), strides=1, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)   # 64
    x = layers.Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(x)
    decoder = layers.Conv2DTranspose(kLABEL_NUM, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)  # 128

    model = keras.Model(input_img, decoder)

    initial_learning_rate_main = 1e-5
    lr_schedule_main = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate_main,
        decay_steps=epoch * 10,
        decay_rate=1e-1,
        staircase=True)

    model.compile(optimizer=tf.keras.optimizers.Adam(),  # learning_rate=lr_schedule_main),
                  loss=tf.keras.losses.mse)

    log_dir = os.path.join("tf_logs", "SegNet", datetime.now().strftime("%Y%m%d-%H%M%S/"))
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch=0)
    lr_callback = keras.callbacks.LearningRateScheduler(schedule=lr_schedule_main)

    save_callback = keras.callbacks.ModelCheckpoint(log_dir,
                                                    monitor='val_loss',
                                                    verbose=True,
                                                    save_best_only=True,
                                                    save_weights_only=False,
                                                    mode='min',
                                                    save_freq='epoch')

    print(model.summary())

    file_writer = tf.summary.create_file_writer(log_dir)

    # Using the file writer, log the reshaped image.
    def log_img_pred(epoch, logs):
        # Use the model to predict the values from the validation dataset.
        test_img = model.predict(test_x[2:3, :, :, :])
        test_lbl = test_y[2, :, :, :].squeeze()
        test_img = test_img.squeeze()

        fig, ax = plt.subplots(2, 4)
        ax[0, 0].imshow(test_lbl[:, :, 0])
        ax[0, 1].imshow(test_lbl[:, :, 1])
        ax[0, 2].imshow(test_lbl[:, :, 2])
        ax[0, 3].imshow(test_lbl[:, :, 3])
        ax[1, 0].imshow(test_img[:, :, 0])
        ax[1, 1].imshow(test_img[:, :, 1])
        ax[1, 2].imshow(test_img[:, :, 2])
        ax[1, 3].imshow(test_img[:, :, 3])

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        # Log the confusion matrix as an image summary.
        with file_writer.as_default():
            tf.summary.image("Test prediction", image, step=epoch)

    # Define the per-epoch callback.
    cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_img_pred)

    model.fit(x=train_x,
              y=train_y,
              batch_size=64,
              epochs=200,
              use_multiprocessing=True,
              validation_data=(test_x, test_y),
              callbacks=[tensorboard_callback,
                         save_callback,
                         cm_callback
                         ])

    showTest(model, test_x, test_y)


def showTest(model: keras.Model, test_x, test_y: np.ndarray):
    test_img = model.predict(test_x[2:3, :, :, :])
    test_lbl = test_y[2, :, :, :].squeeze()
    test_img = test_img.squeeze()

    fig, ax = plt.subplots(2, 4)
    ax[0, 0].imshow(test_lbl[:, :, 0])
    ax[0, 1].imshow(test_lbl[:, :, 1])
    ax[0, 2].imshow(test_lbl[:, :, 2])
    ax[0, 3].imshow(test_lbl[:, :, 3])
    ax[1, 0].imshow(test_img[:, :, 0])
    ax[1, 1].imshow(test_img[:, :, 1])
    ax[1, 2].imshow(test_img[:, :, 2])
    ax[1, 3].imshow(test_img[:, :, 3])
    plt.show()


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    main()
