from __future__ import absolute_import
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

from utils import prepareData

NAME = "clouds recognition{}".format(int(time.time()))


def main():
    img_size = img_h = img_w = 32
    train_x, test_x, train_y, test_y = prepareData(img_size=img_size, sample_size=-30)
    epoch = len(train_x)

    input_img = layers.Input(shape=(img_h, img_w, 1))
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Flatten()(x)
    encoder = layers.Dense(16 ** 2, activation='relu')(x)

    x = layers.Dense(8192, activation='relu')(encoder)
    x = layers.Reshape((8, 8, 128))(x)

    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same')(x)

    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)

    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    decoder = layers.Conv2D(1, (3, 3), activation='relu', padding='same')(x)

    model = keras.Model(input_img, decoder)
    en_model = keras.Model(input_img, encoder)
    initial_learning_rate = 0.01
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=epoch * 5,
        decay_rate=.9,
        staircase=True)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss=keras.losses.mse)

    log_dir = os.path.join("tf_logs\\AE\\", datetime.now().strftime("%Y%m%d-%H%M%S/"))
    os.makedirs(os.path.join(log_dir,'encoder'))
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch=0)
    lr_callback = keras.callbacks.LearningRateScheduler(schedule=lr_schedule)

    save_callback = keras.callbacks.ModelCheckpoint(log_dir, monitor='val_loss', verbose=True, save_best_only=True,
                                                    save_weights_only=False, mode='min', save_freq='epoch')

    print(model.summary())

    file_writer = tf.summary.create_file_writer(log_dir)

    # Using the file writer, log the reshaped image.
    def log_img_pred(epoch, logs):
        # Use the model to predict the values from the validation dataset.
        test_img = model.predict(test_x[:1, :, :, :])
        test_img = test_img.reshape((1, img_size, img_size, 1)).astype(np.uint8)
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(test_x[0, :, :, :].astype(np.uint8).squeeze())
        ax[1].imshow(test_img.squeeze())

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
              y=train_x,
              batch_size=128,
              epochs=50,
              use_multiprocessing=True,
              validation_data=(test_x, test_x),
              callbacks=[tensorboard_callback,
                         save_callback,
                         cm_callback
                         ])

    en_model.save(os.path.join(log_dir, 'encoder'))
    img = test_x[0, :, :, 0].reshape((32, 32))
    showTest(model, img)


def showTest(model: keras.Model, img: np.ndarray):
    h, w = img.shape
    pred = model.predict(img.reshape((1, h, w, 1)))
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img)
    axs[1].imshow(pred.squeeze())
    plt.show()


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    main()
