from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import io

from keras import regularizers
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from datetime import datetime
import time

from utils import prepareData


def main():
    DATA_DIR = "data/mini_data"
    CATEGORIES = os.listdir(DATA_DIR)
    img_size = img_h = img_w = 64
    train_x, test_x, train_y, test_y = \
        prepareData(
            img_folder=DATA_DIR,
            img_size=img_size,
            sample_size=-128,
            normalize=True)
    epoch = len(train_x)

    # Network construction
    #   Encoder
    input_img = layers.Input(shape=(img_h, img_w, 1))
    x = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(input_img)
    x = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(x)
    x = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(x)
    x = layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    x = layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (5, 5), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (5, 5), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (5, 5), activation='relu', padding='same')(x)
    x = layers.Flatten()(x)
    mid_size = img_size // 4
    encoder = layers.Dense(mid_size ** 2, activation='relu', name='encoder_output')(x)

    #   Decoder
    x = layers.Dense(128 * (mid_size ** 2), activation='relu')(encoder)
    x = layers.Reshape((mid_size, mid_size, 128))(x)
    x = layers.Conv2DTranspose(128, (5, 5), activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(128, (5, 5), activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(128, (5, 5), strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(64, (5, 5), activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(64, (5, 5), activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(64, (5, 5), strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(64, (5, 5), activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(64, (5, 5), activation='relu', padding='same')(x)
    # AutoEncoder output
    decoder = layers.Conv2D(1, (5, 5), activation='relu', padding='same', name="decoder_output")(x)

    # ANN connected to the encoder
    x = layers.Flatten()(encoder)
    x = layers.Dense(32, activation='relu', name="Reg_nn")(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.4)(x)

    # Main output
    main_output = layers.Dense(len(CATEGORIES),
                               activation=tf.keras.activations.softmax,
                               name='main_output')(x)

    model = keras.Model(input_img, [main_output, decoder])
    decoder_model = keras.Model(input_img, decoder)

    initial_learning_rate_main = 1e-4
    lr_schedule_main = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate_main,
        decay_steps=epoch * 5,
        decay_rate=1e-1,
        staircase=True)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule_main),
                  metrics={'main_output': 'accuracy'},
                  loss={'main_output': keras.losses.sparse_categorical_crossentropy,
                        'decoder_output': tf.keras.losses.mse},
                  loss_weights={'main_output': 1, 'decoder_output': 1})

    log_dir = os.path.join("tf_logs", "AL", datetime.now().strftime("%Y%m%d-%H%M%S/"))
    os.makedirs(os.path.join(log_dir, 'encoder'))
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch=0)

    save_callback = keras.callbacks.ModelCheckpoint(log_dir, monitor='val_main_output_accuracy', verbose=True,
                                                    save_best_only=True,
                                                    save_weights_only=False, mode='max', save_freq='epoch')

    print(model.summary())

    file_writer = tf.summary.create_file_writer(log_dir)

    # Use the model to display the state of the autoencoder from the validation dataset.
    def log_img_pred(epoch, logs):
        test_img = decoder_model.predict(test_x[2:3, :, :, :])
        test_img = test_img.reshape((1, img_size, img_size, 1))
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(test_x[2, :, :, :].squeeze())
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
              y=[train_y, train_x],
              batch_size=128,
              epochs=200,
              use_multiprocessing=True,
              validation_data=(test_x, [test_y, test_x]),
              callbacks=[tensorboard_callback,
                         save_callback,
                         cm_callback
                         ])

    img = test_x[0, :, :, 0].reshape((img_h, img_w))
    showTest(model, img)


def showTest(model: keras.Model, img: np.ndarray):
    h, w = img.shape
    pred = model.predict(img.reshape((1, h, w, 1)))
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img)
    axs[1].imshow(pred[1].squeeze())
    plt.show()


if __name__ == "__main__":
    if 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
        config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    main()
