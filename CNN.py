import datetime
import os
from dataclasses import dataclass
# from tensorflow.keras.callbacks import TensorBoard
import cv2
import random
# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import callbacks
from keras.models import Sequential
from keras.losses import sparse_categorical_crossentropy
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import scipy
import pickle
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

NAME = "clouds recognition{}".format(int(time.time()))


# tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

def main():
    DATADIR = "data/mini_data"

    CATEGORIES = ["Fish", "Flower", "Gravel", "Sugar"]
    training_data = []
    for category in CATEGORIES:  # do dogs and cats

        path = os.path.join(DATADIR, category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                img_array = cv2.resize(img_array, (128, 128))
                img_h, img_w = img_array.shape
                img_array = img_array.reshape((img_h, img_w, 1))
                training_data.append([img_array, class_num])  # add this to our training_data
            except IOError as e:  # in the interest in keeping the output clean...
                pass

    random.shuffle(training_data)

    X = []
    y = []

    for features, label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X)
    y = np.array(y)
    pickle_out = open("X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()

    pickle_in = open("X.pickle", "rb")
    X = pickle.load(pickle_in)

    pickle_in = open("y.pickle", "rb")
    y = pickle.load(pickle_in)

    model = Sequential()

    model.add(Conv2D(256, (3, 3), input_shape=(img_h, img_w, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(CATEGORIES), activation='softmax'))

    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()

    logdir = "tf_logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = callbacks.TensorBoard(log_dir=logdir)
    model.fit(X, y,
              batch_size=16,
              epochs=3,
              validation_split=0.3,
              callbacks=[tensorboard_callback])


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    main()
