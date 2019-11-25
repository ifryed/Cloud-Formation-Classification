""" Neural Network.
A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).
This example is using TensorFlow layers, see 'neural_network_raw' example for
a raw implementation with variables.
Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

# Import MNIST data
import os
from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


@dataclass
class Datapack:
    images: np.ndarray
    labels: np.ndarray


# Define the neural network
def perceptron(x_dict: dict):
    global num_classes
    # TF Estimator input is a dict, in case of multiple inputs
    x = x_dict['images']
    out_layer = tf.layers.dense(x, num_classes)
    return out_layer


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    logits = perceptron(features)

    # Predictions
    pred_classes = tf.argmax(logits, axis=1)
    pred_probas = tf.nn.softmax(logits)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.comapt.v1.estimator.EstimatorSpec(mode, predictions=pred_classes)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.compat.v1.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.compat.v1.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.compat.v1.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs


def splitData(data: Datapack, ratio: float = 0.7) -> (Datapack, Datapack):
    imgs = data.images
    lbls = data.labels
    n_data = len(lbls)

    idx = [x for x in range(n_data)]
    np.random.shuffle(idx)

    imgs_shuff = imgs[idx, :]
    lbls_shuff = lbls[idx]

    split = int(n_data * ratio)
    train = Datapack(imgs_shuff[:split, :], lbls_shuff[:split])
    test = Datapack(imgs_shuff[split:, :], lbls_shuff[split:])

    return train, test


def loadData(folder_path: str) -> (Datapack, dict):
    print("Loading data...")
    classes = os.listdir(folder_path)
    class2id = {x: i for i, x in enumerate(classes)}

    images = []
    labels = []
    for clz in classes:
        max_samp = 100
        sam_count = 0
        print('\t%s:\t' % clz, end='')
        class_path = os.path.join(folder_path, clz)
        for img_path in os.listdir(class_path):
            img_full_path = os.path.join(class_path, img_path)

            img = cv2.imread(img_full_path, cv2.IMREAD_GRAYSCALE)
            img = img.reshape((1, -1))
            images.append(img)
            labels.append(class2id[clz])
            sam_count += 1
            max_samp -= 1
            if max_samp == 0:
                break

        print(sam_count)

    data = Datapack(
        np.array(images, dtype=np.float32).squeeze(),
        np.array(labels, dtype=np.float32))
    return data, class2id


def run():
    data_folder = os.path.join('data/mini_data')
    data, class2id = loadData(data_folder)
    train, test = splitData(data, ratio=0.7)

    # Parameters
    global learning_rate, display_step
    learning_rate = 0.1
    num_steps = 1000
    batch_size = 128
    display_step = 100

    # Network Parameters
    global num_classes, num_input
    num_input = len(data.images[0])
    num_classes = len(class2id)

    # Build the Estimator
    model = tf.compat.v1.estimator.Estimator(model_fn)
    # Define the input function for training
    input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={'images': train.images}, y=train.labels,
        batch_size=batch_size, num_epochs=None, shuffle=True)
    # Define the input function for evaluating
    test_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={'images': test.images}, y=test.labels,
        batch_size=batch_size, shuffle=False)

    train_spec = tf.compat.v1.estimator.TrainSpec(input_fn=input_fn,
                                                  max_steps=num_steps)
    eval_spec = tf.compat.v1.estimator.EvalSpec(input_fn=test_fn,
                                                steps=10,
                                                start_delay_secs=60,
                                                throttle_secs=60)

    # Train the Model
    tf.compat.v1.estimator.train_and_evaluate(model, train_spec, eval_spec)
    # model.train(input_fn, steps=num_steps)

    # Evaluate the Model
    # Use the Estimator 'evaluate' method
    e = model.evaluate(test_fn)

    print("Testing Accuracy:", e['accuracy'])


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    run()
