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
import datetime
import os
import shutil

from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


@dataclass
class Datapack:
    images: np.ndarray
    labels: np.ndarray
    batch_index = 0

    def next_batch(self, n_batch: int) -> (np.ndarray, np.ndarray):
        if self.batch_index + n_batch >= len(self.images):
            self.batch_index = 0

        return (self.images[self.batch_index:self.batch_index + n_batch, :],
                self.labels[self.batch_index:self.batch_index + n_batch, :],)


def setupWeights(input_num: int, class_num: int) -> (dict, dict):
    # Store layers weight & bias
    weights = {
        'out': tf.Variable(tf.random.normal([input_num, class_num]))
    }
    biases = {
        'out': tf.Variable(tf.random.normal([class_num]))
    }

    return weights, biases


# Define the neural network
def perceptron(x: np.ndarray, weights: dict, biases: dict):
    global num_classes
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(x, weights['out']) + biases['out']
    return out_layer


def setupWeightsANN(input_num: int, class_num: int) -> (dict, dict):
    # Store layers weight & bias
    hidden_1 = 256*2
    hidden_2 = 256
    weights = {
        'L1': tf.Variable(tf.random.normal([input_num, hidden_1])),
        'L2': tf.Variable(tf.random.normal([hidden_1, hidden_2])),
        'out': tf.Variable(tf.random.normal([hidden_2, class_num]))
    }
    biases = {
        'L1': tf.Variable(tf.random.normal([hidden_1])),
        'L2': tf.Variable(tf.random.normal([hidden_2])),
        'out': tf.Variable(tf.random.normal([class_num]))
    }

    return weights, biases


# Define the neural network
def ANN(x: np.ndarray, weights: dict, biases: dict):
    global num_classes
    # Output fully connected layer with a neuron for each class
    L1 = tf.matmul(x, weights['L1']) + biases['L1']
    relu1 = tf.nn.relu(L1)
    L2= tf.matmul(relu1, weights['L2']) + biases['L2']
    relu2 = tf.nn.relu(L2)

    out_layer = tf.matmul(relu2, weights['out']) + biases['out']

    return out_layer


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


def preProcess(img):
    img = img / 255
    thrs = .5
    img[img < thrs] = 0
    img[img >= thrs] = 1

    return img


def loadData(folder_path: str, class_cap: int = -1) -> (Datapack, dict):
    print("Loading data...")
    classes = os.listdir(folder_path)
    class2id = {x: i for i, x in enumerate(classes)}

    images = []
    labels = []
    for clz in classes:
        max_samp = class_cap
        sam_count = 0
        print('\t%s:\t' % clz, end='')
        class_path = os.path.join(folder_path, clz)
        for img_path in os.listdir(class_path):
            img_full_path = os.path.join(class_path, img_path)

            img = cv2.imread(img_full_path, cv2.IMREAD_GRAYSCALE)
            img = preProcess(img)
            img = img.reshape((1, -1))
            images.append(img)
            lbl_vec = np.zeros(len(classes))
            lbl_vec[class2id[clz]] = 1
            labels.append(lbl_vec)
            sam_count += 1
            max_samp -= 1
            if max_samp == 0:
                break

        print(sam_count)

    data = Datapack(
        np.array(images, dtype=np.float32).squeeze(),
        np.array(labels, dtype=np.float32))
    return data, class2id


def build_and_run(nn, n_input: int, n_classes: int,
                  train: Datapack, test: Datapack,
                  n_steps: int, n_batch: int,
                  weight: dict, biases: dict):
    # Construct model
    # tf Graph input
    X = tf.compat.v1.placeholder("float", [None, n_input])
    Y = tf.compat.v1.placeholder("float", [None, n_classes])
    logits = nn(X)

    # TensorBoard
    # Construct model and encapsulating all ops into scopes, making
    # Tensorboard's Graph visualization more convenient
    with tf.name_scope('Model'):
        # Model
        pred = tf.nn.softmax(logits)
    with tf.name_scope('Loss'):
        # Minimize error using cross entropy
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
    with tf.name_scope('SGD'):
        # Gradient Descent
        train_op = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss_op)
    with tf.name_scope('Accuracy'):
        # Accuracy
        acc = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        acc = tf.reduce_mean(tf.cast(acc, tf.float32))

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.compat.v1.global_variables_initializer()

    # Create a summary to monitor accuracy tensor
    tf.compat.v1.summary.scalar("Accuracy", acc)
    tf.compat.v1.summary.scalar("Loss", loss_op)
    merged_summary = tf.compat.v1.summary.merge_all()

    # Logging
    tf_logs_path = os.path.join(os.getcwd(), 'tf_logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(os.path.join(tf_logs_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(tf_logs_path, "test"), exist_ok=True)

    # Start training
    with tf.compat.v1.Session() as sess:
        # op to write logs to Tensorboard
        summary_writer_train = tf.compat.v1.summary.FileWriter(os.path.join(tf_logs_path, "train"),
                                                               graph=tf.compat.v1.get_default_graph())
        summary_writer_test = tf.compat.v1.summary.FileWriter(os.path.join(tf_logs_path, "test"),
                                                              graph=tf.compat.v1.get_default_graph())

        # Run the initializer
        sess.run(init)

        epoch_count = 0
        for step in range(1, n_steps + 1):
            batch_x, batch_y = train.next_batch(n_batch)
            # Run optimization op (backprop)
            c = sess.run(train_op,
                         feed_dict={X: batch_x,
                                    Y: batch_y})

            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                print("Epoch " + str(epoch_count)
                      + ",\t Training Accuracy= " + "{:.3f}".format(acc.eval({X: train.images, Y: train.labels}))
                      + ",\t Test Accuracy= " + "{:.3f}".format(acc.eval({X: test.images, Y: test.labels})))
                epoch_count += 1

                _, _, summary_train = sess.run([acc, loss_op, merged_summary],
                                               feed_dict={X: train.images,
                                                          Y: train.labels})
                summary_writer_train.add_summary(summary_train, step)
                _, _, summary_test = sess.run([acc, loss_op, merged_summary],
                                              feed_dict={X: test.images,
                                                         Y: test.labels})
                summary_writer_test.add_summary(summary_test, step)

        print("Optimization Finished!")

        # Calculate accuracy for the Cloud dataset test images
        print("Testing Accuracy:",
              sess.run(accuracy, feed_dict={X: test.images,
                                            Y: test.labels}))


def run():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    data_folder = os.path.join('data/mini_data')
    data, class2id = loadData(data_folder, 3000)
    train, test = splitData(data, ratio=0.9)

    # Parameters
    global learning_rate, display_step, epoch
    learning_rate = 0.1
    epoch = len(train.images)
    batch_size = 128 * 4
    num_steps = (epoch * 20) // batch_size
    print("Steps:", num_steps)
    display_step = (epoch // batch_size)

    # Network Parameters
    global num_classes, num_input
    num_input = len(data.images[0])
    num_classes = len(class2id)

    USE_ANN = True
    if USE_ANN:
        p_weights, p_bias = setupWeightsANN(num_input, num_classes)
        net = lambda x: ANN(x, p_weights, p_bias)
    else:
        p_weights, p_bias = setupWeights(num_input, num_classes)
        net = lambda x: perceptron(x, p_weights, p_bias),
    build_and_run(
        net,
        n_input=num_input,
        n_classes=num_classes,
        train=train,
        test=test,
        n_steps=num_steps,
        n_batch=batch_size,
        weight=p_weights,
        biases=p_bias
    )


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    run()
