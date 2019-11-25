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
    batch_index = 0

    def next_batch(self, n_batch: int) -> (np.ndarray, np.ndarray):
        if self.batch_index + n_batch >= len(self.images):
            self.batch_index = 0

        return (self.images[self.batch_index:self.batch_index + n_batch, :],
                self.labels[self.batch_index:self.batch_index + n_batch, :],)


# Define the neural network
def perceptron(x: np.ndarray, weights: dict, biases: dict):
    global num_classes
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(x, weights['out']) + biases['out']
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
            img = img.reshape((1, -1))
            # img = img / img.max()
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


def setupWeights(input_num: int, class_num: int) -> (dict, dict):
    # Store layers weight & bias
    weights = {
        'out': tf.Variable(tf.random.normal([input_num, class_num]))
    }
    biases = {
        'out': tf.Variable(tf.random.normal([class_num]))
    }

    return weights, biases


def build_and_run(nn, n_input: int, n_classes: int, train: Datapack, test: Datapack, n_steps: int, n_batch: int):
    # Construct model
    # tf Graph input
    X = tf.compat.v1.placeholder("float", [None, n_input])
    Y = tf.compat.v1.placeholder("float", [None, n_classes])
    logits = nn(X)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=Y))
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.compat.v1.global_variables_initializer()

    # Start training
    with tf.compat.v1.Session() as sess:

        # Run the initializer
        sess.run(init)

        for step in range(1, n_steps + 1):
            batch_x, batch_y = train.next_batch(n_batch)
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= "
                      + "{:.4f}".format(loss) + ", Training Accuracy= "
                      + "{:.3f}".format(acc))

        print("Optimization Finished!")

        # Calculate accuracy for MNIST test images
        print("Testing Accuracy:",
              sess.run(accuracy, feed_dict={X: test.images,
                                            Y: test.labels}))


def run():
    data_folder = os.path.join('data/mini_data')
    data, class2id = loadData(data_folder, -100)
    train, test = splitData(data, ratio=0.7)

    # Parameters
    global learning_rate, display_step
    learning_rate = 0.1
    num_steps = 1000
    batch_size = 128
    display_step = 50

    # Network Parameters
    global num_classes, num_input
    num_input = len(data.images[0])
    num_classes = len(class2id)

    p_weights, p_bais = setupWeights(num_input, num_classes)

    build_and_run(
        lambda x: perceptron(x, p_weights, p_bais),
        n_input=num_input,
        n_classes=num_classes,
        train=train,
        test=test,
        n_steps=num_steps,
        n_batch=batch_size
    )


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    run()
