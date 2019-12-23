import tensorflow as tf
import numpy as np


class SimpleAnn:
    """
    A model of a simple Multi-layer network
    """

    def __init__(self, hidden_lst: list, input_num: int, class_num: int):
        self.hidden_layers = list(hidden_lst)
        self.input_num = input_num
        self.class_num = class_num

        self.weights, self.biases = self.getWeights()

    def getWeights(self) -> (dict, dict):
        """
        Store layers weight & bias
        """
        weights = dict()
        biases = dict()
        last_output = self.input_num
        for idx, hidden_layer in enumerate(self.hidden_layers):
            weights['L' + str(idx)] = tf.Variable(tf.random.truncated_normal([last_output, hidden_layer], stddev=0.1))
            biases['L' + str(idx)] = tf.Variable(tf.constant(0.1, shape=[hidden_layer]))
            last_output = hidden_layer

            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weights['L' + str(idx)])
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, biases['L' + str(idx)])

        weights['out'] = tf.Variable(tf.random.truncated_normal([hidden_layer, self.class_num], stddev=0.1))
        biases['out'] = tf.Variable(tf.constant(0.1, shape=[self.class_num]))

        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weights['out'])
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, biases['out'])
        return weights, biases


def getModel(self, x: np.ndarray):
    """
    Build the network model, based on the input and the weights
    """
    global num_classes
    # Output fully connected layer with a neuron for each class
    layers_keys = list(self.weights.keys())
    Ls = [tf.matmul(x, self.weights[layers_keys[0]]) + self.biases[layers_keys[0]]]
    relus = [tf.nn.relu(Ls[-1])]
    for key in layers_keys[1:-1]:
        newL = tf.add(tf.matmul(relus[-1], self.weights[key]), self.biases[key])
        Ls.append(newL)
        relus.append(tf.nn.sigmoid(newL))

    out_layer = tf.add(tf.matmul(relus[-1], self.weights['out']), self.biases['out'])

    return out_layer
