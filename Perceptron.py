import tensorflow as tf
import numpy as np


class Perceptron:
    def __init__(self, input_num: int, class_num: int):
        self.input_num = input_num
        self.class_num = class_num

        self.weights, self.biases = self.getWeights()

    def getWeights(self) -> (dict, dict):
        # Store layers weight & bias

        weights = {
            'out': tf.Variable(tf.random.truncated_normal([self.input_num, self.class_num], stddev=0.1))
        }
        biases = {
            'out': tf.Variable(tf.constant(0.1, shape=[self.class_num]))
        }

        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weights['out'])
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, biases['out'])
        return weights, biases

    # Define the neural network
    def getModel(self, x: np.ndarray):
        # Output fully connected layer with a neuron for each class
        out_layer = tf.add(tf.matmul(x, self.weights['out']), self.biases['out'])
        return out_layer

