#coding:utf-8
import tensorflow as tf
import numpy as np
from readDataset import readStandardData
from readDataset import readWeights


class Network():
    def __init__(self, num_hidden, batch_size):
        self.batch_size = batch_size

        self.num_hidden = num_hidden


        # graph related
        self.graph = tf.Graph()
        self.tf_train_samples = None
        self.tf_train_labels = None
        self.tf_test_samples = None
        self.tf_test_labels = None
        self.tf_test_prediction = None
        

    def defineGraph(self):
        with self.graph.as_default():
            tf_train_samples = tf.placeholder(
                tf.float32, shape=(self.batch_size, image_size, image_size, num_channels)
            )

            tf_train_labels = tf.placeholder(
                tf.float32, shape=(self.batch_size, num_labels)
            )# length of a label

            tf_test_samples = tf.placeholder(
                tf.float32, shape=(self.batch_size, image_size, image_size, num_channels)
            )

            tf_test_labels = tf.placeholder(
                tf.float32, shape=(self.batch_size, num_labels)
            )# length of a label

            fc1_weights = tf.Variable(
                tf.truncated_normal([image_size*image_size*self.batch_size, self.num_hidden], stddev=0.1)
            )
            fc1_biases  = tf.Variable(tf.constant(0.1, shape=[self.num_hidden]))

            fc2_weights = tf.Variable(
                tf.truncated_normal([self.num_hidden, num_labels], stddev=0.1)
            )
            fc2_biases  = tf.Variable(tf.constant(0.1, shape=[num_labels]))


    def train(self):
        pass

    def test(self):
        pass

    def accuracy(self):
        pass
