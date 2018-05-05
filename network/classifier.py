import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

class Classifier(object):

    def __init__(self, is_training, label, weights):
        self.label = label
        self.weights = weights
        self.is_training = is_training

    def softmax_cross_entropy(self, x):
        with tf.name_scope("loss"):
            label_onehot = tf.one_hot(indices=self.label, depth=FLAGS.num_classes, dtype=tf.int32)
            loss = tf.losses.softmax_cross_entropy(onehot_labels=label_onehot, logits=x, weights=self.weights)
            tf.summary.scalar('loss', loss)
            return loss

    def output(self, x):
        with tf.name_scope("output"):
            output = tf.argmax(x, 1, name="output")
            return output
