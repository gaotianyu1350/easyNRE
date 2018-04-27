import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import datetime
import sys

from network.embedding import Embedding
from network.encoder import Encoder
from network.selector import Selector
from network.classifier import Classifier
import os

FLAGS = tf.app.flags.FLAGS

class Accuracy(object):

    def __init__(self):
        self.correct = 0
        self.total = 0

    def add(self, is_correct):
        self.total += 1
        if is_correct:
            self.correct += 1

    def get(self):
        if self.total == 0:
            return 0
        else:
            return float(self.correct) / self.total

    def clear(self):
        self.correct = 0
        self.total = 0

class Framework(object):

    def __init__(self, is_training):
        # Place Holder
        self.word = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_length], name='input_word')
        self.word_vec = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.word_size], name='word_vec')
        self.pos1 = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_length], name='input_pos1')
        self.pos2 = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_length], name='input_pos2')
        self.length = tf.placeholder(dtype=tf.int32, shape=[None], name='input_length')
        self.mask = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_length], name='input_mask')
        self.label = tf.placeholder(dtype=tf.int32, shape=[None], name='label')
        self.label_for_select = tf.placeholder(dtype=tf.int32, shape=[None], name='label_for_select')
        self.scope = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size + 1], name='scope')    
        self.weights = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size])

        # Network
        self.embedding = Embedding(is_training, self.word_vec, self.word, self.pos1, self.pos2)
        self.encoder = Encoder(is_training, length=self.length, mask=self.mask)
        self.selector = Selector(is_training, self.scope, self.label_for_select)
        self.classifier = Classifier(is_training, self.label, self.weights)

        # Metrics 
        self.acc_NA = Accuracy()
        self.acc_not_NA = Accuracy()
        self.acc_total = Accuracy()
        self.step = 0
        
        # Session
        self.sess = None

    def load_data(self):
        print 'reading training data...'
        self.data_word_vec = np.load(os.path.join(FLAGS.export_path, 'vec.npy'))
        self.data_instance_triple = np.load(os.path.join(FLAGS.export_path, 'train_instance_triple.npy'))
        self.data_instance_scope = np.load(os.path.join(FLAGS.export_path, 'train_instance_scope.npy'))
        self.data_train_length = np.load(os.path.join(FLAGS.export_path, 'train_len.npy'))
        self.data_train_label = np.load(os.path.join(FLAGS.export_path, 'train_label.npy'))
        self.data_train_word = np.load(os.path.join(FLAGS.export_path, 'train_word.npy'))
        self.data_train_pos1 = np.load(os.path.join(FLAGS.export_path, 'train_pos1.npy'))
        self.data_train_pos2 = np.load(os.path.join(FLAGS.export_path, 'train_pos2.npy'))
        self.data_train_mask = np.load(os.path.join(FLAGS.export_path, 'train_mask.npy'))
        print 'reading finished'
        print 'mentions         : %d' % (len(self.data_instance_triple))
        print 'sentences        : %d' % (len(self.data_train_length))
        print 'relations        : %d' % (FLAGS.num_classes)
        print 'word size        : %d' % (FLAGS.word_size)
        print 'position size     : %d' % (FLAGS.pos_size)
        print 'hidden size        : %d' % (FLAGS.hidden_size)

        self.reltot = {}
        for index, i in enumerate(self.data_train_label):
            if not i in self.reltot:
                self.reltot[i] = 1.0
            else:
                self.reltot[i] += 1.0
        for i in self.reltot:
            self.reltot[i] = 1 / (self.reltot[i] ** (0.05))
        print self.reltot

    def init_model(self, loss, output, optimizer=tf.train.GradientDescentOptimizer, pretrain_model=None):
        print 'initializing model...'
        # Loss and output
        self.loss = loss
        self.output = output

        # Optimizer
        self.sess = tf.Session()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        tf.summary.scalar('learning_rate', FLAGS.learning_rate)
        self.optimizer = optimizer(FLAGS.learning_rate)
        self.grads_and_vars = self.optimizer.compute_gradients(loss)
        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)

        # Summary
        self.merged_summary = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(FLAGS.summary_dir, self.sess.graph)

        # Saver
        self.saver = tf.train.Saver(max_to_keep=None)
        if pretrain_model is None:
            self.sess.run(tf.global_variables_initializer())
        else:
            self.saver.restore(self.sess, pretrain_model)

        print 'initializing finished'

    def train_one_step(self, index, scope, weights, label, result_needed=[]):
        feed_dict = {
            self.word: self.data_train_word[index, :],
            self.word_vec: self.data_word_vec,
            self.pos1: self.data_train_pos1[index, :],
            self.pos2: self.data_train_pos2[index, :],
            self.mask: self.data_train_mask[index, :],
            self.length: self.data_train_length,
            self.label: label,
            self.label_for_select: self.data_train_label[index],
            self.scope: np.array(scope),
            self.weights: weights
        }
        result = self.sess.run([self.train_op, self.global_step, self.merged_summary, self.output] + result_needed, feed_dict)
        self.step = result[1]
        _output = result[3]
        result = result[4:]

        # Training accuracy
        for i, prediction in enumerate(_output):
            if label[i] == 0:
                self.acc_NA.add(prediction == label[i])
            else:
                self.acc_not_NA.add(prediction == label[i])
            self.acc_total.add(prediction == label[i])

        return result

    def train(self, one_step=train_one_step):
        train_order = range(len(self.data_instance_triple))
        for one_epoch in range(FLAGS.max_epoch):
            print('epoch ' + str(one_epoch + 1) + ' starts...')
            self.acc_NA.clear()
            self.acc_not_NA.clear()
            self.acc_total.clear()
            np.random.shuffle(train_order)
            for i in range(int(len(train_order) / float(FLAGS.batch_size))):
                input_scope = np.take(self.data_instance_scope, train_order[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size], axis=0)
                index = []
                scope = [0]
                weights = []
                label = []
                for num in input_scope:
                    index = index + range(num[0], num[1] + 1)
                    label.append(self.data_train_label[num[0]])
                    scope.append(scope[len(scope) - 1] + num[1] - num[0] + 1)
                    weights.append(self.reltot[self.data_train_label[num[0]]])

                loss = one_step(self, index, scope, weights, label, [self.loss])

                time_str = datetime.datetime.now().isoformat()
                sys.stdout.write("epoch %d step %d time %s | loss : %f, NA accuracy: %f, not NA accuracy: %f, total accuracy %f" % (one_epoch, i, time_str, loss[0], self.acc_NA.get(), self.acc_not_NA.get(), self.acc_total.get()) + '\r')
                sys.stdout.flush()

            if (one_epoch + 1) % FLAGS.save_epoch == 0:
                print 'epoch ' + str(one_epoch + 1) + ' has finished'
                print 'saving model...'
                path = self.saver.save(self.sess, os.path.join(FLAGS.model_dir, 'checkpoint'), global_step=self.step)
                print 'have saved model to ' + path
