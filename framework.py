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

    def load_train_data(self):
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

    def load_test_data(self):
        print 'reading test data...'
        self.data_word_vec = np.load(os.path.join(FLAGS.export_path, 'vec.npy'))
        self.data_instance_triple = np.load(os.path.join(FLAGS.export_path, 'test_instance_triple.npy'))
        self.data_instance_scope = np.load(os.path.join(FLAGS.export_path, 'test_instance_scope.npy'))
        self.data_test_length = np.load(os.path.join(FLAGS.export_path, 'test_len.npy'))
        self.data_test_label = np.load(os.path.join(FLAGS.export_path, 'test_label.npy'))
        self.data_test_word = np.load(os.path.join(FLAGS.export_path, 'test_word.npy'))
        self.data_test_pos1 = np.load(os.path.join(FLAGS.export_path, 'test_pos1.npy'))
        self.data_test_pos2 = np.load(os.path.join(FLAGS.export_path, 'test_pos2.npy'))
        self.data_test_mask = np.load(os.path.join(FLAGS.export_path, 'test_mask.npy'))

        print 'reading finished'
        print 'mentions         : %d' % (len(self.data_instance_triple))
        print 'sentences        : %d' % (len(self.data_test_length))
        print 'relations        : %d' % (FLAGS.num_classes)
        print 'word size        : %d' % (FLAGS.word_size)
        print 'position size     : %d' % (FLAGS.pos_size)
        print 'hidden size        : %d' % (FLAGS.hidden_size)

    def init_train_model(self, loss, output, optimizer=tf.train.GradientDescentOptimizer, pretrain_model=None):
        print 'initializing training model...'
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

    def init_test_model(self, x, output):
        print 'initializing test model...'
        self.x = x
        self.output = output
        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=None)
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

    def test_one_step(self, index, scope, label, result_needed=[]):
        feed_dict = {
            self.word: self.data_test_word[index, :],
            self.word_vec: self.data_word_vec,
            self.pos1: self.data_test_pos1[index, :],
            self.pos2: self.data_test_pos2[index, :],
            self.mask: self.data_test_mask[index, :],
            self.length: self.data_test_length,
            self.label: label,
            self.label_for_select: self.data_test_label[index],
            self.scope: np.array(scope),
        }
        result = self.sess.run([self.output, self.x] + result_needed, feed_dict)
        self.test_output = result[0]
        self.test_x = result[1]
        result = result[2:]

        return result
    
    def train(self, one_step=train_one_step):
        train_order = range(len(self.data_instance_triple))
        for epoch in range(FLAGS.max_epoch):
            print('epoch ' + str(epoch + 1) + ' starts...')
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
                sys.stdout.write("epoch %d step %d time %s | loss : %f, NA accuracy: %f, not NA accuracy: %f, total accuracy %f" % (epoch, i, time_str, loss[0], self.acc_NA.get(), self.acc_not_NA.get(), self.acc_total.get()) + '\r')
                sys.stdout.flush()

            if (epoch + 1) % FLAGS.save_epoch == 0:
                print 'epoch ' + str(epoch + 1) + ' has finished'
                print 'saving model...'
                path = self.saver.save(self.sess, os.path.join(FLAGS.checkpoint_dir, 'checkpoint'), global_step=epoch)
                print 'have saved model to ' + path

    def test(self, epoch_range, one_step=test_one_step):
        for epoch in epoch_range:
            print 'start testing checkpoint, iteration =', epoch
            self.saver.restore(self.sess, os.path.join(FLAGS.checkpoint_dir, 'checkpoint-' + str(epoch)))
            stack_x = []
            stack_label = []
            for i in range(int(len(self.data_instance_scope) / FLAGS.test_batch_size)):
                input_scope = self.data_instance_scope[i * FLAGS.test_batch_size:(i + 1) * FLAGS.test_batch_size]
                index = []
                scope = [0]
                label = []
                for num in input_scope:
                    index = index + range(num[0], num[1] + 1) 
                    label.append(self.data_test_label[num[0]])
                    scope.append(scope[len(scope) - 1] + num[1] - num[0] + 1)

                one_step(self, index, scope, label, [])
                stack_x.append(self.test_x)
                stack_label.append(label)
                assert(len(self.test_x) == len(label))
            
            print 'evaluating...'
            
            stack_x = np.concatenate(stack_x, axis=0)
            stack_label = np.hstack(stack_label)
            exclude_na_flatten_output = stack_x

            exclude_na_flatten_label = stack_label

            # np.save(os.path.join(FLAGS.test_result_dir, 'test_result' + '_' + str(epoch) + '.npy'), exclude_na_flatten_output)
            # np.save(os.path.join(FLAGS.test_result_dir, 'test_label.npy'), exclude_na_flatten_label)

            average_precision = average_precision_score(exclude_na_flatten_label, exclude_na_flatten_output, average="micro")
            print 'average precision:', average_precision

            pr = draw_pr_plot(stack_label, stack_x)
            np.savetxt('pr%d.txt' % epoch, pr)

def average_precision_score(labels, output, average='micro'):
    return 0.0

def draw_pr_plot(labels, output):
    N = len(output)
    K = output.shape[1]
    candidates = []
    for i in range(0, N):
        for j in range(1, K):
            candidates.append((output[i, j], j, i))
    
    cnt_rel = sum([(labels[i] != 0) for i in range(0, N)])
    cnt = 0
    res = []
    for past, (_, L, idx) in enumerate(sorted(candidates, reverse=True)):
        if labels[idx] == L:
            cnt += 1
            res.append([float(cnt) / cnt_rel, float(cnt) / (past + 1)])
    return np.array(res)
