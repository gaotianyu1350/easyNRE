import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

class Encoder(object):

    def __init__(self, is_training, length=None, mask=None):
        self.mask = mask
        self.is_training = is_training
        self.length = length

    def pcnn(self, x, activation=tf.nn.relu):
        with tf.name_scope("pcnn"):
            mask_embedding = tf.constant([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=np.float32)
            pcnn_mask = tf.nn.embedding_lookup(mask_embedding, self.mask)
            x = tf.expand_dims(x, axis=1)
            x = tf.layers.conv2d(inputs=x, filters=FLAGS.hidden_size, kernel_size=[1,3], strides=[1, 1], padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
            x = tf.reshape(x, [-1, FLAGS.max_length, FLAGS.hidden_size, 1])
            x = tf.reduce_max(tf.reshape(pcnn_mask, [-1, 1, FLAGS.max_length, 3]) * tf.transpose(x,[0, 2, 1, 3]), axis = 2)
            x = activation(tf.reshape(x,[-1, FLAGS.hidden_size * 3]))
            return x

    def cnn(self, x, activation=tf.nn.relu):
        with tf.name_scope("cnn"):
            x = tf.expand_dims(x, axis=1)
            x = tf.layers.conv2d(inputs=x, filters=FLAGS.hidden_size, kernel_size=[1,3], strides=[1, 1], padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d()) 
            x = tf.reduce_max(x, axis=2)
            x = tf.nn.relu(tf.squeeze(x))
            return x

    def __rnn_cell__(self, dim, cell_name='lstm'):
        if isinstance(cell_name, list) or isinstance(cell_name, tuple):
            if len(cell_name) == 1:
                return self.__rnn_cell__(dim, cell_name[0])
            cells = [self.__rnn_cell__(dim, c) for c in cell_name]
            return tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
        if cell_name.lower() == 'lstm':
            return tf.contrib.rnn.BasicLSTMCell(dim, state_is_tuple=True)
        elif cell_name.lower() == 'gru':
            return tf.contrib.rnn.GRUCell(dim)
        raise NotImplementedError

    def rnn(self, x, cell_name='lstm'):
        with tf.name_scope('rnn'):
            x = tf.layers.dropout(x, rate=FLAGS.keep_prob, training=self.is_training)
            cell = self.__rnn_cell__(FLAGS.hidden_size, cell_name)
            _, states = tf.nn.dynamic_rnn(cell, x, sequence_length=self.len, dtype=tf.float32, scope='dynamic-rnn')
            if isinstance(states, tuple):
                states = states[0]
            return states

    def birnn(self, x, cell_name='lstm'):
        with tf.name_scope('bi-rnn'):
            x = tf.layers.dropout(x, rate=FLAGS.keep_prob, training=is_training)
            fw_cell = self.__rnn_cell__(FLAGS.hidden_size, cell_name)
            bw_cell = self.__rnn_cell__(FLAGS.hidden_size, cell_name)
            _, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, sequence_length=self.length, dtype=tf.float32, scope='bi-dynamic-rnn')
            fw_states, bw_states = states
            if isinstance(fw_states, tuple):
                fw_states = fw_states[0]
                bw_states = bw_states[0]
            x = tf.concat(states, axis=1)
            return x 
