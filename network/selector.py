import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

class Selector(object):

    def __init__(self, is_training, scope, label_for_select=None):
        self.scope = scope
        self.label_for_select = label_for_select
        self.is_training = is_training
   
    def no_bag(self, x):
        with tf.name_scope("no_bag"):
            x = tf.layers.dropout(x, rate=FLAGS.drop_prob, training=self.is_training)
            relation_matrix = tf.get_variable('relation_matrix', [FLAGS.num_classes, x.shape[1]], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('bias', [FLAGS.num_classes], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            logits = tf.matmul(x, tf.transpose(relation_matrix)) + bias
            return logits

    def attention(self, x, use_dropout=True, dropout_before=True):
        with tf.name_scope("attention"):
            relation_matrix = tf.get_variable('relation_matrix', [FLAGS.num_classes, x.shape[1]], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('bias', [FLAGS.num_classes], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    
            if self.is_training:
                if use_dropout and dropout_before:
                    x = tf.layers.dropout(x, rate=FLAGS.drop_prob, training=self.is_training)

                current_attention = tf.nn.embedding_lookup(relation_matrix, self.label_for_select)
                attention_logit = tf.reduce_sum(current_attention * x, 1)
                tower_repre = []
                for i in range(FLAGS.batch_size):
                    sen_matrix = x[self.scope[i]:self.scope[i + 1]]
                    attention_score = tf.nn.softmax(tf.reshape(attention_logit[self.scope[i]:self.scope[i + 1]], [1, -1]))
                    final_repre = tf.squeeze(tf.matmul(attention_score, sen_matrix))
                    tower_repre.append(final_repre)
                if use_dropout and not dropout_before:
                    stack_repre = tf.layers.dropout(tf.stack(tower_repre), rate=FLAGS.drop_prob, training=self.is_training)
                else:
                    stack_repre = tf.stack(tower_repre)

                logits = tf.matmul(stack_repre, tf.transpose(relation_matrix)) + bias
                return logits
            else:
                with tf.name_scope("test"):
                    test_attention_logit = tf.matmul(x, tf.transpose(relation_matrix))
                    test_tower_output = []
                    for i in range(FLAGS.batch_size):
                        test_attention_score = tf.nn.softmax(tf.transpose(test_attention_logit[self.scope[i]:self.scope[i+1],:]))
                        final_repre = tf.matmul(test_attention_score, x[self.scope[i]:self.scope[i+1]])
                        logits = tf.matmul(final_repre, tf.transpose(relation_matrix)) + bias
                        output = tf.diag_part(tf.nn.softmax(logits)) * FLAGS.drop_prob
                        test_tower_output.append(output)
                    test_stack_output = tf.reshape(tf.stack(test_tower_output), [FLAGS.batch_size, FLAGS.num_classes])
                    test_output = test_stack_output
                    return test_output 

    def average(self, x, dropout_before=True):
        with tf.name_scope("average"):
            relation_matrix = tf.get_variable('relation_matrix', [FLAGS.num_classes, x.shape[1]], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('bias', [FLAGS.num_classes], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            tower_repre = []
            if dropout_before:
                x = tf.layers.dropout(x, rate=FLAGS.drop_prob, training=self.is_training)
            for i in range(FLAGS.batch_size):
                repre_mat = x[self.scope[i]:self.scope[i + 1]]
                repre = tf.reduce_mean(repre_mat, axis=0)
                tower_repre.append(repre)
            if not dropout_before:
                stack_repre = tf.layers.dropout(tf.stack(tower_repre), rate=FLAGS.drop_prob, training=self.is_training)
            else:
                stack_repre = tf.stack(tower_repre)

            logits = tf.matmul(stack_repre, tf.transpose(relation_matrix)) + bias
            return logits

    def maximum(self, x, dropout_before=True):
        with tf.name_scope("maximum"):
            relation_matrix = tf.get_variable('relation_matrix', [FLAGS.num_classes, x.shape[1]], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('bias', [FLAGS.num_classes], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            tower_repre = []
            if dropout_before:
                x = tf.layers.dropout(x, rate=FLAGS.drop_prob, training=self.is_training)
            for i in range(FLAGS.batch_size):
                repre_mat = x[self.scope[i]:self.scope[i + 1]]
                logit = tf.matmul(repre_mat, tf.transpose(relation_matrix)) + bias
                j = tf.argmax(tf.reduce_max(logit, axis=1))
                tower_repre.append(repre_mat[j])
            if not dropout_before:
                stack_repre = tf.layers.dropout(tf.stack(tower_repre), rate=FLAGS.drop_prob, training=self.is_training)
            else:
                stack_repre = tf.stack(tower_repre)
            logits = tf.matmul(stack_repre, tf.transpose(relation_matrix)) + bias
            return logits

