import tensorflow as tf
import numpy as np
import time
import datetime
import os
import network
import json
from sklearn.metrics import average_precision_score
import sys

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('export_path','./data','path to data')

config_file = open(os.path.join(FLAGS.export_path, "config"), 'r')
config = json.loads(config_file.read())
config_file.close()

tf.app.flags.DEFINE_float('max_length', config['fixlen'], 'maximum of number of words in one sentence')
tf.app.flags.DEFINE_float('pos_num', config['maxlen'] * 2 + 1, 'number of position embedding vectors')
tf.app.flags.DEFINE_float('num_classes', len(config['relation2id']),'maximum of relations')

tf.app.flags.DEFINE_float('hidden_size',230,'hidden feature size')
tf.app.flags.DEFINE_float('pos_size',5,'position embedding size')
tf.app.flags.DEFINE_float('word_size', 50, 'word embedding size')

tf.app.flags.DEFINE_float('max_epoch',60,'maximum of training epochs')
tf.app.flags.DEFINE_float('batch_size',160,'entity numbers used each training time')
tf.app.flags.DEFINE_float('learning_rate',0.5,'entity numbers used each training time')
tf.app.flags.DEFINE_float('weight_decay',0.00001,'weight_decay')
tf.app.flags.DEFINE_float('keep_prob',0.7,'dropout rate')

tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'path to store checkpoint')
tf.app.flags.DEFINE_string('summary_dir', './summary', 'path to store summary_dir')
tf.app.flags.DEFINE_string('test_result_dir', './test_result', 'path to store the test results')
tf.app.flags.DEFINE_string('use_adv', False, 'use adversarial training or not')
tf.app.flags.DEFINE_string('save_epoch', 2, 'save the checkpoint after how many epoches')

tf.app.flags.DEFINE_string('model_name', 'pcnn_att', 'model\'s name')

from framework import Framework 
def main(_):
    from model.pcnn_att import pcnn_att
    from model.cnn_att import cnn_att
    from model.pcnn_att_adv import pcnn_att_adv
    from model.pcnn_att_soft_label import pcnn_att_soft_label

    model = locals()[FLAGS.model_name]
    model(is_training=True)

if __name__ == "__main__":
    tf.app.run() 
