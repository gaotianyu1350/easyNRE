import tensorflow as tf
import numpy as np
import time
import datetime
import os
import network
import json
from sklearn.metrics import average_precision_score
import sys

export_path = "../data/"

word_vec = np.load(export_path + 'vec.npy')
f = open(export_path + "config", 'r')
config = json.loads(f.read())
f.close()

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('max_length', config['fixlen'], 'maximum of number of words in one sentence')
tf.app.flags.DEFINE_float('pos_num', config['maxlen'] * 2 + 1, 'number of position embedding vectors')
tf.app.flags.DEFINE_float('num_classes', len(config['relation2id']),'maximum of relations')

tf.app.flags.DEFINE_float('hidden_size',230,'hidden feature size')
tf.app.flags.DEFINE_float('pos_size',5,'position embedding size')

tf.app.flags.DEFINE_float('max_epoch',60,'maximum of training epochs')
tf.app.flags.DEFINE_float('batch_size',160,'entity numbers used each training time')
tf.app.flags.DEFINE_float('learning_rate',0.5,'entity numbers used each training time')
tf.app.flags.DEFINE_float('weight_decay',0.00001,'weight_decay')
tf.app.flags.DEFINE_float('keep_prob',0.7,'dropout rate')

tf.app.flags.DEFINE_string('model_dir','./model/','path to store model')
tf.app.flags.DEFINE_string('summary_dir','./summary','path to store summary_dir')
tf.app.flags.DEFINE_string('use_adv', False, 'use adversarial training or not')


def MakeSummary(name, value):
    """Creates a tf.Summary proto with the given name and value."""
    summary = tf.Summary()
    val = summary.value.add()
    val.tag = str(name)
    val.simple_value = float(value)
    return summary


def make_shape(array,last_dim):
    output = []
    for i in array:
        for j in i:
            output.append(j)
    output = np.array(output)
    if np.shape(output)[-1]==last_dim:
        return output
    else:
        print 'Make Shape Error!'

def main(_):

    print 'reading word embedding'
    word_vec = np.load(export_path + 'vec.npy')
    print 'reading training data'
    instance_triple = np.load(export_path + 'train_instance_triple.npy')
    instance_scope = np.load(export_path + 'train_instance_scope.npy')
    train_len = np.load(export_path + 'train_len.npy')
    train_label = np.load(export_path + 'train_label.npy')
    train_word = np.load(export_path + 'train_word.npy')
    train_pos1 = np.load(export_path + 'train_pos1.npy')
    train_pos2 = np.load(export_path + 'train_pos2.npy')
    train_mask = np.load(export_path + 'train_mask.npy')
    print 'reading finished'
    print 'mentions         : %d' % (len(instance_triple))
    print 'sentences        : %d' % (len(train_len))
    print 'relations        : %d' % (FLAGS.num_classes)
    print 'word size        : %d' % (len(word_vec[0]))
    print 'position size     : %d' % (FLAGS.pos_size)
    print 'hidden size        : %d' % (FLAGS.hidden_size)
    reltot = {}
    for index, i in enumerate(train_label):
        if not i in reltot:
            reltot[i] = 1.0
        else:
            reltot[i] += 1.0
    for i in reltot:
        reltot[i] = 1/(reltot[i] ** (0.05)) 
    print reltot
    print 'building network...'
    sess = tf.Session()
    if FLAGS.model.lower() == "cnn":
        model = network.CNN(is_training = True, word_embeddings = word_vec)
    elif FLAGS.model.lower() == "pcnn":
        model = network.PCNN(is_training = True, word_embeddings = word_vec)
    elif FLAGS.model.lower() == "lstm":
        model = network.RNN(is_training = True, word_embeddings = word_vec, cell_name = "LSTM", simple_position = True)
    elif FLAGS.model.lower() == "gru":
        model = network.RNN(is_training = True, word_embeddings = word_vec, cell_name = "GRU", simple_position = True)
    elif FLAGS.model.lower() == "bi-lstm" or FLAGS.model.lower() == "bilstm":
        model = network.BiRNN(is_training = True, word_embeddings = word_vec, cell_name = "LSTM", simple_position = True)
    elif FLAGS.model.lower() == "bi-gru" or FLAGS.model.lower() == "bigru":
        model = network.BiRNN(is_training = True, word_embeddings = word_vec, cell_name = "GRU", simple_position = True)
    elif FLAGS.model.lower() == "pcnn_adv":
        model = network.PCNN_ADV(is_training = True, word_embeddings = word_vec)
    elif FLAGS.model.lower() == "pcnn_soft":
        model = network.PCNN_SOFT(is_training = True, word_embeddings = word_vec)
    
    global_step = tf.Variable(0,name='global_step',trainable=False)
    tf.summary.scalar('learning_rate', FLAGS.learning_rate)
    #optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    # train_op = optimizer.minimize(model.ce_loss, global_step=global_step)
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    grads_and_vars = optimizer.compute_gradients(model.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step)
    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=None)
    print 'building finished'

    def train_step(word, pos1, pos2, mask, leng, label_index, label, scope, weights):
        feed_dict = {
            model.word: word,
            model.pos1: pos1,
            model.pos2: pos2,
            model.mask: mask,
            model.len : leng,
            model.label_index: label_index,
            model.label: label,
            model.scope: scope,
            model.keep_prob: FLAGS.keep_prob,
            model.weights: weights
        }
        _, step, loss, summary, output, correct_predictions = sess.run([train_op, global_step, model.loss, merged_summary, model.output, model.correct_predictions], feed_dict)
        summary_writer.add_summary(summary, step)
        return output, loss, correct_predictions

    stack_output = []
    stack_label = []
    stack_ce_loss = []

    train_order = range(len(instance_triple))

    save_epoch = 2
    eval_step = 300
    # print FLAGS.model_dir + FLAGS.model+"-"+str(3664*20)
    # saver.restore(sess, FLAGS.model_dir + FLAGS.model+"-"+str(3664*20))
    for one_epoch in range(FLAGS.max_epoch):

        print('epoch '+str(one_epoch+1)+' starts!')
        
        np.random.shuffle(train_order)
        s1 = 0.0
        s2 = 0.0
        tot1 = 0.0
        tot2 = 0.0
        losstot = 0.0
        for i in range(int(len(train_order)/float(FLAGS.batch_size))):
            input_scope = np.take(instance_scope, train_order[i * FLAGS.batch_size:(i+1)*FLAGS.batch_size], axis=0)
            index = []
            scope = [0]
            label = []
            weights = []
            for num in input_scope:
                index = index + range(num[0], num[1] + 1)
                label.append(train_label[num[0]])
                scope.append(scope[len(scope)-1] + num[1] - num[0] + 1)
                weights.append(reltot[train_label[num[0]]])
            label_ = np.zeros((FLAGS.batch_size, FLAGS.num_classes))
            label_[np.arange(FLAGS.batch_size), label] = 1

            output, loss, correct_predictions = train_step(train_word[index,:], train_pos1[index,:], train_pos2[index,:], train_mask[index,:], train_len[index],train_label[index], label_, np.array(scope), weights)
            num = 0
            s = 0
            losstot += loss
            for num in correct_predictions:
                if label[s] == 0:
                    tot1 += 1.0
                    if num:
                        s1+= 1.0
                else:
                    tot2 += 1.0
                    if num:
                        s2 += 1.0
                s = s + 1

            time_str = datetime.datetime.now().isoformat()
            sys.stdout.write("batch %d step %d time %s | loss : %f, NA accuracy: %f, not NA accuracy: %f" % (one_epoch, i, time_str, loss, s1 / tot1, s2 / tot2)+'\r')
            sys.stdout.flush()

        #     stack_output.append(output)
        #     stack_label.append(input_label)
        #     stack_ce_loss.append(loss)

            current_step = tf.train.global_step(sess, global_step)
        print losstot

        #     if current_step % eval_step==0:

        #         print 'evaluating...'

        #         stack_output = np.reshape(np.array(stack_output),(eval_step*FLAGS.batch_size,FLAGS.num_classes))
        #         stack_label = np.reshape(np.array(stack_label),(eval_step*FLAGS.batch_size,FLAGS.num_classes))

        #         exclude_na_flatten_output = np.reshape(stack_output[:,1:],(-1))
        #         exclude_na_flatten_label = np.reshape(stack_label[:,1:],(-1))

        #         average_precision = average_precision_score(exclude_na_flatten_label,exclude_na_flatten_output)

        #         summary_writer.add_summary(MakeSummary('train/pr',average_precision), current_step)

        #         stack_output = []
        #         stack_label = []
                
        #         print 'pr: '+str(average_precision)

        #         avg_ce_loss = np.mean(np.array(stack_ce_loss))

        #         time_str = datetime.datetime.now().isoformat()
        #         tempstr = "{}: step {}, ce_loss {:g}, avg_ce_loss {:g}".format(time_str, current_step, ce_loss, avg_ce_loss)
        #         print(tempstr)

        #         stack_ce_loss = []

        if (one_epoch + 1) % save_epoch == 0:
            print 'epoch '+str(one_epoch+1)+' has finished'
            print 'saving model...'
            path = saver.save(sess,FLAGS.model_dir+FLAGS.model, global_step=current_step)
            print 'have savde model to '+path

if __name__ == "__main__":
    tf.app.run() 
