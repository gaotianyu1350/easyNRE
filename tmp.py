import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

FLAGS = tf.app.flags.FLAGS

class NN(object):

    def __init__(self, is_training, word_embeddings, simple_position = False):
        self.max_length = FLAGS.max_length
        self.num_classes = FLAGS.num_classes
        # weights_regularizer = tf.contrib.layers.l2_regularizer(FLAGS.weight_decay)
        self.word = tf.placeholder(dtype=tf.int32,shape=[None, self.max_length], name='input_word')
        self.pos1 = tf.placeholder(dtype=tf.int32,shape=[None, self.max_length], name='input_pos1')
        self.pos2 = tf.placeholder(dtype=tf.int32,shape=[None, self.max_length], name='input_pos2')
        self.mask = tf.placeholder(dtype=tf.int32,shape=[None, self.max_length],name='input_mask')
        self.len = tf.placeholder(dtype=tf.int32,shape=[None],name='input_len')
        self.label_index = tf.placeholder(dtype=tf.int32,shape=[None], name='label_index')
        self.label = tf.placeholder(dtype=tf.float32,shape=[FLAGS.batch_size, self.num_classes], name='input_label')
        self.scope = tf.placeholder(dtype=tf.int32,shape=[FLAGS.batch_size+1], name='scope')    
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
        self.weights = tf.placeholder(dtype=tf.float32,shape=[FLAGS.batch_size])

        with tf.name_scope("embedding-lookup"):
            temp_word_embedding = tf.get_variable(initializer=word_embeddings,name = 'temp_word_embedding',dtype=tf.float32)
            unk_word_embedding = tf.get_variable('unk_embedding',[len(word_embeddings[0])], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
            word_embedding = tf.concat([temp_word_embedding,tf.reshape(unk_word_embedding,[1,len(word_embeddings[0])]),tf.reshape(tf.constant(np.zeros(len(word_embeddings[0]),dtype=np.float32)),[1,len(word_embeddings[0])])],0)

            if simple_position:
                temp_pos_array = np.zeros((FLAGS.pos_num + 1, FLAGS.pos_size), dtype=np.float32)
                temp_pos_array[(FLAGS.pos_num - 1) / 2] = np.ones(FLAGS.pos_size, dtype=np.float32)
                pos1_embedding = tf.constant(temp_pos_array)
                pos2_embedding = tf.constant(temp_pos_array)
            else:
                temp_pos1_embedding = tf.get_variable('temp_pos1_embedding',[FLAGS.pos_num,FLAGS.pos_size],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
                pos1_embedding = tf.concat([temp_pos1_embedding,tf.reshape(tf.constant(np.zeros(FLAGS.pos_size,dtype=np.float32)),[1, FLAGS.pos_size])],0)
                temp_pos2_embedding = tf.get_variable('temp_pos2_embedding',[FLAGS.pos_num,FLAGS.pos_size],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
                pos2_embedding = tf.concat([temp_pos2_embedding,tf.reshape(tf.constant(np.zeros(FLAGS.pos_size,dtype=np.float32)),[1, FLAGS.pos_size])],0)

            self.input_word = tf.nn.embedding_lookup(word_embedding, self.word)
            self.input_pos1 = tf.nn.embedding_lookup(pos1_embedding, self.pos1)
            self.input_pos2 = tf.nn.embedding_lookup(pos2_embedding, self.pos2)
            self.input_embedding = tf.concat(values = [self.input_word, self.input_pos1, self.input_pos2], axis = 2)

class CNN(NN):

    def __init__(self, is_training, word_embeddings, simple_position = False):
        NN.__init__(self, is_training, word_embeddings, simple_position)

        with tf.name_scope("conv-maxpool"):
            input_sentence = tf.expand_dims(self.input_embedding, axis=1)
            x = tf.layers.conv2d(inputs = input_sentence, filters=FLAGS.hidden_size, kernel_size=[1,3], strides=[1, 1], padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d()) 
            x = tf.reduce_max(x, axis=2)
            x = tf.nn.relu(tf.squeeze(x))

        with tf.name_scope("sentence-level-attention"):
            relation_matrix = tf.get_variable('relation_matrix',[self.num_classes, FLAGS.hidden_size],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('bias',[self.num_classes],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
            current_attention = tf.nn.embedding_lookup(relation_matrix, self.label_index)
            attention_logit = tf.reduce_sum(current_attention * x, 1)
            tower_repre = []
            for i in range(FLAGS.batch_size):
                sen_matrix = x[self.scope[i]:self.scope[i+1]]
                attention_score = tf.nn.softmax(tf.reshape(attention_logit[self.scope[i]:self.scope[i+1]], [1, -1]))
                final_repre = tf.reshape(tf.matmul(attention_score, sen_matrix),[FLAGS.hidden_size])
                tower_repre.append(final_repre)
            stack_repre = tf.layers.dropout(tf.stack(tower_repre), rate = self.keep_prob, training = is_training)

        with tf.name_scope("loss"):
            logits = tf.matmul(stack_repre, tf.transpose(relation_matrix)) + bias
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label,logits=logits))
            self.loss = tf.losses.softmax_cross_entropy(onehot_labels = self.label, logits = logits, weights = self.weights)
            self.output = tf.nn.softmax(logits)
            tf.summary.scalar('loss',self.loss)
            self.predictions = tf.argmax(logits, 1, name="predictions")
            self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")

        if not is_training:
            with tf.name_scope("test"):
                test_attention_logit = tf.matmul(x, tf.transpose(relation_matrix))
                test_tower_output = []
                for i in range(FLAGS.test_batch_size):
                    test_attention_score = tf.nn.softmax(tf.transpose(test_attention_logit[self.scope[i]:self.scope[i+1],:]))
                    final_repre = tf.matmul(test_attention_score, x[self.scope[i]:self.scope[i+1]])
                    logits = tf.matmul(final_repre, tf.transpose(relation_matrix)) + bias
                    output = tf.diag_part(tf.nn.softmax(logits))
                    test_tower_output.append(output)
                test_stack_output = tf.reshape(tf.stack(test_tower_output),[FLAGS.test_batch_size, self.num_classes])
                self.test_output = test_stack_output

class PCNN_ADV(NN):
    def __init__(self, is_training, word_embeddings, simple_position = False):
        NN.__init__(self, is_training, word_embeddings, simple_position)

        if is_training:
            input_embedding = tf.concat(values = [self.input_word, self.input_pos1, self.input_pos2], axis = 2)
            raw_loss, raw_output, raw_correct_predictions = self.basic_model(False, is_training, input_embedding)
            raw_perturb = tf.gradients(raw_loss, self.input_word)
            raw_perturb = tf.reshape(raw_perturb, [-1, self.max_length, 50])
            perturb = 0.01 * tf.stop_gradient(tf.nn.l2_normalize(raw_perturb, dim=[0, 1, 2]))
            input_word_adv = self.input_word + perturb
            input_embedding_adv = tf.concat(values = [input_word_adv, self.input_pos1, self.input_pos2], axis = 2)
            self.loss, self.output, self.correct_predictions = self.basic_model(True, is_training, input_embedding_adv)
        else:
            input_embedding = tf.concat(values = [self.input_word, self.input_pos1, self.input_pos2], axis = 2)
            self.loss, self.output, self.correct_predictions = self.basic_model(False, is_training, input_embedding)

    def basic_model(self, reuse, is_training, input_embedding):
        with tf.variable_scope("ATT_PCNN_ADV", reuse=reuse):
            with tf.name_scope("conv-maxpool"):
                mask_embedding = tf.constant([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
                pcnn_mask = tf.nn.embedding_lookup(mask_embedding, self.mask)
                input_sentence = tf.expand_dims(input_embedding, axis=1)
                x = tf.layers.conv2d(inputs = input_sentence, filters=FLAGS.hidden_size, kernel_size=[1,3], strides=[1, 1], padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
                x = tf.reshape(x, [-1, self.max_length, FLAGS.hidden_size, 1])
                x = tf.reduce_max(tf.reshape(pcnn_mask, [-1, 1, self.max_length, 3]) * tf.transpose(x,[0, 2, 1, 3]), axis = 2)
                x = tf.nn.relu(tf.reshape(x,[-1, FLAGS.hidden_size * 3]))

            with tf.name_scope("sentence-level-attention"):
                relation_matrix = tf.get_variable('relation_matrix',[self.num_classes, FLAGS.hidden_size * 3],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
                bias = tf.get_variable('bias',[self.num_classes],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
                current_attention = tf.nn.embedding_lookup(relation_matrix, self.label_index)
                attention_logit = tf.reduce_sum(current_attention * x, 1)
                tower_repre = []
                for i in range(FLAGS.batch_size):
                    sen_matrix = x[self.scope[i]:self.scope[i+1]]
                    attention_score = tf.nn.softmax(tf.reshape(attention_logit[self.scope[i]:self.scope[i+1]], [1, -1]))
                    final_repre = tf.reshape(tf.matmul(attention_score, sen_matrix),[FLAGS.hidden_size * 3])
                    tower_repre.append(final_repre)
                stack_repre = tf.layers.dropout(tf.stack(tower_repre), rate = self.keep_prob, training = is_training)

            with tf.name_scope("loss"):
                logits = tf.matmul(stack_repre, tf.transpose(relation_matrix)) + bias
                # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label,logits=logits))
                loss = tf.losses.softmax_cross_entropy(onehot_labels = self.label, logits = logits, weights = self.weights)
                output = tf.nn.softmax(logits)
                tf.summary.scalar('loss',loss)
                predictions = tf.argmax(logits, 1, name="predictions")
                correct_predictions = tf.equal(predictions, tf.argmax(self.label, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

            if not is_training:
                with tf.name_scope("test"):
                    test_attention_logit = tf.matmul(x, tf.transpose(relation_matrix))
                    test_tower_output = []
                    for i in range(FLAGS.test_batch_size):
                        test_attention_score = tf.nn.softmax(tf.transpose(test_attention_logit[self.scope[i]:self.scope[i+1],:]))
                        final_repre = tf.matmul(test_attention_score, x[self.scope[i]:self.scope[i+1]])
                        logits = tf.matmul(final_repre, tf.transpose(relation_matrix)) + bias
                        output = tf.diag_part(tf.nn.softmax(logits))
                        test_tower_output.append(output)
                    test_stack_output = tf.reshape(tf.stack(test_tower_output),[FLAGS.test_batch_size, self.num_classes])
                    self.test_output = test_stack_output
            return loss, output, correct_predictions

class PCNN(NN):

    def __init__(self, is_training, word_embeddings, simple_position = False):
        NN.__init__(self, is_training, word_embeddings, simple_position)
        with tf.name_scope("conv-maxpool"):
            mask_embedding = tf.constant([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
            pcnn_mask = tf.nn.embedding_lookup(mask_embedding, self.mask)
            input_sentence = tf.expand_dims(self.input_embedding, axis=1)
            x = tf.layers.conv2d(inputs = input_sentence, filters=FLAGS.hidden_size, kernel_size=[1,3], strides=[1, 1], padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
            x = tf.reshape(x, [-1, self.max_length, FLAGS.hidden_size, 1])
            x = tf.reduce_max(tf.reshape(pcnn_mask, [-1, 1, self.max_length, 3]) * tf.transpose(x,[0, 2, 1, 3]), axis = 2)
            x = tf.nn.relu(tf.reshape(x,[-1, FLAGS.hidden_size * 3]))

        with tf.name_scope("sentence-level-attention"):
            relation_matrix = tf.get_variable('relation_matrix',[self.num_classes, FLAGS.hidden_size * 3],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('bias',[self.num_classes],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
            current_attention = tf.nn.embedding_lookup(relation_matrix, self.label_index)
            attention_logit = tf.reduce_sum(current_attention * x, 1)
            tower_repre = []
            for i in range(FLAGS.batch_size):
                sen_matrix = x[self.scope[i]:self.scope[i+1]]
                attention_score = tf.nn.softmax(tf.reshape(attention_logit[self.scope[i]:self.scope[i+1]], [1, -1]))
                final_repre = tf.reshape(tf.matmul(attention_score, sen_matrix),[FLAGS.hidden_size * 3])
                tower_repre.append(final_repre)
            stack_repre = tf.layers.dropout(tf.stack(tower_repre), rate = self.keep_prob, training = is_training)

        with tf.name_scope("loss"):
            logits = tf.matmul(stack_repre, tf.transpose(relation_matrix)) + bias
            # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label,logits=logits))
            self.loss = tf.losses.softmax_cross_entropy(onehot_labels = self.label, logits = logits, weights = self.weights)
            self.output = tf.nn.softmax(logits)
            tf.summary.scalar('loss',self.loss)
            self.predictions = tf.argmax(logits, 1, name="predictions")
            self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")

        if not is_training:
            with tf.name_scope("test"):
                test_attention_logit = tf.matmul(x, tf.transpose(relation_matrix))
                test_tower_output = []
                for i in range(FLAGS.test_batch_size):
                    test_attention_score = tf.nn.softmax(tf.transpose(test_attention_logit[self.scope[i]:self.scope[i+1],:]))
                    final_repre = tf.matmul(test_attention_score, x[self.scope[i]:self.scope[i+1]])
                    logits = tf.matmul(final_repre, tf.transpose(relation_matrix)) + bias
                    output = tf.diag_part(tf.nn.softmax(logits))
                    test_tower_output.append(output)
                test_stack_output = tf.reshape(tf.stack(test_tower_output),[FLAGS.test_batch_size, self.num_classes])
                self.test_output = test_stack_output

class PCNN_SOFT(NN):

    def __init__(self, is_training, word_embeddings, simple_position = False):
        NN.__init__(self, is_training, word_embeddings, simple_position)

        with tf.name_scope("conv-maxpool"):
            mask_embedding = tf.constant([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
            pcnn_mask = tf.nn.embedding_lookup(mask_embedding, self.mask)
            input_sentence = tf.expand_dims(self.input_embedding, axis=1)
            x = tf.layers.conv2d(inputs = input_sentence, filters=FLAGS.hidden_size, kernel_size=[1,3], strides=[1, 1], padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b = tf.get_variable("conv-b", shape=[FLAGS.hidden_size], initializer=tf.contrib.layers.xavier_initializer())
            h = tf.nn.tanh(tf.nn.bias_add(x, b))

            self.num_filters = FLAGS.hidden_size
            seq_len = self.len
            
            npem1 = np.zeros([4, FLAGS.hidden_size], dtype=np.float32)
            npem1 -= 100
            npem1[1] = 0
            npem2 = np.zeros([4, FLAGS.hidden_size], dtype=np.float32)
            npem2 -= 100
            npem2[2] = 0
            npem3 = np.zeros([4, FLAGS.hidden_size], dtype=np.float32)
            npem3 -= 100
            npem3[3] = 0
       
            em1 = tf.constant(npem1, dtype=np.float32)
            em2 = tf.constant(npem2, dtype=np.float32)
            em3 = tf.constant(npem3, dtype=np.float32)
            
            self.mask1 = tf.nn.embedding_lookup(em1, self.mask)
            self.mask2 = tf.nn.embedding_lookup(em2, self.mask)
            self.mask3 = tf.nn.embedding_lookup(em3, self.mask)
        
            #self.h1 = tf.add(h, tf.expand_dims(self.mask1, 2))
            #self.h2 = tf.add(h, tf.expand_dims(self.mask1, 2))
            #self.h3 = tf.add(h, tf.expand_dims(self.mask1, 2))

            self.h1 = tf.add(h, tf.expand_dims(self.mask1, 1))
            self.h2 = tf.add(h, tf.expand_dims(self.mask2, 1))
            self.h3 = tf.add(h, tf.expand_dims(self.mask3, 1))

            pooled1 = tf.nn.max_pool(self.h1, ksize=[1, 1, FLAGS.max_length, 1], strides=[1, 1, 1, 1], padding="VALID",name="pool")
            poolre1 = tf.reshape(pooled1, [-1, FLAGS.hidden_size])
            pooled2 = tf.nn.max_pool(self.h2, ksize=[1, 1, FLAGS.max_length, 1], strides=[1, 1, 1, 1], padding="VALID",name="pool")
            poolre2 = tf.reshape(pooled2, [-1, FLAGS.hidden_size])
            pooled3 = tf.nn.max_pool(self.h3, ksize=[1, 1, FLAGS.max_length, 1], strides=[1, 1, 1, 1], padding="VALID",name="pool")
            poolre3 = tf.reshape(pooled3, [-1, FLAGS.hidden_size])
            poolre = tf.concat([poolre1, poolre2, poolre3], 1)
            x = tf.nn.dropout(poolre, FLAGS.keep_prob)

            #x = tf.reshape(x, [-1, self.max_length, FLAGS.hidden_size, 1])
            #x = tf.reduce_max(tf.reshape(pcnn_mask, [-1, 1, self.max_length, 3]) * tf.transpose(x,[0, 2, 1, 3]), axis = 2)
            #x = tf.nn.relu(tf.reshape(x,[-1, FLAGS.hidden_size * 3]))

        with tf.name_scope("sentence-level-attention"):
            relation_matrix = tf.get_variable('relation_matrix',[self.num_classes, FLAGS.hidden_size * 3],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('bias',[self.num_classes],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
            current_attention = tf.nn.embedding_lookup(relation_matrix, self.label_index)
            attention_logit = tf.reduce_sum(current_attention * x, 1)
            tower_repre = []
            for i in range(FLAGS.batch_size):
                sen_matrix = x[self.scope[i]:self.scope[i+1]]
                attention_score = tf.nn.softmax(tf.reshape(attention_logit[self.scope[i]:self.scope[i+1]], [1, -1]))
                final_repre = tf.reshape(tf.matmul(attention_score, sen_matrix),[FLAGS.hidden_size * 3])
                tower_repre.append(final_repre)
            stack_repre = tf.layers.dropout(tf.stack(tower_repre), rate = self.keep_prob, training = is_training)

        with tf.name_scope("loss"):
            #logits = tf.matmul(stack_repre, tf.transpose(relation_matrix)) + bias
            #self.output = tf.nn.softmax(logits)
            #rate_np = np.ones((self.num_classes), dtype=np.float32)
            #rate_np *= 0.9
            ##rate_np[0] = 0.9
            #rate = tf.convert_to_tensor(rate_np)
            #self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label,logits=logits))
            #tf.summary.scalar('loss',self.loss)
            #self.predictions = tf.argmax(logits, 1, name="predictions")
            #self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.label, 1))
            #self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")

            logits = tf.matmul(stack_repre, tf.transpose(relation_matrix)) + bias
            self.output = tf.nn.softmax(logits)
            rate_np = np.ones((self.num_classes), dtype=np.float32)
            rate_np *= 0.9
            #rate_np[0] = 0.9
            rate = tf.convert_to_tensor(rate_np)
            self.nscore = self.output + 0.9 * tf.reshape(tf.reduce_max(self.output, 1), [-1, 1]) * tf.cast(self.label, tf.float32)
            self.nlabel = tf.one_hot(indices=tf.reshape(tf.argmax(self.nscore, axis=1), [-1]), depth=self.num_classes, dtype=tf.int32)
            self.nlabel_result = tf.argmax(self.nlabel, 1)
            # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label,logits=logits))
            self.loss = tf.losses.softmax_cross_entropy(onehot_labels = self.nlabel, logits = logits, weights = self.weights)
            tf.summary.scalar('loss',self.loss)
            self.predictions = tf.argmax(logits, 1, name="predictions")
            self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")


        if not is_training:
            with tf.name_scope("test"):
                test_attention_logit = tf.matmul(x, tf.transpose(relation_matrix))
                test_tower_output = []
                for i in range(FLAGS.test_batch_size):
                    test_attention_score = tf.nn.softmax(tf.transpose(test_attention_logit[self.scope[i]:self.scope[i+1],:]))
                    final_repre = tf.matmul(test_attention_score, x[self.scope[i]:self.scope[i+1]])
                    logits = tf.matmul(final_repre, tf.transpose(relation_matrix)) + bias
                    output = tf.diag_part(tf.nn.softmax(logits))
                    test_tower_output.append(output)
                test_stack_output = tf.reshape(tf.stack(test_tower_output),[FLAGS.test_batch_size, self.num_classes])
                self.test_output = test_stack_output

class RNN(NN):

    def get_rnn_cell(self, dim, cell_name = 'lstm'):
        if isinstance(cell_name,list) or isinstance(cell_name, tuple):
            if len(cell_name) == 1:
                return get_rnn_cell(dim, cell_name[0])
            cells = [get_rnn_cell(dim, c) for c in cell_name]
            return tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
        if cell_name.lower() == 'lstm':
            return tf.contrib.rnn.BasicLSTMCell(dim, state_is_tuple=True)
        elif cell_name.lower() == 'gru':
            return tf.contrib.rnn.GRUCell(dim)
        raise NotImplementedError

    def __init__(self, is_training, word_embeddings, cell_name, simple_position = False):
        NN.__init__(self, is_training, word_embeddings, simple_position)
        input_sentence = tf.layers.dropout(self.input_embedding, rate = self.keep_prob, training = is_training)
        # input_sentence = self.input_embedding
        with tf.name_scope('rnn'):
            cell = self.get_rnn_cell(FLAGS.hidden_size, cell_name)
            outputs, states = tf.nn.dynamic_rnn(cell, input_sentence,
                                            sequence_length = self.len,
                                            dtype = tf.float32,
                                            scope = 'dynamic-rnn')
            if isinstance(states, tuple):
                states = states[0]
            x = states

        with tf.name_scope("sentence-level-attention"):
            relation_matrix = tf.get_variable('relation_matrix',[self.num_classes, FLAGS.hidden_size],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('bias',[self.num_classes],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
            current_attention = tf.nn.embedding_lookup(relation_matrix, self.label_index)
            attention_logit = tf.reduce_sum(current_attention * x, 1)
            tower_repre = []
            for i in range(FLAGS.batch_size):
                sen_matrix = x[self.scope[i]:self.scope[i+1]]
                attention_score = tf.nn.softmax(tf.reshape(attention_logit[self.scope[i]:self.scope[i+1]], [1, -1]))
                final_repre = tf.reshape(tf.matmul(attention_score, sen_matrix),[FLAGS.hidden_size])
                tower_repre.append(final_repre)
            # stack_repre = tf.layers.dropout(tf.stack(tower_repre), rate = self.keep_prob, training = is_training)
            stack_repre = tf.stack(tower_repre)

        with tf.name_scope("loss"):
            l2_loss = tf.constant(0.0)
            l2_loss += tf.nn.l2_loss(relation_matrix)
            l2_loss += tf.nn.l2_loss(bias)
            logits = tf.matmul(stack_repre, tf.transpose(relation_matrix)) + bias
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label,logits=logits))
            self.output = tf.nn.softmax(logits)
            tf.summary.scalar('loss',self.loss)
            self.predictions = tf.argmax(logits, 1, name="predictions")
            self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")

        if not is_training:
            with tf.name_scope("test"):
                test_attention_logit = tf.matmul(x, tf.transpose(relation_matrix))
                test_tower_output = []
                for i in range(FLAGS.test_batch_size):
                    test_attention_score = tf.nn.softmax(tf.transpose(test_attention_logit[self.scope[i]:self.scope[i+1],:]))
                    final_repre = tf.matmul(test_attention_score, x[self.scope[i]:self.scope[i+1]])
                    logits = tf.matmul(final_repre, tf.transpose(relation_matrix)) + bias
                    output = tf.diag_part(tf.nn.softmax(logits))
                    test_tower_output.append(output)
                test_stack_output = tf.reshape(tf.stack(test_tower_output),[FLAGS.test_batch_size, self.num_classes])
                self.test_output = test_stack_output

class BiRNN(NN):

    def get_rnn_cell(self, dim, cell_name = 'lstm'):
        if isinstance(cell_name,list) or isinstance(cell_name, tuple):
            if len(cell_name) == 1:
                return get_rnn_cell(dim, cell_name[0])
            cells = [get_rnn_cell(dim, c) for c in cell_name]
            return tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
        if cell_name.lower() == 'lstm':
            return tf.contrib.rnn.BasicLSTMCell(dim, state_is_tuple=True)
        elif cell_name.lower() == 'gru':
            return tf.contrib.rnn.GRUCell(dim)
        raise NotImplementedError

    def __init__(self, is_training, word_embeddings, cell_name, simple_position = False):
        NN.__init__(self, is_training, word_embeddings, simple_position)
        input_sentence = tf.layers.dropout(self.input_embedding, rate = self.keep_prob, training = is_training)
        with tf.name_scope('bi-rnn'):
            fw_cell = self.get_rnn_cell(FLAGS.hidden_size, cell_name)
            bw_cell = self.get_rnn_cell(FLAGS.hidden_size, cell_name)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                            fw_cell, bw_cell, input_sentence,
                            sequence_length = self.len,
                            dtype = tf.float32,
                            scope = 'bi-dynamic-rnn')
            fw_states, bw_states = states
            if isinstance(fw_states, tuple):
                fw_states = fw_states[0]
                bw_states = bw_states[0]
            x = tf.concat(states, axis=1)

        with tf.name_scope("sentence-level-attention"):
            relation_matrix = tf.get_variable('relation_matrix',[self.num_classes, FLAGS.hidden_size * 2],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('bias',[self.num_classes],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
            current_attention = tf.nn.embedding_lookup(relation_matrix, self.label_index)
            attention_logit = tf.reduce_sum(current_attention * x, 1)
            tower_repre = []
            for i in range(FLAGS.batch_size):
                sen_matrix = x[self.scope[i]:self.scope[i+1]]
                attention_score = tf.nn.softmax(tf.reshape(attention_logit[self.scope[i]:self.scope[i+1]], [1, -1]))
                final_repre = tf.reshape(tf.matmul(attention_score, sen_matrix),[FLAGS.hidden_size * 2])
                tower_repre.append(final_repre)
            #stack_repre = tf.layers.dropout(tf.stack(tower_repre), rate = 1.0, training = is_training)
            stack_repre = tf.stack(tower_repre)
        with tf.name_scope("loss"):
            logits = tf.matmul(stack_repre, tf.transpose(relation_matrix)) + bias
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label,logits=logits))
            self.output = tf.nn.softmax(logits)
            tf.summary.scalar('loss',self.loss)
            self.predictions = tf.argmax(logits, 1, name="predictions")
            self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")

        if not is_training:
            with tf.name_scope("test"):
                test_attention_logit = tf.matmul(x, tf.transpose(relation_matrix))
                test_tower_output = []
                for i in range(FLAGS.test_batch_size):
                    test_attention_score = tf.nn.softmax(tf.transpose(test_attention_logit[self.scope[i]:self.scope[i+1],:]))
                    final_repre = tf.matmul(test_attention_score, x[self.scope[i]:self.scope[i+1]])
                    logits = tf.matmul(final_repre, tf.transpose(relation_matrix)) + bias
                    output = tf.diag_part(tf.nn.softmax(logits))
                    test_tower_output.append(output)
                test_stack_output = tf.reshape(tf.stack(test_tower_output),[FLAGS.test_batch_size, self.num_classes])
                self.test_output = test_stack_output

