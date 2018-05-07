from framework import Framework
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def pcnn_att_adv(is_training):
    with tf.variable_scope('pcnn_att_adv', reuse=False): 
        framework = Framework(is_training=False)
        word_embedding = framework.embedding.word_embedding()
        pos_embedding = framework.embedding.pos_embedding()
        embedding = framework.embedding.concat_embedding(word_embedding, pos_embedding)
        x = framework.encoder.pcnn(embedding, activation=tf.nn.relu)
        x = framework.selector.attention(x)

    if is_training:
        # Add perturbation
        loss = framework.classifier.softmax_cross_entropy(x)
        perturb = tf.gradients(loss, embedding)
        perturb = tf.reshape((0.01 * tf.stop_gradient(tf.nn.l2_normalize(perturb, dim=[0, 1, 2]))), [-1, FLAGS.max_length, FLAGS.word_size + 2 * FLAGS.pos_size])
        embedding = embedding + perturb
        
        # Train
        with tf.variable_scope('pcnn_att_adv', reuse=True): 
            framework.is_training = True
            x = framework.encoder.pcnn(embedding, activation=tf.nn.relu)
            x = framework.selector.attention(x)
            loss = framework.classifier.softmax_cross_entropy(x)
            output = framework.classifier.output(x)
        framework.init_train_model(loss, output, optimizer=tf.train.GradientDescentOptimizer)
        framework.load_train_data()
        framework.train()
    else:
        framework.init_test_model(x)
        framework.load_test_data()
        framework.test()

