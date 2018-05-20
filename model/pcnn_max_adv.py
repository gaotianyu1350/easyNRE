from framework import Framework
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def pcnn_max_adv(is_training):
    if is_training:
        with tf.variable_scope('pcnn_max_adv', reuse=False): 
            framework = Framework(is_training=True)
            word_embedding = framework.embedding.word_embedding()
            pos_embedding = framework.embedding.pos_embedding()
            embedding = framework.embedding.concat_embedding(word_embedding, pos_embedding)
            x = framework.encoder.pcnn(embedding, activation=tf.nn.relu)
            x = framework.selector.maximum(x)

        # Add perturbation
        loss = framework.classifier.softmax_cross_entropy(x)
        embedding = framework.adversarial(loss, embedding)
        
        # Train
        with tf.variable_scope('pcnn_max_adv', reuse=True): 
            x = framework.encoder.pcnn(embedding, activation=tf.nn.relu)
            x = framework.selector.maximum(x)
            loss = framework.classifier.softmax_cross_entropy(x)
            output = framework.classifier.output(x)
        framework.init_train_model(loss, output, optimizer=tf.train.GradientDescentOptimizer)
        framework.load_train_data()
        framework.train()
    else:
        with tf.variable_scope('pcnn_max_adv', reuse=False): 
            framework = Framework(is_training=False)
            word_embedding = framework.embedding.word_embedding()
            pos_embedding = framework.embedding.pos_embedding()
            embedding = framework.embedding.concat_embedding(word_embedding, pos_embedding)
            x = framework.encoder.pcnn(embedding, activation=tf.nn.relu)
            x = framework.selector.maximum(x)

        framework.init_test_model(tf.nn.softmax(x))
        framework.load_test_data()
        framework.test()

