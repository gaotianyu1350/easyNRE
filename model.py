from framework import Framework 

def get_pcnn_model(is_training):
    framework = Framework(is_training=is_training)

    word_embedding = framework.embedding.word_embedding()
    pos_embedding = framework.embedding.pos_embedding()
    embedding = framework.embedding.concat_embedding(word_embedding, pos_embedding)
    x = framework.encoder.pcnn(embedding, activation=tf.nn.relu)
    x = framework.selector.attention(x)
    loss = framework.classifier.softmax_cross_entropy(x)
    output = framework.classifier.output(x)
    
    framework.init_model(loss, output, optimizer=tf.train.GradientDescentOptimizer)
    framework.load_data()
    framework.train_bag()

    return framework
