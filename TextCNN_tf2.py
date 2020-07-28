'''
Created by: Kevin De Angeli
Email: Kevindeangeli@utk.edu
Date: 7/20/20

TextCNN:
Here, the embeddings are random and are not trainable parameters. (see tf.gather)
'''

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

class Convolutions(layers.Layer):
    #For the feature, make number of convolution layers a variable, where you can seelct num filters, strides for each.
    def __init__(self, num_filters, name="Conv1DLayer", **kwargs):
        super(Convolutions, self).__init__(name=name, **kwargs)
        self.conv3 = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=3,padding='same', activation=tf.nn.relu,
                                            kernel_initializer='glorot_uniform')
        self.conv4 = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=4,padding='same', activation=tf.nn.relu,
                                            kernel_initializer='glorot_uniform')
        self.conv5 = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=5,padding='same', activation=tf.nn.relu,
                                            kernel_initializer='glorot_uniform')

    def call(self, inputs):
        x1 = self.conv3(inputs)
        x2 = self.conv4(inputs)
        x3 = self.conv5(inputs)
        pool3 = tf.reduce_max(x1, 1)
        pool4 = tf.reduce_max(x2, 1)
        pool5 = tf.reduce_max(x3, 1)
        doc_embed = tf.concat([pool3, pool4, pool5], 1)
        return doc_embed


class TextCNNv2(keras.Model):
    def __init__(self,embedding_matrix, num_classes,
                 num_filters=300, dropout_keep=0.5, name="TextCNNv2"):
        super(TextCNNv2, self).__init__(name=name)
        self.dropout_keep = dropout_keep
        self.convolve = Convolutions(num_filters)
        self.dense = layers.Dense(num_classes,kernel_initializer='glorot_uniform',name="logitsLayer",activation="softmax")
        self.embeddings = embedding_matrix

    def call(self, inputs):
        word_embeds = tf.gather(self.embeddings, inputs) #non-trainable
        x = self.convolve(word_embeds)
        x = tf.nn.dropout(x, self.dropout_keep)
        logits = self.dense(x)
        return logits

    def pred(self, data):
        forwardsPass = self(data)
        return np.argmax(forwardsPass,axis=1)




if __name__ == "__main__":
    # params
    batch_size = 64
    lr = 0.0001
    epochs = 15
    train_samples = 1000
    test_samples = 1000
    vocab_size = 750
    max_words = 500
    num_classes = 3
    embedding_size = 100
    num_filters = 100

    #Create Random/Toy Data
    vocab = np.random.rand(vocab_size, embedding_size)
    vocab[0, :] = 0
    X = np.random.randint(1, vocab_size,
                          (train_samples + test_samples, max_words))

    # test train split
    X_train = X[:train_samples]
    X_test = X[train_samples:]
    y_train = np.random.randint(0, num_classes, train_samples)
    y_test = np.random.randint(0, num_classes, test_samples)
    y_testing = y_test

    y_train = tf.keras.backend.one_hot(y_train, num_classes)
    y_test = tf.keras.backend.one_hot(y_test, num_classes)
    X_train = tf.convert_to_tensor(X_train, dtype=tf.int32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.int32)

    #Define and Compile Model
    model = TextCNNv2(vocab,num_classes)
    optimizer = tf.keras.optimizers.Adam(lr, 0.9, 0.99)
    model.compile(optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    earlyStopping = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=3)

    model.fit(X_train, y_train, epochs=2, batch_size=64, validation_data=(X_test, y_test),callbacks=[earlyStopping])
    print("Evalaute: ----")
    model.evaluate(X_test, y_test, verbose=1)
    model.summary()

    ######### Save/Load Models ##########
    # #Save the model:
    # path = "my_mtCNN"
    # model.save(path)

    # #Load the model:
    # path = "my_mtCNN"
    # load_model = keras.models.load_model(path)
    # load_model.evaluate(X_test, [y_test, y_test2], verbose=1)
    #
    # load_model.fit(X_train, [y_train,y_train2], epochs=epochs, batch_size=64, validation_data=(X_val, [y_val,y_val2]),
    #           callbacks=[earlyStopping])








