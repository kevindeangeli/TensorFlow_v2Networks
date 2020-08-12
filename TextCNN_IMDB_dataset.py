'''
Created by: Kevin De Angeli
Email: Kevindeangeli@utk.edu
Date: 8/11/20
#IMDB Dataset for binary classification (positive/negative reviews)
'''
from tensorflow.keras.datasets import imdb
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
        self.doc_embed = tf.concat([pool3, pool4, pool5], 1)
        #self.doc_embed = tf.nn.dropout(concat, self.dropout)
        return self.doc_embed


class TextCNNv2(keras.Model):
    def __init__(self,embedding_size,vocab_size, num_classes,
                 num_filters=300, dropout_keep=0.5, name="TextCNNv2"):
        super(TextCNNv2, self).__init__(name=name)
        self.dropout_keep = dropout_keep
        self.embL = layers.Embedding(vocab_size, embedding_size)
        self.embSize = embedding_size
        self.convolve = Convolutions(num_filters)
        self.dense = layers.Dense(num_classes,kernel_initializer='glorot_uniform',name="DenseLayer",activation="softmax")

    def call(self, inputs):
        #word_embeds = tf.gather( self.embeddings, inputs)
        word_embeds = self.embL(inputs)
        x = self.convolve(word_embeds)
        x = tf.nn.dropout(x, self.dropout_keep)
        logits = self.dense(x)
        return logits

    def predictLabel(self,inputs):
        preds = self.predict(inputs)
        return np.argmax(preds, axis=1)

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = imdb.load_data(path="imdb_full.pkl",
                                                          nb_words=None,
                                                          skip_top=0,
                                                          maxlen=None,
                                                          seed=113,
                                                          start_char=1,
                                                          oov_char=2,
                                                          index_from=3)

    X_val = X_test[0:10000]
    X_test = X_test[10000:]
    y_val = y_test[0:10000]
    y_test = y_test[10000:]

    X = np.concatenate((X_test, X_train, X_val))

    # maxes = [np.max(i) for i in X]
    # maxes = np.array(maxes)
    # print(len(maxes))
    # print(np.max(maxes))  #<-- vocab size = 88586+1

    vocab_size =  88586+1
    doc_lenghts = [len(i) for i in X]
    mean_words = np.mean(doc_lenghts)
    max_words = int(mean_words)  # Use the mean number of words as the max (234)

    new_x = []
    for i in X_train:
        len_i = len(i)
        if len_i > max_words:
            new_x.append(i[0:max_words])
        else:
            new_x.append(np.pad(i, (0, max_words - len_i), constant_values=(0, 0)))
    X_train = np.array(new_x)

    new_x = []
    for i in X_test:
        len_i = len(i)
        if len_i > max_words:
            new_x.append(i[0:max_words])
        else:
            new_x.append(np.pad(i, (0, max_words - len_i), constant_values=(0, 0)))
    X_test = np.array(new_x)

    new_x = []
    for i in X_val:
        len_i = len(i)
        if len_i > max_words:
            new_x.append(i[0:max_words])
        else:
            new_x.append(np.pad(i, (0, max_words - len_i), constant_values=(0, 0)))
    X_val = np.array(new_x)

    y_train = keras.utils.to_categorical(y_train, 2)
    y_test = keras.utils.to_categorical(y_test, 2)
    y_val = keras.utils.to_categorical(y_val, 2)

    # params
    batch_size = 128
    lr = 0.0001
    epochs = 99999 #Patience used
    num_classes = 2
    embedding_size = 100
    num_filters = 100


    #Define and Compile Model
    model = TextCNNv2(vocab_size=vocab_size,embedding_size=embedding_size,num_classes=num_classes)
    optimizer = tf.keras.optimizers.Adam(lr, 0.9, 0.99)
    model.compile(optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    earlyStopping = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=5)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),callbacks=[earlyStopping])
    print(" ")
    model.evaluate(X_test, y_test, verbose=1)
    model.summary()

    print("Class Prediction for Dcoument 1: ")
    print(model.predictLabel(X_test[0:1]))



