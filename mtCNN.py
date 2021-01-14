'''
Created by: Kevin De Angeli
Email: Kevindeangeli@utk.edu
Date: 7/20/20

TF Version: 2.3.0

Multitask CNN:
Given Text Documents, this architecture can handle multiple classification task at once.
The Y's for each task are passed a list ([Y_task1, Y_task2 ...]). (This applies to Train, Test, Val)

When  defining the model, the dense layers are created based on the number of labels in each task.
paramter num_classes is a list: [num_class_task1, num_class_task1, ...]

'''
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import f1_score
import random


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
        return self.doc_embed


class TextCNNv2(keras.Model):
    def __init__(self,embedding_size,vocab_size, num_classes,
                 num_filters=300, dropout_keep=0.5, name="TextCNNv2"):
        super(TextCNNv2, self).__init__(name=name)
        self.dropout_keep = dropout_keep
        self.num_classes = num_classes
        #self.embL = layers.Embedding(vocab_size, embedding_size, embeddings_initializer='GlorotNormal')
        self.embL = layers.Embedding(vocab_size, embedding_size,embeddings_initializer=tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None))
        self.embSize = embedding_size
        self.convolve = Convolutions(num_filters)
        self.classifyLayers = []
        for classes in self.num_classes:
            self.classifyLayers.append(layers.Dense(classes,kernel_initializer='glorot_uniform',name="DenseLayer",
                                                    activation="softmax"))

    def call(self, inputs):
        word_embeds = self.embL(inputs)
        x = self.convolve(word_embeds)
        x = tf.nn.dropout(x, self.dropout_keep)
        logits = []
        for denseLayer in self.classifyLayers:
            logits.append(denseLayer(x))
        return logits

    def predictLabel(self, inputs):
        logits = self.predict(inputs)
        Y_p = [np.argmax(i,axis=1) for i in logits]
        return Y_p


if __name__ == "__main__":
    # params
    batch_size = 64
    lr = 0.0001
    epochs = 2 #Patience is implemented. Change this to a big number for early stopping.
    train_samples = 100
    test_samples = 100
    val_examples = 50
    vocab_size = 750
    max_words = 500
    embedding_size = 100
    num_filters = 100

    #Create Random/Toy Data
    X = np.random.randint(1, vocab_size,
                          (train_samples + test_samples, max_words))
    # test train split
    X_train = X[:train_samples]
    X_test = X[train_samples:]
    X_val = X[:val_examples]

    num_classes = 10 #Task 1 has these number of classes
    y_train = np.random.randint(0, num_classes, train_samples)
    y_test = np.random.randint(0, num_classes, test_samples)
    y_val = np.random.randint(0, num_classes, val_examples)

    num_classes2 = 15 #Task 2 has these number of classes
    y_train2 = np.random.randint(0, num_classes2, train_samples)
    y_test2 = np.random.randint(0, num_classes2, test_samples)
    y_val2 = np.random.randint(0, num_classes2, val_examples)

    #Y_train_all = [y_train,y_train2]

    #Shuffle the Data
    xy = list(zip(X_train, y_train, y_train2))
    random.Random(9).shuffle(xy)  #is the random seed to ensure we use the same (first) batch of random samples
    X_train, y_train,y_train2 = zip(*xy)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    y_train2 = np.array(y_train2)

    #Collect all Ys
    Y_train_all = [y_train,y_train2]



    num_classes = [num_classes,num_classes2] #Pass the number of classes per each task
    model = TextCNNv2(vocab_size=vocab_size,embedding_size=embedding_size,num_classes=num_classes)
    optimizer = tf.keras.optimizers.Adam()
    #model.compile(optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    model.compile(optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"],run_eagerly=True)#False is more efficient
    earlyStopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    filepath = "checkPointsTest"
    mc = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False,
                        save_weights_only=False, mode='auto',period=1) #Checkpoint after each iteration.
    model.fit(X_train, Y_train_all, epochs=epochs, batch_size=64, validation_data=(X_val, [y_val,y_val2]),
              callbacks=[earlyStopping])
    print("Evalaute: ----")
    model.evaluate(X_test, [y_test,y_test2], verbose=1)
    model.summary()

    #See predictions for Document 1:
    print("Prediction for document 1: ")
    print(model.predictLabel(X_test[0:1]))

    y_pred = model.predictLabel(X_test)
    for taskIndex, predictions in enumerate(y_pred):
        print("Task: ", taskIndex)
        micro = f1_score(y_test, predictions, average='micro')
        macro = f1_score(y_test, predictions, average='macro')
        print("Micro: ", micro, "Macro: ", macro, "\n")


    ######### Save/Load Models ##########

    # # #Save the model:
    # path = "my_mtCNN_testing"
    # model.save(path)

    # #Load the model:
    #path = "my_mtCNN_testing"
    #load_model = keras.models.load_model(path)
    # load_model.evaluate(X_test, [y_test, y_test2], verbose=1)
    #
    # load_model.fit(X_train, [y_train,y_train2], epochs=epochs, batch_size=64, validation_data=(X_val, [y_val,y_val2]),
    #           callbacks=[earlyStopping])







