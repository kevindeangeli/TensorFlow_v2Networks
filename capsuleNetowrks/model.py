"""
This is model 9 with the new loss function.

I want to see if the new loss function makes a diference.

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from capsulelayers import CapsuleLayer, PrimaryCap, Length
from tensorflow.keras import callbacks
from tensorflow import keras
from sklearn.metrics import f1_score
import pandas as pd
import time
import pickle


def CapsNet(input_shape, args):
    """
    A Capsule Network for text classification
    :param input_shape: Number of words per document
    :param n_class: number of classes
    :param routings: number of routing iterations
    :param batch_size: size of batch
    :return: keras model
    """
    x = layers.Input(shape=input_shape, batch_size=args.batch_size)

    #embds = layers.Embedding(vocab_size, embedding_size)(x)
    embds = layers.Embedding(args.vocab_size, args.embedding_size,embeddings_initializer=tf.keras.initializers.Constant(args.vocab),trainable=False)(x)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = tf.keras.layers.Conv1D(filters=256, kernel_size=5, strides=1, padding='valid', activation='relu', name='conv1a',
                                            kernel_initializer='glorot_uniform')(embds)

    conv1 = tf.keras.layers.Dropout(rate=0.30)(conv1)

    conv2 = tf.keras.layers.Conv1D(filters=256, kernel_size=5, strides=1, padding='valid', activation='relu', name='conv1b',
                                            kernel_initializer='glorot_uniform')(conv1)

    conv2 = tf.keras.layers.Dropout(rate=0.30)(conv2)

    conv3 = tf.keras.layers.Conv1D(filters=256, kernel_size=5, strides=1, padding='valid', activation='relu', name='conv1c',
                                            kernel_initializer='glorot_uniform')(conv2)




    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv3, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')


    # Layer 3: Capsule layer. Routing algorithm works here.
    numCaps = 8
    digitcaps = CapsuleLayer(num_capsule=numCaps, dim_capsule=8, routings=args.routings, name='digitcaps')(primarycaps)

    numFeatures = 200
    flatt_digits = layers.Flatten()(digitcaps)

    dense1 = layers.Dense(units=numCaps*numFeatures,activation='relu')(flatt_digits)
    dense1 = tf.keras.layers.Dropout(rate=0.5)(dense1)
    dense2 = layers.Dense(units=args.num_classes,activation='softmax')(dense1)


    # Layer 4: Compute the norm of the capsules
    #out_caps = Length(name='capsnet')(digitcaps)

    # Models for training and evaluation (prediction)
    train_model = models.Model(x, dense2)

    return train_model

class restoreTrainingCB(keras.callbacks.Callback):
    def __init__(self, save_dir):
        self.save_dir = save_dir
    def on_train_batch_begin(self, batch, logs=None):
        #keys = list(logs.keys())
        filename = self.save_dir
        file = open(filename, 'wb')
        pickle.dump(batch, file)
        file.close()

# @tf.function
# def margin_loss(y_true, y_pred):
#     """
#     Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
#     :param y_true: [None, n_classes]
#     :param y_pred: [None, num_capsule]
#     :return: a scalar loss value.
#     """
#     # return tf.reduce_mean(tf.square(y_pred))
#     L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
#         0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))
#
#     return tf.reduce_mean(tf.reduce_sum(L, 1))


def trainCapsNet_MultiConv(data, args, continueTraining=False):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """

    # unpacking the data
    (x_train, y_train), (x_validation, y_validation) = data


    # callbacks
    #log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/trained_model.h5', monitor='val_accuracy',
                                           save_best_only=False, save_weights_only=True,save_freq=10)
    restoreTr = restoreTrainingCB(args.githubFolder + "restoreBatch")
    #Declare model
    #mirrored_strategy = tf.distribute.MirroredStrategy()

    #with mirrored_strategy.scope():
    model = CapsNet(input_shape=x_train.shape[1:],args=args)


    trainingData = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    valData = tf.data.Dataset.from_tensor_slices((x_validation, y_validation))
    valData = valData.shuffle(buffer_size=x_validation.shape[0], reshuffle_each_iteration=False,
                                         seed=5).batch(args.batch_size)

    if continueTraining is True:
        model = loadModel(model, args)
        #load seed used.
        file = open(args.githubFolder + "randomSeed", 'rb')
        randomSeed = pickle.load(file)
        file.close()
        train_dataset = trainingData.shuffle(buffer_size=x_train.shape[0], reshuffle_each_iteration=False, seed=randomSeed).batch(args.batch_size)

        file = open(args.githubFolder + "restoreBatch", 'rb')
        datasetRestoreIdx = pickle.load(file)
        file.close()
        train_dataset = train_dataset.skip(datasetRestoreIdx)


        patience_count = get_patient(args.githubFolder + args.model_name+ "_validation" +".csv")

        #load patient count.
        #load batch number to skip those.

    else:
        # compile the model and train from scratch
        model.compile(optimizer=optimizers.Adam(args.lr,0.9,0.99,1e-8),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=["accuracy"],
                      run_eagerly=False)
        patience_count = 0
        randomSeed = np.random.randint(1000)
        train_dataset = trainingData.shuffle(buffer_size=x_train.shape[0], reshuffle_each_iteration=False, seed=randomSeed).batch(args.batch_size)
        #store the random seed used for shuffle:
        file = open(args.githubFolder + "randomSeed", 'wb')
        pickle.dump(randomSeed, file)
        file.close()

    model.summary()

    print("\nCurrent Patience: ", patience_count)

    for i in range(args.epochs):
        #if patient_count < 5 break. return trained model to test it.
        if patience_count >= args.patience:
            print("\n\n Patience ", args.patience , " reached \n\n")
            break
        start_time = time.time()
        model.fit(train_dataset,
                  validation_data=valData,
                  epochs=1,
                  callbacks=[checkpoint,restoreTr],
                  verbose=1)
        #model.save_weights(args.save_dir + '/trained_model.h5')
        #print('\n\nTrained model saved to \'%s/trained_model.h5\'' % args.save_dir, "\n")
        saveData(model=model, X_val=x_validation, y_val=y_validation, continueTraining=continueTraining,
                 start_time=start_time, args=args)
        continueTraining = True

        #Before a new epoch starts:
        randomSeed = np.random.randint(1000)
        train_dataset = trainingData.shuffle(buffer_size=x_train.shape[0], reshuffle_each_iteration=False, seed=randomSeed).batch(args.batch_size)
        #store the random seed used for shuffle:
        file = open(args.githubFolder + "randomSeed", 'wb')
        pickle.dump(randomSeed, file)
        file.close()
        #get new patient
        patience_count = get_patient(args.githubFolder + args.model_name+ "_validation" +".csv")
        print("\nCurrent Patience: ", patience_count)


    return model

def testModel(args, x_test, y_test):
    print("Tesst accuracy: ")
    model = CapsNet(input_shape=args.num_words,args=args)
    model = loadModel(model, args)
    y_pred = model.predict(x_test)
    y_p = [np.argmax(i) for i in y_pred]
    y_p = np.array(y_p)
    #y_p = keras.utils.to_categorical(y_p, args.num_classes)

    micro = f1_score(y_test, y_p, average='micro')
    macro = f1_score(y_test, y_p, average='macro')
    print("\n\n Micro Score: ", micro)
    print("Macro Score: ", macro, "\n\n")



def loadModel(model,args, w_file='trained_model.h5'):
    #saveW = '/gpfs/alpine/proj-shared/med107/kevindeangeli/active_learning_textCNN/learnedParameters/capsNetW/'
    saveW = args.save_dir
    weightF = w_file
    print("Loading Model")
    model.load_weights(saveW+"/"+weightF)

    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"],
                  run_eagerly=False)

    return model

def saveData(model,X_val, y_val,continueTraining, start_time,args):
    #model.summary(print_fn=myprint)
    with open(args.githubFolder+args.model_name +'.txt', 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))


    y_pred = model.predict(X_val)
    y_p = [np.argmax(i) for i in y_pred]
    y_p = np.array(y_p)
    #y_p = keras.utils.to_categorical(y_p, args.num_classes)

    micro = f1_score(y_val, y_p, average='micro')
    macro = f1_score(y_val, y_p, average='macro')
    print("\n\n Micro Score: ", micro)
    print("Macro Score: ", macro, "\n\n")

    totalTime = np.array([int(time.time() - start_time)])

    data_path = args.githubFolder + args.model_name+ "_validation" +".csv"
    if continueTraining == False:
        X = np.vstack((micro.T, macro.T, totalTime.T))
        df0 = pd.DataFrame(X.T, columns=['Micro', 'Macro', 'Time'])
        df0.to_csv(data_path)
    else:
        df0 = pd.read_csv(data_path)
        micros = df0['Micro'].to_numpy()
        macros = df0['Macro'].to_numpy()
        times = df0['Time'].to_numpy()
        micros = np.append(micros,[micro])
        macros = np.append(macros,[macro])
        times = np.append(times,[totalTime])
        X = np.vstack((micros.T, macros.T, times.T))
        df0 = pd.DataFrame(X.T, columns=['Micro', 'Macro', 'Time'])
        df0.to_csv(data_path)


def get_patient(data_path):
    df0 = pd.read_csv(data_path)
    micros = df0['Micro'].to_numpy()
    last = micros[-1]
    patient_count = 0
    for i in reversed(micros[0:-1]):
        if i >= last:
            patient_count +=1
            last = i
        else:
            return patient_count
    return patient_count