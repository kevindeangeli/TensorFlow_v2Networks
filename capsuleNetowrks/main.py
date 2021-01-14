'''
Created by: Kevin De Angeli
Email: Kevindeangeli@utk.edu
Date: 12/2/20
'''
import numpy as np
from model import trainCapsNet_MultiConv,testModel
import os
from tensorflow.keras.datasets import imdb

class modelParams():
    def __init__(self):
        self.epochs = 10000 #use patience
        self.batch_size = 128
        self.lr = 0.0001
        self.routings = 3
        self.embedding_size = 100
        self.patience = 5
        self.num_words = None
        self.num_classes = 2
        self.save_dir = None #to save model parameters
        self.vocab_size = None
        self.vocab = None
        self.githubFolder = None #to store data
        self.model_name = None

def cut_documents(X_data, max_words):
    new_x = []
    for i in X_data:
        len_i = len(i)
        if len_i > max_words:
            new_x.append(i[0:max_words])
        else:
            new_x.append(np.pad(i, (0, max_words - len_i), constant_values=(0, 0)))
    return np.array(new_x)



if __name__ == "__main__":
    modelName = "IMDB_capsNet"
    args = modelParams()
    args.model_name = modelName
    args.save_dir = modelName+"/"

    githubFolder = modelName+"/"
    if not os.path.exists(githubFolder):
        os.makedirs(githubFolder)
    args.githubFolder = githubFolder

    #Read Data
    #X_train, X_validation, X_test, y_train, y_validation, y_test, num_classes = readPartialData(taskName,0.25)

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

    X_train=cut_documents(X_train, max_words)
    X_test=cut_documents(X_test,max_words)
    X_val=cut_documents(X_val,max_words)

    #create random vocab
    args.vocab_size=vocab_size
    args.vocab = np.random.random((vocab_size,args.embedding_size)) #Idealy this should be a w2v vocab


    #cut the documents so that the last batch is equal to the batch size. 
    X_train = X_train[0:int(len(X_train)/args.batch_size)*args.batch_size,:]
    X_validation = X_val[0:int(len(X_val)/args.batch_size)*args.batch_size,:]
    X_test = X_test[0:int(len(X_test)/args.batch_size)*args.batch_size,:]
    y_train = y_train[0:int(len(y_train)/args.batch_size)*args.batch_size]
    y_validation = y_val[0:int(len(y_val)/args.batch_size)*args.batch_size]
    y_test = y_test[0:int(len(y_test)/args.batch_size)*args.batch_size]
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    print(X_validation.shape)
    print(y_validation.shape)
    print(" ")
    print(args.vocab.shape)
    print(args.vocab_size)



    # params
    #variable used to recover training from crashed
    #useful for the supercomputer (2hs limit)
    continueTraining = False

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print("\nTraining from Scratch\n")
    else:
        print("\nFile Exists: Continue Training\n")
        continueTraining = False

    print(X_train[0])
    print(X_test[0])
    print(X_val[0])



    trainedModel = trainCapsNet_MultiConv(data=((X_train, y_train), (X_validation, y_validation)), args=args, continueTraining=continueTraining)
    #testModel(args, X_test, y_test)










