# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 22:24:40 2022

@author: AMIT CHAKRABORTY
"""

from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import uniform, choice
from keras.preprocessing.image import ImageDataGenerator
from dataload import load_dataset
from keras.utils import np_utils 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt


def data():
    nb_classes = 2
    # the data, shuffled and split between train and test sets
    X_train, y_train, X_test, y_test = load_dataset()
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    #Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_train = y_train
    Y_test = y_test
    #Y_test = np_utils.to_categorical(y_test, nb_classes)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    datagen2 = ImageDataGenerator(rescale=1. / 255)

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)
    datagen2.fit(X_test)

    return datagen, datagen2, X_train, Y_train, X_test, Y_test

def model(datagen, datagen2, X_train, Y_train, X_test, Y_test):
    batch_size = 16
    nb_epoch = 20
    img_width, img_height = 224, 224

    # input image dimensions
    input_shape = (img_width, img_height, 3)
    # the CIFAR10 images are RGB
    model = Sequential()
    model.add(Conv2D(32, (2, 2), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, .5)}}))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer={{choice(['adadelta','adam','rmsprop', 'sgd', 'Ftrl', 'Nadam', 'Adamax'])}}, metrics=['accuracy'])

    # fit the model on the batches generated by datagen.flow()
    history = model.fit_generator(datagen.flow(X_train, Y_train,
                        batch_size=batch_size),
                        steps_per_epoch=X_train.shape[0] // batch_size ,
                        epochs=nb_epoch,
                        validation_data= datagen2.flow(X_test, Y_test, batch_size = batch_size),
                        validation_steps=X_test.shape[0] // batch_size)

    score, acc = model.evaluate(X_test, Y_test, verbose=0)
    
    

    return {'loss': -acc, 'status': STATUS_OK, 'model': model, 'history.val_loss':history.history['val_loss'], 
           'history.val_acc': history.history['val_accuracy'], 'history.loss': history.history['loss'], 
           'history.acc': history.history['accuracy'] }

def train():
    datagen, datagen2, X_train, Y_train, X_test, Y_test = data()

    trials = Trials()
    
    best_run, best_model = optim.minimize(model=model,
                                          data=data, 
                                          algo=tpe.suggest,
                                          max_evals=7, 
                                          trials=trials
                                          #,notebook_name='hyperopt_notebook'
                                          )
    
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    
    print("trials.vals",trials.vals)
    
    oplist = ['adadelta','adam','rmsprop', 'sgd', 'Ftrl', 'Nadam', 'Adamax']
    opl = []
    for i in range(7):
        opti = trials.vals['optimizer'][i]
        opti = oplist[opti]
        drp = round(trials.vals['Dropout'][i], 2)
        drp = str(drp)
        opti = str(opti) + '_' + drp
        opl.append(opti)
    print(opl)
    
    oplist = ['adadelta','adam','rmsprop', 'sgd', 'Ftrl', 'Nadam', 'Adamax']
    plt.figure(figsize=(15,8))
    #cmap = plt.get_cmap('jet_r')
    plt.style.use('ggplot')
    with plt.ion():
        for i in range(7):
            opti = trials.vals['optimizer'][i]
            opti = oplist[opti]
            loss = trials.trials[i]['result']['history.acc']
            new_list = set(loss)
            new_list.remove(max(new_list))
            if(max(new_list) > 0.71):
                plt.plot(loss,  linewidth=3, linestyle = 'dashed', marker='8')
            #plt.figure(figsize=(15,8))
            else:
                plt.plot(loss)
            plt.legend(opl)
            plt.savefig('accuracy.png')
    
    oplist = ['adadelta','adam','rmsprop', 'sgd', 'Ftrl', 'Nadam', 'Adamax']
    plt.figure(figsize=(15,8))
    for i in range(7):
        opti = trials.vals['optimizer'][i]
        opti = oplist[opti]
        loss = trials.trials[i]['result']['history.val_acc']
        new_list = set(loss)
        new_list.remove(max(new_list))
        if(max(new_list) > 0.71):
            plt.plot(loss,  linewidth=3, linestyle = 'dashed', marker='8')
        else:
            plt.plot(loss)
        #plt.figure(figsize=(15,8))
        plt.legend(opl)
        plt.savefig('loss.png')
    
    
    
if __name__ == '__main__':
    train()
    
    
    
    

    
    
    
    
    
    
    
    
    
    