# -*- coding: utf-8 -*-
"""
Created on 2018/3/9

@author: will4906
"""
import keras
from keras import Sequential
from keras.callbacks import LearningRateScheduler
from keras.layers import BatchNormalization, Conv1D, Activation, LocallyConnected1D, Dropout, Flatten, Dense
from keras.optimizers import SGD


def learning_rate_func(index):
    if 16 <= index < 24:
        return 0.1
    elif index >= 24:
        return 0.01
    else:
        return 1.0


def load_cnn_model():
    """
    根据论文实现得cnn模型
    :return:
    """
    model = Sequential()

    # 1
    model.add(BatchNormalization(input_shape=[16, 8], momentum=0.9))
    model.add(Conv1D(filters=64, kernel_size=3, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))

    # 2
    model.add(Conv1D(filters=64, kernel_size=3, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))

    # 3
    model.add(LocallyConnected1D(filters=64, kernel_size=1))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))

    # 4
    model.add(LocallyConnected1D(filters=64, kernel_size=1))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # 5
    model.add(Flatten())
    model.add(Dense(units=512))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # 6
    model.add(Dense(units=512))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # 7
    model.add(Dense(units=128))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))

    # 8
    model.add(Dense(units=8, activation='softmax'))
    sgd = SGD(lr=1, decay=0.0)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=sgd, metrics=['accuracy'])

    return model, LearningRateScheduler(learning_rate_func)
