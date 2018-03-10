import logging
import keras
import numpy as np

from keras import Sequential
from keras.callbacks import LearningRateScheduler
from keras.layers import Dense, Flatten, BatchNormalization, Activation, \
    LocallyConnected1D, Dropout, Conv1D
from keras.optimizers import SGD, Adam

from util import *


logging.getLogger().setLevel(logging.DEBUG)


def learning_rate_func(index):
    if 16 <= index < 24:
        return 0.1
    elif index >= 24:
        return 0.01
    else:
        return 1.0


learning_rate_scheduler = LearningRateScheduler(learning_rate_func)

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
train_x, train_y, test_x, test_y = load_single_train_data(os.path.join('dba', 'data'), '8')
train_x = train_x.reshape(train_x.shape[0], 16, 8)
test_x = test_x.reshape(test_x.shape[0], 16, 8)
train_y = keras.utils.to_categorical(train_y - 1, 8)
test_y = keras.utils.to_categorical(test_y - 1, 8)

logging.info("数据加载完成")
model.fit(train_x, train_y, batch_size=1000, validation_data=(test_x, test_y), epochs=28, callbacks=[learning_rate_scheduler])
# model.save('guabeishi.h5')