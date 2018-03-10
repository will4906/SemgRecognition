# -*- coding: utf-8 -*-
"""
Created on 2018/3/8

@author: will4906
"""
import logging
import os

from data import CapgSubject
from model import load_cnn_model

logging.getLogger().setLevel(logging.DEBUG)

base_path = os.path.join('dba', 'data')
subject1 = CapgSubject(base_path, 1)


model, learning_rate_scheduler = load_cnn_model()
train_x, train_y, test_x, test_y = subject1.get_train_data()

model.fit(train_x, train_y, batch_size=1000, validation_data=(test_x, test_y), epochs=28, callbacks=[learning_rate_scheduler])

