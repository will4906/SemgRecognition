# -*- coding: utf-8 -*-
"""
Created on 2018/3/10

@author: will4906
"""
import os
from keras.models import load_model

from data import CapgSubject

if __name__ == '__main__':
    aver = 0.0
    for i in range(1, 19):
        subject = CapgSubject(os.path.join('dba', 'data'), i)
        model = load_model(os.path.join('save180310a', 'srep%s.h5' % i))
        train_x, train_y, test_x, test_y = subject.get_train_data()
        result = model.evaluate(test_x, test_y, batch_size=1000)
        aver += result[1]
        print('subject%s' % i, 'is', result[1])
    print(aver / 18)
