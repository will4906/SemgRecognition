# -*- coding: utf-8 -*-
"""
Created on 2018/3/9

@author: will4906
"""
import numpy as np
import logging
import os

import keras
import scipy.io as sio

logging.getLogger().setLevel(logging.DEBUG)


def get_frame(subject_path: str, gesture: int, repeat_time: int, index: int):
    """
    获取单帧数据
    :param subject_path:
    :param gesture:
    :param repeat_time:
    :param index:
    :return:
    """
    assert 0 < gesture < 9
    assert 0 < repeat_time < 11
    assert 0 <= index < 1000
    gesture = '00%s' % (gesture,)
    subject = subject_path.split(os.sep)[-1]
    repeat_time = '00%s' % (repeat_time,) if repeat_time < 10 else '0%s' % (repeat_time,)
    mat_name = '%s.mat' % '-'.join([subject, gesture, repeat_time])
    mat = sio.loadmat(os.path.join(subject_path, mat_name))
    data = mat.get('data')
    # logging.info(data[index].reshape(1, 16, 8).shape)
    return data[index].reshape(16, 8), keras.utils.to_categorical(int(gesture) - 1, 8)


def get_trail(subject_path: str, gesture: int, repeat_time: int):
    """
    获取一序列数据
    :param subject_path:
    :param gesture:
    :param repeat_time:
    :return:
    """
    assert 0 < gesture < 9
    assert 0 < repeat_time < 11
    gesture = '00%s' % (gesture,)
    subject = subject_path.split(os.sep)[-1]
    repeat_time = '00%s' % (repeat_time,) if repeat_time < 10 else '0%s' % (repeat_time,)
    mat_name = '%s.mat' % '-'.join([subject, gesture, repeat_time])
    mat = sio.loadmat(os.path.join(subject_path, mat_name))
    data = mat.get('data')
    data = data.reshape((data.shape[0], 16, 8))
    gesture_list = np.zeros(data.shape[0]) + int(gesture) - 1
    return data, keras.utils.to_categorical(gesture_list, 8)


class CapgSubject:
    """
    CapgMyo 对象数据
    """

    def __init__(self, base_path: str, subject_id: int):
        self.base_path = base_path
        self.subject_id = subject_id
        if subject_id > 9:
            self.subject_path = base_path + os.sep + '0' + str(subject_id)
        else:
            self.subject_path = base_path + os.sep + '00' + str(subject_id)

    def yield_train_data(self):
        """
        无效
        :return:
        """
        for gesture in range(1, 9):
            for repeat in filter(lambda x: x % 2 == 1, [x for x in range(1, 11)]):
                for i in range(1, 1001):
                    train_x, train_y = get_frame(self.subject_path, gesture, repeat, i)
                    logging.info(train_x.shape)
                    logging.info(train_y)
                    yield train_x, train_y

    def get_train_data(self):
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        mat_list = os.listdir(self.subject_path)
        for mat in mat_list:
            mat_split = mat.split('.')
            if mat_split[-1] == 'mat':
                if int(mat_split[0].split('-')[-1]) % 2 == 1:
                    mat_file = sio.loadmat(self.subject_path + os.sep + mat)
                    for frame in mat_file.get('data'):
                        train_x.append(frame)
                        train_y.append(
                            mat_file.get('gesture')[0][0])
                else:
                    mat_file = sio.loadmat(self.subject_path + os.sep + mat)
                    for frame in mat_file.get('data'):
                        test_x.append(frame)
                        test_y.append(mat_file.get('gesture')[0][0])
        train_x = np.asarray(train_x)
        test_x = np.asarray(test_x)
        train_y = np.asarray(train_y)
        test_y = np.asarray(test_y)
        return train_x.reshape(train_x.shape[0], 16, 8), keras.utils.to_categorical(train_y - 1, 8), test_x.reshape(test_x.shape[0], 16, 8), keras.utils.to_categorical(test_y - 1, 8)

    def yield_test_data(self):
        """
        无效
        :return:
        """
        for gesture in range(1, 9):
            for repeat in filter(lambda x: x % 2 == 0, [x for x in range(1, 11)]):
                for i in range(1, 1001):
                    yield get_frame(self.subject_path, gesture, repeat, i)

    def yield_train_trail(self):
        """
        无效
        :return:
        """
        while True:
            for gesture in range(1, 9):
                for repeat in filter(lambda x: x % 2 == 1, [x for x in range(1, 11)]):
                    # print(self.subject_path, gesture, repeat)
                    train_x, train_y = get_trail(self.subject_path, gesture, repeat)
                    yield train_x, train_y

    def yield_test_trail(self):
        """
        无效
        :return:
        """
        while True:
            for gesture in range(1, 9):
                for repeat in filter(lambda x: x % 2 == 0, [x for x in range(1, 11)]):
                    yield get_trail(self.subject_path, gesture, repeat)

    def get_test_trail(self):
        test_x = []
        test_y = []
        for gesture in range(1, 9):
            for repeat in filter(lambda x: x % 2 == 0, [x for x in range(1, 11)]):
                for index in range(1000):
                    x, y = get_frame(self.subject_path, gesture, repeat, index)
                    test_x.append(x)
                    test_y.append(y)
        test_x, test_y = np.asarray(test_x), np.asarray(test_y)
        logging.info(test_x.shape)
        return test_x, test_y
