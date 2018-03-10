# -*- coding: utf-8 -*-
"""
Created on 2018/3/8

@author: will4906
"""

import os
import numpy as np
import scipy.io


def yield_single_train_data(base_path, subject_id):
    """
    单人数据迭代器
    :param base_path:
    :param subject_id:
    :return:
    """
    subject_id = int(subject_id)
    if subject_id > 9:
        subject_path = base_path + os.sep + '0' + str(subject_id)
    else:
        subject_path = base_path + os.sep + '00' + str(subject_id)
    mat_list = os.listdir(subject_path)
    for mat in mat_list:
        mat_split = mat.split('.')


def load_single_train_data(base_path, subject_id):
    '''
    加载单人数据
    '''
    subject_id = int(subject_id)
    if subject_id > 9:
        subject_path = base_path + os.sep + '0' + str(subject_id)
    else:
        subject_path = base_path + os.sep + '00' + str(subject_id)
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    mat_list = os.listdir(subject_path)
    for mat in mat_list:
        mat_split = mat.split('.')
        if mat_split[-1] == 'mat':
            if int(mat_split[0].split('-')[-1]) % 2 == 1:
                mat_file = scipy.io.loadmat(subject_path + os.sep + mat)
                for frame in mat_file.get('data'):
                    train_x.append(frame)
                    train_y.append(
                        mat_file.get('gesture')[0][0])
            else:
                mat_file = scipy.io.loadmat(subject_path + os.sep + mat)
                for frame in mat_file.get('data'):
                    test_x.append(frame)
                    test_y.append(mat_file.get('gesture')[0][0])
    return np.asarray(train_x), np.asarray(train_y), np.asarray(test_x), np.asarray(test_y)


def load_whole_train_data(base_path):
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    mat_list = os.listdir(base_path)
    for mat_name in mat_list:
        mat_split = mat_name.split('.')
        if mat_split[-1] == 'mat':
            if int(mat_split[0].split('-')[1]) < 9:
                if int(mat_split[0].split('-')[-1]) % 2 == 1:
                    mat_file = scipy.io.loadmat(base_path + os.sep + mat_name)
                    for frame in mat_file.get('data'):
                        train_x.append(frame)
                        train_y.append(
                            mat_file.get('gesture')[0][0])
                else:
                    mat_file = scipy.io.loadmat(base_path + os.sep + mat_name)
                    for frame in mat_file.get('data'):
                        test_x.append(frame)
                        test_y.append(mat_file.get('gesture')[0][0])

    return np.asarray(train_x), np.asarray(train_y), np.asarray(test_x), np.asarray(test_y)


def load_17_train_data(base_path):
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    subject_list = os.listdir(base_path)
    for subject_name in subject_list:
        if os.path.isdir(base_path + os.sep + subject_name):
            if int(subject_name) < 18:
                mat_list = os.listdir(base_path + os.sep + subject_name)
                for mat_name in mat_list:
                    mat_split = mat_name.split('.')
                    if mat_split[-1] == 'mat':
                        if int(mat_split[0].split('-')[1]) < 9:
                            mat_file = scipy.io.loadmat(base_path + os.sep + subject_name + os.sep + mat_name)
                            for frame in mat_file.get('data'):
                                train_x.append(frame)
                                train_y.append(
                                    mat_file.get('gesture')[0][0])
            else:
                mat_list = os.listdir(base_path + os.sep + subject_name)
                for mat_name in mat_list:
                    mat_split = mat_name.split('.')
                    if mat_split[-1] == 'mat':
                        if int(mat_split[0].split('-')[1]) < 9:
                            mat_file = scipy.io.loadmat(base_path + os.sep + subject_name + os.sep + mat_name)
                            for frame in mat_file.get('data'):
                                test_x.append(frame)
                                test_y.append(
                                    mat_file.get('gesture')[0][0])
    return np.asarray(train_x), np.asarray(train_y), np.asarray(test_x), np.asarray(test_y)