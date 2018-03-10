# -*- coding: utf-8 -*-
"""
Created on 2018/3/10

@author: will4906
"""
import logging
import click
import os

from data import CapgSubject
from model import load_cnn_model

logging.getLogger().setLevel(logging.DEBUG)


def init_save_path(save_path):
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)


@click.command()
@click.option('--subject', help='id of subject', default=1)
@click.option('--save_path', help='the save path of all models', default='save')
def start(subject, save_path):
    init_save_path(save_path)
    base_path = os.path.join('dba', 'data')
    logging.info('loading the subject no.%s' % subject)
    subject1 = CapgSubject(base_path, subject)

    model, learning_rate_scheduler = load_cnn_model()
    train_x, train_y, test_x, test_y = subject1.get_train_data()

    model.fit(train_x, train_y, batch_size=1000, validation_data=(test_x, test_y), epochs=28,
              callbacks=[learning_rate_scheduler])
    model.save(os.path.join(save_path, 'srep%s.h5' % subject))


if __name__ == '__main__':
    start()

