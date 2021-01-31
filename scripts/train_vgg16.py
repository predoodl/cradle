
import os
import time

from keras.datasets import cifar10
from keras.utils import to_categorical
import keras

from network.vgg16 import vgg16


basedir = os.path.abspath(os.path.dirname(__file__)).split('scripts')[0]


def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(-1, 32, 32, 3)
    x_test = x_test.reshape(-1, 32, 32, 3)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    # y_train = y_train.reshape(-1, 1, 10)
    # y_test = y_test.reshape(-1, 1, 10)
    return (x_train, y_train), (x_test, y_test)


def get_save_path():
    path_name = os.path.join(basedir, 'network', 'models')
    if not os.path.exists(path_name):
        os.mkdir(path_name)
    path_name = os.path.join(path_name, 'vgg16')
    if not os.path.exists(path_name):
        os.mkdir(path_name)
    return path_name


def train_model():
    model = vgg16.get_model_from_manual()
    model_file_name = 'vgg16_cifar10_' + keras.backend.backend() + '.h5'
    model_file_name = os.path.join(get_save_path(), model_file_name)
    model.save(model_file_name)
    (x_train, y_train), (x_test, y_test) = load_data()
    print('backend: ' + keras.backend.backend())
    print('start_time: ' + time.asctime(time.localtime(time.time())))
    model.fit(x_train, y_train, batch_size=32, epochs=100, validation_split=0.1, verbose=1)
    print('end_time: ' + time.asctime(time.localtime(time.time())))
    model.save(model_file_name)
    print(model.evaluate())
    print('\n\n\n')


if __name__ == '__main__':
    train_model()
