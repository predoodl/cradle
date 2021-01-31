
import os

from keras.datasets import cifar10
from keras.datasets import mnist

from cradle.cradle import cradle


basedir = os.path.abspath(os.path.dirname(__file__)).split('scripts')[0]


def cradle_for_vgg16():
    mo_path = os.path.join(basedir, 'network', 'models', 'vgg16', 'vgg16_cifar10_tensorflow.h5')
    mc_path = os.path.join(basedir, 'network', 'models', 'vgg16', 'vgg16_cifar10_theano.h5')
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    cradle(mo_path, mc_path, x_test, y_test, dis_threshold=16, top_k=5)


def cradle_for_lenet():
    mo_path = os.path.join(basedir, 'network', 'models', 'lenet', 'lenet_mnist_tensorflow.h5')
    mc_path = os.path.join(basedir, 'network', 'models', 'lenet', 'lenet_mnist_theano.h5')
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    cradle(mo_path, mc_path, x_test, y_test, dis_threshold=16, top_k=5)


if __name__ == '__main__':
    cradle_for_vgg16()
