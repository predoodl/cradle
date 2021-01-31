
import time

from keras.datasets import mnist
from keras import backend

from network.lenet.lenet import LeNet


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    return (x_train, y_train), (x_test, y_test)


def train_lenet():
    (x_train, y_train), (x_test, y_test) = load_data()
    print('start time: ' + time.asctime(time.localtime(time.time())))
    lenet = LeNet(model_filename='lenet_mnist_'+backend.backend()+'.h5', epochs=10, input_shape=(28, 28, 1), weight_decay=1e-3)
    lenet.train(x_train / 255.0, y_train)
    print('end time: ' + time.asctime(time.localtime(time.time())))


if __name__ == '__main__':
    train_lenet()
