import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from typing import Literal

def discriminator_loss(real_img: tf.Tensor, fake_img: tf.Tensor) -> tf.Tensor:
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss

def generator_loss(fake_img: tf.Tensor) -> tf.Tensor:
    return -tf.reduce_mean(fake_img)

def load_dataset(name: Literal['mnist', 'fmnist', 'cifar']) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], dict[int, str]]:
    (x_train, y_train), (x_test, y_test) = {
        'mnist': tf.keras.datasets.mnist.load_data,
        'fmnist': tf.keras.datasets.fashion_mnist.load_data,
        'cifar': tf.keras.datasets.cifar10.load_data,
    }[name]()
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    if name in ('mnist', 'fmnist'):
        x_train = np.repeat(x_train[..., np.newaxis], 3, -1)
        x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)))
        x_test = np.repeat(x_test[..., np.newaxis], 3, -1)
        x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)))
    
    key = {
        'mnist': {
            0: '0',
            1: '1',
            2: '2',
            3: '3',
            4: '4',
            5: '5',
            6: '6',
            7: '7',
            8: '8',
            9: '9',
        },
        'fmnist': {
            0 : "T-shirt/top",
            1 : "Trouser",
            2 : "Pullover",
            3 : "Dress",
            4 : "Coat",
            5 : "Sandal",
            6 : "Shirt",
            7 : "Sneaker",
            8 : "Bag",
            9 : "Ankle boot"
        },
        'cifar': {
            0: 'airplane', 
            1: 'automobile', 
            2: 'bird', 
            3: 'cat', 
            4: 'deer', 
            5: 'dog', 
            6: 'frog', 
            7: 'horse', 
            8: 'ship', 
            9: 'truck'
        },
    }[name]
    
    return (x_train, y_train), (x_test, y_test), key