import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras.datasets import fashion_mnist
import numpy as np
import sys

from typing import Tuple, Union

from utils import fit_wgan, load_dataset


def main(dataset: str, epochs: Union[int, str], batch_size: Union[int, str] = 128, noise_dim: Union[int, str] = 128) -> None:
    (x_train, y_train), (x_test, y_test), key = load_dataset(dataset)
    y_train = y_train.reshape((-1, 1))
    input_shape = (28, 28, 3) if dataset != 'cifar' else (32, 32, 3)
    fit_wgan(
        train_images=x_train,
        train_labels=y_train,
        epochs=int(epochs),
        batch_size=int(batch_size),
        noise_dim=int(noise_dim),
        input_shape=input_shape,
    )


if __name__ == '__main__':
    main(sys.argv[1:])
