import tensorflow as tf
from keras import layers, Model

from typing import Tuple


def conv_block(
    x: tf.Tensor,
    filters: int,
    activation: layers.Layer,
    kernel_size: Tuple[int, int] = (5, 5),
    strides: Tuple[int, int] = (2, 2),
    use_dropout: bool = False,
    drop_value: float = 0.3,
) -> tf.Tensor:
    x = layers.Conv2D(filters, kernel_size, strides, 'same')(x)
    x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)

    return x


def get_discriminator_model(input_shape: Tuple[int, int, int], num_classes: int = 10) -> Model:
    # label input
    label_inp = layers.Input(shape=(1,))
    x = layers.Embedding(num_classes, 50)(label_inp)
    x = layers.Dense(input_shape[0]*input_shape[1])(x)
    x = layers.Reshape((input_shape[0], input_shape[1], 1))(x)
    # image input
    image_inp = layers.Input(shape=input_shape)
    merged = layers.Concatenate()([image_inp, x])
    # zero pad the input to increase the input images size to (32, 32, 1)
    y = layers.ZeroPadding2D((2, 2))(merged)
    y = conv_block(y, 64, layers.LeakyReLU(0.2))
    y = conv_block(y, 128, layers.LeakyReLU(0.2), use_dropout=True)
    y = conv_block(y, 256, layers.LeakyReLU(0.2), use_dropout=True)
    y = conv_block(y, 512, layers.LeakyReLU(0.2))
    y = layers.Flatten()(y)
    y = layers.Dropout(0.2)(y)
    y = layers.Dense(1)(y)

    return Model([image_inp, label_inp], y, name='discriminator')


if __name__ == '__main__':
    d_model = get_discriminator_model((28, 28, 1))
    d_model.summary()
