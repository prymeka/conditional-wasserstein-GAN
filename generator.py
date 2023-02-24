import tensorflow as tf
from keras import layers, Model

from typing import Tuple


def upsample_block(
    x: tf.Tensor,
    filters: int,
    activation: layers.Layer,
    kernel_size: Tuple[int, int] = (3, 3),
    strides: Tuple[int, int] = (1, 1),
    up_size: Tuple[int, int] = (2, 2),
    use_bn: bool = True,
    use_bias: bool = False,
    use_dropout: bool = False,
    drop_value: float = 0.3
) -> tf.Tensor:
    x = layers.UpSampling2D(up_size)(x)
    x = layers.Conv2D(filters, kernel_size, strides, 'same', use_bias=use_bias)(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    if activation:
        x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)

    return x


def get_generator_model(latent_dim: int, num_classes: int = 10) -> Model:
    # label input
    label_inp = layers.Input(shape=(1,))
    x = layers.Embedding(num_classes, 50)(label_inp)
    x = layers.Dense(8*8)(x)
    x = layers.Reshape((8, 8, 1))(x)
    # latent vector input
    latent_inp = layers.Input(shape=(latent_dim,))
    y = layers.Dense(4*4*256, use_bias=False)(latent_inp)
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU(0.2)(y)
    y = layers.Reshape((4, 4, 256))(y)
    y = upsample_block(y, 128, layers.LeakyReLU(0.2))
    # y is right now (8, 8, 128)
    merge = layers.Concatenate()([y, x])
    y = upsample_block(merge, 64, layers.LeakyReLU(0.2))
    y = upsample_block(y, 1, layers.Activation('tanh'))
    # y is right now (32, 32, 1), hence we use Cropping2D
    y = layers.Cropping2D((2, 2))(y)

    return Model([latent_inp, label_inp], y, name='generator')


if __name__ == '__main__':
    g_model = get_generator_model(128)
    g_model.summary()
