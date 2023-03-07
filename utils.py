import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

from discriminator import get_discriminator_model
from generator import get_generator_model
from wgan import ConditionalWGAN
from callbacks import GANMonitor

from typing import Dict, Literal, Optional, Tuple


def discriminator_loss(real_img: tf.Tensor, fake_img: tf.Tensor) -> tf.Tensor:
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


def generator_loss(fake_img: tf.Tensor) -> tf.Tensor:
    return -tf.reduce_mean(fake_img)


def load_dataset(name: Literal['mnist', 'fmnist', 'cifar']) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Dict[int, str]]:
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
            0: "T-shirt/top",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle boot"
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


def fit_wgan(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    epochs: int,
    batch_size: int,
    initial_epoch: Optional[int] = None,
    discriminator: Optional[tf.keras.models.Model] = None,
    generator: Optional[tf.keras.models.Model] = None,
    noise_dim: int = 128,
    input_shape: Tuple[int, int, int] = (28, 28, 3),
) -> None:

    generator_optimizer = Adam(
        learning_rate=0.0002, beta_1=0.5, beta_2=0.9
    )
    discriminator_optimizer = Adam(
        learning_rate=0.0002, beta_1=0.5, beta_2=0.9
    )

    cbk = GANMonitor(num_imgs=3, latent_dim=noise_dim)
    wgan = ConditionalWGAN(
        image_shape=input_shape,
        discriminator=(
            get_discriminator_model(input_shape)
            if initial_epoch is not None else discriminator
        ),
        generator=(
            get_generator_model(noise_dim)
            if initial_epoch is not None else generator
        ),
        latent_dim=noise_dim,
        discriminator_extra_steps=5,
    )
    wgan.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        g_loss_fn=generator_loss,
        d_loss_fn=discriminator_loss,
    )
    wgan.fit(train_images, train_labels, batch_size=batch_size,
             epochs=epochs, initial_epoch=initial_epoch, callbacks=[cbk], verbose=1)

    wgan.generator.save(f'models/g_{epochs}epochs')
    wgan.generator.save_weights(f'models/g_weights_{epochs}epochs')
    wgan.discriminator.save(f'models/d_{epochs}epochs')
    wgan.discriminator.save_weights(f'models/d_weights_{epochs}epochs')
