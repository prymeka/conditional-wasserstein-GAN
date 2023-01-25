import tensorflow as tf
import keras
from keras.datasets import fashion_mnist
import numpy as np

from typing import Tuple

from discriminator import get_discriminator_model
from generator import get_generator_model
from wgan import ConditionalWGAN
from callbacks import GANMonitor


def run(
    epochs: int,
    input_shape: Tuple[int, int, int],
    noise_dim: int,
    batch_size: int,
    train_images: np.ndarray,
    train_labels: np.ndarray
) -> None:
    def discriminator_loss(real_img: tf.Tensor, fake_img: tf.Tensor) -> tf.Tensor:
        real_loss = tf.reduce_mean(real_img)
        fake_loss = tf.reduce_mean(fake_img)
        return fake_loss - real_loss

    def generator_loss(fake_img: tf.Tensor) -> tf.Tensor:
        return -tf.reduce_mean(fake_img)

    generator_optimizer = keras.optimizer_v2.adam.Adam(
        learning_rate=0.0002, beta_1=0.5, beta_2=0.9
    )
    discriminator_optimizer = keras.optimizer_v2.adam.Adam(
        learning_rate=0.0002, beta_1=0.5, beta_2=0.9
    )

    cbk = GANMonitor(num_imgs=3, latent_dim=noise_dim)
    wgan = ConditionalWGAN(
        image_shape=input_shape,
        discriminator=get_discriminator_model(input_shape),
        generator=get_generator_model(noise_dim),
        latent_dim=noise_dim,
        discriminator_extra_steps=3,
    )
    wgan.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        g_loss_fn=generator_loss,
        d_loss_fn=discriminator_loss,
    )
    wgan.fit(train_images, train_labels, batch_size=batch_size,
             epochs=epochs, callbacks=[cbk], verbose=1)

    wgan.generator.save(f'models/g_{epochs}epochs')
    wgan.generator.save_weights(f'models/g_weights_{epochs}epochs')
    wgan.discriminator.save(f'models/d_{epochs}epochs')
    wgan.discriminator.save_weights(f'models/d_weights_{epochs}epochs')


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    y_train = y_train.reshape((-1, 1))
    run(
        epochs=2,
        input_shape=(28, 28, 1),
        noise_dim=128,
        batch_size=64,
        train_images=x_train,
        train_labels=y_train
    )
