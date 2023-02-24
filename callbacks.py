import tensorflow as tf
import keras
from keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

from typing import Any


class GANMonitor(keras.callbacks.Callback):

    def __init__(self, num_imgs: int = 6, latent_dim: int = 128) -> None:
        self.num_imgs = num_imgs
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch: int, logs: Any = None):
        latent_vectors = tf.random.normal(
            shape=(self.num_imgs, self.latent_dim))
        fake_labels = tf.random.uniform(
            (self.num_imgs, 1), minval=0, maxval=10, dtype=tf.int32)
        generated_images = self.model.generator([latent_vectors, fake_labels])
        generated_images = (generated_images*127.5)+127.5
        generated_images = tf.cast(generated_images, dtype=tf.uint8)

        for i in range(self.num_imgs):
            img = generated_images[i].numpy()
            img = img_to_array(img)
            # images will be saved as npy since converting to PIL.Image is causing problems
            with open(f'imgs/generated_img_{i}_epoch{epoch}.npy', 'wb') as f:
                np.save(f, img)
