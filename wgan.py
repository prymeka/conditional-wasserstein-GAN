import tensorflow as tf

from keras import Model
from keras.optimizers import Optimizer

from typing import Callable, Dict, Tuple


class ConditionalWGAN(Model):

    def __init__(
        self,
        image_shape: Tuple[int, int, int],
        discriminator: Model,
        generator: Model,
        latent_dim: int,
        discriminator_extra_steps: int = 5,
        gp_weight: float = 10.0
    ) -> None:
        super().__init__()
        self.image_shape = image_shape
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(
        self,
        d_optimizer: Optimizer,
        g_optimizer: Optimizer,
        d_loss_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        g_loss_fn: Callable[[tf.Tensor], tf.Tensor]
    ) -> None:
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(
        self,
        batch_size: int,
        real_images: tf.Tensor,
        fake_images: tf.Tensor,
        real_labels: tf.Tensor,
        fake_labels: tf.Tensor
    ) -> tf.Tensor:
        """
        Implementation of the Gradient Penalty to maintain L2 norm of the
        gradient of the critic's output w.r.t. the inputs near to unity.
        """
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        # interpolation between real data and fake data
        fake_images = tf.cast(fake_images, dtype=tf.float32)
        real_images = tf.cast(real_images, dtype=tf.float32)
        interpolation = real_images + alpha*(fake_images-real_images)
        # real labels will be used as the interpolation labels
        # alpha = tf.reshape(alpha, (batch_size, 1))
        # interpolation_labels = real_labels + alpha*(fake_labels-real_labels)
        interpolation_labels = real_labels

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolation)
            # discriminator's output for the interpolated image
            pred = self.discriminator(
                [interpolation, interpolation_labels], training=True)

        # gradient w.r.t. the interpolated image
        gradient = gp_tape.gradient(pred, [interpolation])[0]
        # norm of the gradient
        norm = tf.sqrt(tf.reduce_sum(tf.square(gradient), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm-1.0)**2)

        return gp

    def train_step(self, data: tf.Tensor) -> Dict[str, float]:
        real_images, real_labels = data
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        if len(real_images.shape) == 3:
            real_images = tf.expand_dims(real_images, axis=-1)
        batch_size = tf.shape(real_images)[0]

        # training the discriminator, typically with 5 extra steps compared
        # to the generator
        for _ in range(self.d_steps):
            latent_vector = tf.random.normal((batch_size, self.latent_dim))
            fake_labels = tf.random.uniform(
                (batch_size, 1), minval=0, maxval=10, dtype=tf.int32)
            with tf.GradientTape() as tape:
                # generate fake images
                fake_images = self.generator(
                    [latent_vector, fake_labels], training=True)
                # logits for the fake and real images
                fake_logits = self.discriminator(
                    [fake_images, fake_labels], training=True)
                real_logits = self.discriminator(
                    [real_images, real_labels], training=True)
                # calculate the gradient penalty
                gp = self.gradient_penalty(
                    batch_size, real_images, fake_images, real_labels, fake_labels)
                # calculate the discriminator loss with the gradient penalty
                d_loss = self.d_loss_fn(
                    real_logits, fake_logits)+gp*self.gp_weight
            # get the gradient w.r.t. the discriminator loss
            d_gradient = tape.gradient(
                d_loss, self.discriminator.trainable_variables)
            # update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables))

        # training the generator
        latent_vector = tf.random.normal((batch_size, self.latent_dim))
        fake_labels = tf.random.uniform(
            (batch_size, 1), minval=0, maxval=10, dtype=tf.int32)
        with tf.GradientTape() as tape:
            # generate fake images
            fake_images = self.generator(
                [latent_vector, fake_labels], training=True)
            # get the discriminator logits for fake images
            fake_logits = self.discriminator(
                [fake_images, fake_labels], training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(fake_logits)
        # get the gradient w.r.t. the generator loss
        g_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(g_gradient, self.generator.trainable_variables))

        return {'d_loss': d_loss, 'g_loss': g_loss}
