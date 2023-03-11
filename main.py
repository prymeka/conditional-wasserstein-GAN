import tensorflow as tf
import argparse

from typing import Optional, Tuple

from utils import fit_wgan, load_dataset


def parse_tuple(s: str) -> Tuple[int, ...]:
    try:
        return tuple(map(int, s.strip('()').split(',')))
    except:
        raise argparse.ArgumentTypeError('Invalid tuple value.')


# create an ArgumentParser object
parser = argparse.ArgumentParser(description="""
    Train a conditional Wasserstein GAN model from scratch or continue training using discriminator
    and generator loaded from paths provided. Note that the pre-defined models are designed for 
    28x28 images. 
""")

# add some arguments
parser.add_argument('-d', '--dataset', type=str, required=True,
                    help='Name of Keras dataset: mnist, fmnist, or cifar.')
parser.add_argument('-e', '--epochs', type=int, required=True,
                    help='Number of epochs to trian the model for.')
parser.add_argument('-bs', '--batch-size', type=int,
                    required=False, help='Batch size.')
parser.add_argument('-ie', '--initial-epoch', type=int, default=0,
                    help='Epoch number at which to start training (default: 0).')
parser.add_argument('-dis', '--discriminator-path', type=str, default=None,
                    help='Path to the discriminator model (default: None). It\'s not required when training a model from scratch.')
parser.add_argument('-gen', '--generator-path', type=str, default=None,
                    help='Path to the discriminator model (default: None). It\'s not required when training a model from scratch.')
parser.add_argument('-n', '--noise-dim', type=int, required=False,
                    help='The dimension of the input, noise/latent vector to the generator model.')
parser.add_argument('-is', '--input-shape', type=parse_tuple,
                    required=False, help='The shape of the images.')


def main(
    dataset: str,
    epochs: int,
    batch_size: int = 128,
    initial_epoch: int = None,
    discriminator_path: Optional[str] = None,
    generator_path: Optional[str] = None,
    noise_dim: int = 128,
    input_shape: Tuple[int, int, int] = None
) -> None:
    (x_train, y_train), (x_test, y_test), key = load_dataset(dataset)
    input_shape = input_shape if input_shape is not None else (
        28, 28, 3) if dataset != 'cifar' else (32, 32, 3)
    discriminator = tf.keras.models.load_model(discriminator_path)
    generator = tf.keras.models.load_model(generator_path)
    fit_wgan(
        train_images=x_train,
        train_labels=y_train,
        epochs=int(epochs),
        batch_size=int(batch_size),
        initial_epoch=int(initial_epoch),
        discriminator=discriminator,
        generator=generator,
        noise_dim=int(noise_dim),
        input_shape=input_shape,
    )


if __name__ == '__main__':
    # args = parser.parse_args()
    # main(
    #     dataset=args.dataset,
    #     epochs=args.epochs,
    #     batch_size=args.batch_size,
    #     initial_epoch=args.initial_epoch,
    #     discriminator=args.discriminator,
    #     generator=args.generator,
    #     noise_dim=args.noise_dim,
    #     input_shape=args.input_shape
    # )
    args = {
        'dataset': 'fmnist',
        'epochs': '150',
        'batch_size': 512,
        'initial_epoch': 50,
        'discriminator_path': './models/d_50epochs',
        'generator_path': './models/g_50epochs',
        'noise_dim': 128,
        'input_shape': (28, 28, 3)
    }
    main(**args)
