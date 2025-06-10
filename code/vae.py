import tensorflow as tf
from tensorflow import keras
import numpy as np


@keras.utils.register_keras_serializable()
class Sampling(keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


@keras.utils.register_keras_serializable()
class VAE(keras.Model):
    def __init__(self, input_dim, latent_dim, hidden_dim, **kwargs):
        # Handle policy before super() call
        if 'policy' in kwargs and isinstance(kwargs['policy'], str):
            kwargs['policy'] = tf.keras.mixed_precision.Policy(
                kwargs['policy'])

        super(VAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        # Rest of the implementation remains the same
        # Encoder
        # Encoder
        self.encoder = keras.Sequential([
            keras.layers.InputLayer(input_shape=input_dim),

            # Convolutional Downsampling (adjust to get to 4x4 spatial for 64x64 input)
            keras.layers.Conv2D(hidden_dim, (3, 3), strides=2,
                                padding='same', activation=None),  # 64x64 -> 32x32
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.2),

            keras.layers.Conv2D(hidden_dim * 2, (3, 3), strides=2,
                                padding='same', activation=None),  # 32x32 -> 16x16
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.2),

            keras.layers.Conv2D(hidden_dim * 4, (3, 3), strides=2,
                                padding='same', activation=None),  # 16x16 -> 8x8
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.2),

            # Add another conv layer to get to 4x4 if input is 64x64
            keras.layers.Conv2D(hidden_dim * 8, (3, 3), strides=2,  # 8x8 -> 4x4
                                padding='same', activation=None),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.2),

            keras.layers.Flatten(),
            # Increased dense size
            keras.layers.Dense(hidden_dim * 8 * 2, activation='relu'),
            keras.layers.Dense(latent_dim * 2)  # Outputs [z_mean, z_log_var]
        ])

        # Sampling Layer
        self.sampling = Sampling()

        # Decoder
        self.decoder = keras.Sequential([
            keras.layers.InputLayer(input_shape=(latent_dim,)),

            # Upsampling (Dense + Reshape)
            # This Dense output size must match the flattened size from the encoder's last conv layer
            # For 4x4 spatial output from encoder's last conv (hidden_dim*8 filters)
            # hidden_dim * 8 for 4x4 spatial
            keras.layers.Dense(4 * 4 * hidden_dim * 8, activation='relu'),
            # Matches the final spatial size and filters before flatten in encoder
            keras.layers.Reshape((4, 4, hidden_dim * 8)),

            # Transposed Convolutions (mirroring the encoder)
            keras.layers.Conv2DTranspose(
                hidden_dim * 4, (3, 3), strides=2, padding='same', activation=None),  # 4x4 -> 8x8
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.2),

            keras.layers.Conv2DTranspose(
                hidden_dim * 2, (3, 3), strides=2, padding='same', activation=None),  # 8x8 -> 16x16
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.2),

            keras.layers.Conv2DTranspose(
                hidden_dim, (3, 3), strides=2, padding='same', activation=None),  # 16x16 -> 32x32
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.2),

            keras.layers.Conv2DTranspose(
                input_dim[2], (3, 3), strides=2, padding='same', activation='sigmoid')  # 32x32 -> 64x64 (output channels should be input_dim[2])
        ])

    def encode(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = tf.split(x, num_or_size_splits=2, axis=1)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z

    def decode(self, z):
        return self.decoder(z)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encode(inputs)
        reconstructed = self.decode(z)
        return reconstructed, z_mean, z_log_var

    def get_config(self):
        config = super(VAE, self).get_config()
        config.update({
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def get_z_mean_embeddings(data, vae_model):
    embeddings = []
    chunk_size = 2000  # Process data in chunks to avoid memory issues
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        transformed_chunk, _, _ = vae_model.encode(chunk)
        embeddings.append(transformed_chunk)

    return np.vstack(embeddings)
