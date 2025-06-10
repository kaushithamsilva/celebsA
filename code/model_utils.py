import os
import tensorflow as tf
from vae import VAE, Sampling


def save_model(model, file_path, file_name):
    # Create the full file path by joining the directory path with the file name
    full_file_path = os.path.join(file_path, file_name)

    # Extract the directory from the full file path
    directory = os.path.dirname(full_file_path)

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the model
    model.save(f"{full_file_path}.keras")
    print(f"Model saved to {full_file_path}")


def load_vae_model(path, name):
    full_path = os.path.join(path, name + '.keras')
    # Custom objects are crucial for loading custom Keras Layers/Models
    custom_objects = {"VAE": VAE, "Sampling": Sampling}
    if not os.path.exists(full_path):
        print(f"Error: Model not found at {full_path}")
        return None
    model = tf.keras.models.load_model(
        full_path, custom_objects=custom_objects)
    print(f"Model loaded: {full_path}")
    return model
