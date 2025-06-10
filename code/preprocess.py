import tensorflow as tf
import os
import pandas as pd
import io


def load_and_preprocess_image(file_path, image_size=(128, 128)):
    """
    Loads an image from a file path, decodes it, resizes it, and normalizes pixel values.

    Args:
        file_path (tf.Tensor): The path to the image file.
        image_size (tuple): Desired (height, width) for the image.

    Returns:
        tf.Tensor: The preprocessed image tensor.
    """
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0,1]
    return image


def get_celeba_dataset_with_attributes(image_dir, attr_csv_path, image_size=(128, 128), batch_size=32, shuffle=True):
    """
    Creates a TensorFlow Dataset for CelebA, returning images and their corresponding attributes.

    Args:
        image_dir (str): Directory containing the CelebA image files.
        attr_csv_path (str): Path to the list_attr_celeba.csv file.
        image_size (tuple): Desired (height, width) for the image.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        tf.data.Dataset: A dataset yielding (image, attributes) pairs.
    """
    # 1. Load attribute data
    # The first row contains the attribute names, so header=0 (default) is correct.
    # We replace -1 with 0 for binary attributes.
    attr_df = pd.read_csv(attr_csv_path, index_col=0)
    attr_df = attr_df.replace(-1, 0)

    # Convert DataFrame to a dictionary for faster lookup
    # Keys will be image filenames (e.g., '000001.jpg'), values will be attribute arrays
    attribute_dict = {index: row.values.astype(
        'float32') for index, row in attr_df.iterrows()}

    # Get all image paths and filter for those present in the attribute dictionary
    all_image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(
        image_dir) if fname.endswith('.jpg')]

    # Filter image paths to ensure they have corresponding attribute entries
    # This handles cases where an image might exist but not be in the CSV, or vice-versa
    valid_image_paths = [
        path for path in all_image_paths if os.path.basename(path) in attribute_dict]

    # Extract corresponding attribute vectors in the same order as valid_image_paths
    valid_attributes = [attribute_dict[os.path.basename(
        path)] for path in valid_image_paths]

    # Create TensorFlow Datasets for paths and attributes
    path_ds = tf.data.Dataset.from_tensor_slices(valid_image_paths)
    attr_ds = tf.data.Dataset.from_tensor_slices(
        tf.constant(valid_attributes, dtype=tf.float32))

    # Pair paths and attributes
    dataset = tf.data.Dataset.zip((path_ds, attr_ds))

    # Map function to load and preprocess images, now also including attributes
    def load_image_and_attributes(file_path_tensor, attributes_tensor):
        image = load_and_preprocess_image(file_path_tensor, image_size)
        return image, attributes_tensor

    image_attr_ds = dataset.map(
        load_image_and_attributes, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        # Shuffle the dataset, ensuring image and attributes stay paired
        image_attr_ds = image_attr_ds.shuffle(buffer_size=1000)

    image_attr_ds = image_attr_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return image_attr_ds
