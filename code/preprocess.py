import tensorflow as tf
import os
import pandas as pd
import io


def load_and_preprocess_image(file_path, image_size=(64, 64)):
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


def get_celeba_datasets_with_splits(image_dir, attr_csv_path, eval_csv_path, image_size=(64, 64), batch_size=32):
    """
    Creates TensorFlow Datasets for CelebA, returning images and their corresponding attributes,
    split into training, validation, and testing sets.

    Args:
        image_dir (str): Directory containing the CelebA image files.
        attr_csv_path (str): Path to the list_attr_celeba.csv file.
        eval_csv_path (str): Path to the list_eval_partition.csv file.
        image_size (tuple): Desired (height, width) for the image.
        batch_size (int): Number of samples per batch for each dataset.

    Returns:
        tuple: A tuple containing (train_dataset, val_dataset, test_dataset),
               each yielding (image, attributes) pairs.
    """
    # 1. Load attribute data
    attr_df = pd.read_csv(attr_csv_path, index_col=0)
    attr_df = attr_df.replace(-1, 0)  # Replace -1 with 0 for binary attributes

    # 2. Load evaluation partition data
    eval_df = pd.read_csv(eval_csv_path, index_col=0, header=None, names=[
                          'image_id', 'partition_type'])

    # 3. Merge attribute and partition dataframes
    # Ensure indices (image filenames) are aligned
    merged_df = attr_df.merge(eval_df, left_index=True, right_index=True)

    # Prepare data for TensorFlow Datasets
    image_filenames = merged_df.index.values  # This is '000001.jpg', etc.
    # All columns except the last one (partition_type)
    attributes = merged_df.iloc[:, :-1].values
    # 0 for train, 1 for val, 2 for test
    partition_types = merged_df['partition_type'].values

    # Explicitly cast partition_types to a specific TensorFlow integer type
    # For example, tf.int32, to ensure consistent comparison.
    # We choose int32 as it's common for such indices.
    partition_types_tf = tf.constant(partition_types, dtype=tf.int32)

    # Create a base TensorFlow Dataset from image filenames, attributes, and partition types
    # This keeps everything aligned.
    full_dataset = tf.data.Dataset.from_tensor_slices(
        (image_filenames, attributes, partition_types_tf))

    # Define filter functions for each split
    # Partition types: 0 = training, 1 = validation, 2 = testing
    def is_train(image_name, attrs, partition_type):
        # Cast the literal 0 to the same type as partition_type
        return tf.equal(partition_type, tf.cast(0, partition_type.dtype))

    def is_val(image_name, attrs, partition_type):
        # Cast the literal 1 to the same type as partition_type
        return tf.equal(partition_type, tf.cast(1, partition_type.dtype))

    def is_test(image_name, attrs, partition_type):
        # Cast the literal 2 to the same type as partition_type
        return tf.equal(partition_type, tf.cast(2, partition_type.dtype))

    # Function to load image and return (image, attributes) only
    def _parse_function(image_name, attributes, partition_type):
        file_path = tf.strings.join(
            [image_dir, image_name])  # Construct full path
        image = load_and_preprocess_image(file_path, image_size)
        return image, attributes

    # Filter and map for each split
    train_dataset = full_dataset.filter(is_train).map(
        _parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = full_dataset.filter(is_val).map(
        _parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = full_dataset.filter(is_test).map(
        _parse_function, num_parallel_calls=tf.data.AUTOTUNE)

    # Apply shuffling, batching, and prefetching
    train_dataset = train_dataset.shuffle(buffer_size=10000).batch(
        batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset
