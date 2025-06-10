import tensorflow as tf
import os


def load_and_preprocess_image(file_path, image_size=(128, 128)):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0,1]
    return image


def get_celeba_dataset(image_dir, image_size=(128, 128), batch_size=32, shuffle=True):
    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(
        image_dir) if fname.endswith('.jpg')]

    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    image_ds = path_ds.map(lambda x: load_and_preprocess_image(
        x, image_size), num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        image_ds = image_ds.shuffle(buffer_size=1000)

    image_ds = image_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return image_ds
