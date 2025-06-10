import tensorflow as tf
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# Import your custom modules
from vae import VAE, Sampling  # Assuming VAE and Sampling classes are in vae.py
from preprocess import get_celeba_datasets_with_splits
import model_utils  # Your model_utils.py with save_model and load_model
from hyperplane import Hyperplane  # Your provided Hyperplane class

# --- Configuration Paths (MUST BE CORRECTED) ---
# Base directory where your models are saved
SAVE_PATH = '../../models/ci_vae/'
CHECKPOINT_PATH = os.path.join(SAVE_PATH, 'checkpoints/')
DISCRIMINATOR_SAVE_PATH = os.path.join(SAVE_PATH, 'discriminators/')
EPOCH = 400


VAE_MODEL_PATH = CHECKPOINT_PATH
VAE_MODEL_NAME = 'vae-e400'  # As saved in the training script

IMAGE_DIR = '../data/img_align_celeba/img_align_celeba/'
ATTRIBUTES_CSV = '../data/list_attr_celeba.csv'
EVAL_PARTITION_CSV = '../data/list_eval_partition.csv'

# Output directory for interpolated images
OUTPUT_DIR = '../figures/interpolated_mustache_samples/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Model Parameters (MUST MATCH TRAINING) ---
INPUT_IMAGE_SIZE = (64, 64)
INPUT_IMAGE_CHANNELS = 3
LATENT_DIM = 128
HIDDEN_DIM = 64  # Base filters used in VAE constructor


def get_attribute_index(attribute_name, attr_names_list):
    """
    Helper to get the index of a specific attribute.
    """
    try:
        return attr_names_list.index(attribute_name)
    except ValueError:
        raise ValueError(
            f"Attribute '{attribute_name}' not found in attribute list.")


def select_images_with_attribute(dataset, attribute_index, num_samples=2):
    """
    Selects a specified number of images that have the given attribute.
    Iterates over batches and collects samples.
    """
    selected_images = []
    selected_attributes = []

    # The dataset already yields batches (image_batch, attr_batch)
    for image_batch, attr_batch in dataset:  # dataset is already batched
        # Ensure attr_batch is indeed 2D, even if batch size is 1 or it's the last incomplete batch

        # FIX: Compare with an integer literal, or cast 1.0 to attr_batch's dtype
        # Assuming attributes are 0 or 1 (integers)
        has_attribute_mask = (
            attr_batch[:, attribute_index] == 1)  # Changed 1.0 to 1

        # Use tf.boolean_mask to get the images and attributes that match within this batch
        images_with_attribute = tf.boolean_mask(
            image_batch, has_attribute_mask)
        attrs_with_attribute = tf.boolean_mask(attr_batch, has_attribute_mask)

        # Iterate over the filtered samples and add to our list
        for i in range(tf.shape(images_with_attribute)[0]):
            selected_images.append(images_with_attribute[i])
            selected_attributes.append(attrs_with_attribute[i])

            # Stop once we have enough samples
            if len(selected_images) >= num_samples:
                return selected_images[:num_samples], selected_attributes[:num_samples]

    # If the loop finishes and we don't have enough samples
    if len(selected_images) < num_samples:
        raise ValueError(
            f"Could not find {num_samples} images with the specified attribute.")
    return selected_images[:num_samples], selected_attributes[:num_samples]


def plot_images(images, title="", filename=""):
    """
    Plots a list of images in a grid and saves them.
    """
    num_images = len(images)
    if num_images == 0:
        print("No images to plot.")
        return

    fig_cols = min(num_images, 5)  # Max 5 columns for better layout
    fig_rows = int(np.ceil(num_images / fig_cols))

    plt.figure(figsize=(2 * fig_cols, 2 * fig_rows))
    for i, img in enumerate(images):
        plt.subplot(fig_rows, fig_cols, i + 1)
        plt.imshow(img.numpy() if tf.is_tensor(img) else img)
        plt.axis('off')
    plt.suptitle(title)
    # Adjust layout to prevent title overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if filename:
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        print(f"Saved plot to {os.path.join(OUTPUT_DIR, filename)}")
    plt.show()


if __name__ == "__main__":
    print("--- Starting Latent Space Interpolation ---")

    # 1. Load VAE model
    print(f"Loading VAE model from {VAE_MODEL_PATH}...")
    vae_model = model_utils.load_model(
        VAE_MODEL_PATH, VAE_MODEL_NAME
    )
    if vae_model is None:
        print("Failed to load VAE model. Exiting.")
        exit()
    print("VAE model loaded successfully.")

    # 2. Load CelebA dataset (only need train_ds for selecting images)
    print("Loading CelebA training dataset for image selection...")
    # Get attribute names from CSV to find 'Mustache' index
    attr_df_for_names = pd.read_csv(ATTRIBUTES_CSV, index_col=0)
    celeba_attribute_names = attr_df_for_names.columns.tolist()

    # We only need the training dataset to pick images
    train_ds, _, _ = get_celeba_datasets_with_splits(
        image_dir=IMAGE_DIR,
        attr_csv_path=ATTRIBUTES_CSV,
        eval_csv_path=EVAL_PARTITION_CSV,
        image_size=INPUT_IMAGE_SIZE,
        batch_size=32  # Small batch size for iterating to find samples
    )
    print("Dataset loaded.")

    # Get the index for 'Mustache' attribute
    try:
        mustache_attr_idx = get_attribute_index(
            'Mustache', celeba_attribute_names)
        print(f"Found 'Mustache' attribute at index: {mustache_attr_idx}")
    except ValueError as e:
        print(
            f"Error: {e}. Please check the exact attribute name in your list_attr_celeba.csv.")
        print("Available attributes:", celeba_attribute_names)
        exit()

    # 3. Select two images with 'Mustache'
    print(f"Selecting two images with 'Mustache' attribute...")
    try:
        # Pass the already batched train_ds directly
        mustache_images, _ = select_images_with_attribute(
            train_ds,  # No .unbatch() here!
            mustache_attr_idx,
            num_samples=2
        )
        mustache_image1 = tf.expand_dims(
            mustache_images[0], axis=0)  # Add batch dim
        mustache_image2 = tf.expand_dims(
            mustache_images[1], axis=0)  # Add batch dim

        plot_images([mustache_image1[0], mustache_image2[0]],
                    title="Original Mustache Images",
                    filename="original_mustache_images.png")

    except ValueError as e:
        print(f"Error selecting images: {e}")
        print("Consider checking your dataset or num_samples if you have very few matching images.")
        exit()

    # 4. Encode the selected images
    print("Encoding images into latent space...")
    z_mean1, _, _ = vae_model.encode(mustache_image1)
    z_mean2, _, _ = vae_model.encode(mustache_image2)
    print(f"Latent vector 1 shape: {z_mean1.shape}")
    print(f"Latent vector 2 shape: {z_mean2.shape}")

    # 5. Interpolate in latent space
    num_interpolations = 10
    interpolation_factors = np.linspace(
        0, 1, num_interpolations)  # 0 to 1 in 10 steps

    interpolated_latent_vectors = []
    for alpha in interpolation_factors:
        # Linear interpolation: z_interp = (1 - alpha) * z1 + alpha * z2
        interpolated_z = (1 - alpha) * z_mean1 + alpha * z_mean2
        interpolated_latent_vectors.append(interpolated_z)

    # Concatenate to a single tensor for batch decoding
    interpolated_latent_vectors_batch = tf.concat(
        interpolated_latent_vectors, axis=0)
    print(f"Generated {num_interpolations} interpolated latent vectors.")

    # 6. Decode and visualize
    print("Decoding interpolated latent vectors...")
    decoded_images = vae_model.decode(interpolated_latent_vectors_batch)

    plot_images(decoded_images,
                title=f"Latent Space Interpolation (Mustache: {mustache_attr_idx})",
                filename="mustache_interpolation_samples.png")

    print("--- Interpolation complete ---")
