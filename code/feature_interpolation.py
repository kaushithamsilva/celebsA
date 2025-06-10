import tensorflow as tf
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# Import your custom modules
from vae import VAE, Sampling  # Assuming VAE and Sampling classes are in vae.py
from preprocess import get_celeba_datasets_with_splits
import model_utils  # Your model_utils.py with save_model and load_model
# Your provided Hyperplane class (though not strictly needed for interpolation itself)
from hyperplane import Hyperplane

# --- Configuration Paths (MUST BE CORRECTED) ---
# Base directory where your models are saved
SAVE_PATH = '../../models/ci_vae/'
CHECKPOINT_PATH = os.path.join(SAVE_PATH, 'checkpoints/')
# DISCRIMINATOR_SAVE_PATH = os.path.join(SAVE_PATH, 'discriminators/') # Not directly used in interpolation script
EPOCH = 400

VAE_MODEL_PATH = CHECKPOINT_PATH
VAE_MODEL_NAME = f'vae-e{EPOCH}'  # Load the VAE from the specified epoch

IMAGE_DIR = '../data/img_align_celeba/img_align_celeba/'
ATTRIBUTES_CSV = '../data/list_attr_celeba.csv'
EVAL_PARTITION_CSV = '../data/list_eval_partition.csv'

# Output directory for interpolated images
OUTPUT_DIR = '../figures/interpolated_mustache_hat_samples/'  # New output directory
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
            f"Attribute '{attribute_name}' not found in attribute list. Available: {attr_names_list}")


def select_images_with_specific_attributes(dataset, attribute_indices, desired_values, num_samples=1):
    """
    Selects a specified number of images that match ALL desired attribute values.
    Args:
        dataset (tf.data.Dataset): The dataset to search.
        attribute_indices (list): List of attribute indices to check.
        desired_values (list): List of desired binary values (0 or 1) corresponding to attribute_indices.
                               E.g., [1, 0] for (mustache=1, hat=0)
        num_samples (int): Number of samples to find.
    Returns:
        tuple: (list of selected images, list of selected attribute tensors)
    """
    selected_images = []
    selected_attributes = []

    for image_batch, attr_batch in dataset:
        # Start with a mask that assumes all in batch match
        batch_mask = tf.constant(True, shape=(tf.shape(image_batch)[0],))

        for idx, desired_val in zip(attribute_indices, desired_values):
            # Create a mask for each attribute and combine them
            attr_mask = (attr_batch[:, idx] == desired_val)
            batch_mask = tf.logical_and(batch_mask, attr_mask)

        # Apply the combined mask to get filtered images and attributes
        images_matching = tf.boolean_mask(image_batch, batch_mask)
        attrs_matching = tf.boolean_mask(attr_batch, batch_mask)

        for i in range(tf.shape(images_matching)[0]):
            selected_images.append(images_matching[i])
            selected_attributes.append(attrs_matching[i])

            if len(selected_images) >= num_samples:
                return selected_images[:num_samples], selected_attributes[:num_samples]

    if len(selected_images) < num_samples:
        raise ValueError(
            f"Could not find {num_samples} images with attributes {list(zip(attribute_indices, desired_values))}.")
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
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if filename:
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        print(f"Saved plot to {os.path.join(OUTPUT_DIR, filename)}")
    plt.show()


if __name__ == "__main__":
    print("--- Starting Latent Space Interpolation (Mustache No-Hat to Mustache Hat) ---")

    # 1. Load VAE model
    print(f"Loading VAE model from {VAE_MODEL_PATH}...")
    vae_model = model_utils.load_model(VAE_MODEL_PATH, VAE_MODEL_NAME)
    if vae_model is None:
        print("Failed to load VAE model. Exiting.")
        exit()
    print("VAE model loaded successfully.")

    # 2. Load CelebA dataset
    print("Loading CelebA training dataset for image selection...")
    attr_df_for_names = pd.read_csv(ATTRIBUTES_CSV, index_col=0)
    celeba_attribute_names = attr_df_for_names.columns.tolist()

    train_ds, _, _ = get_celeba_datasets_with_splits(
        image_dir=IMAGE_DIR,
        attr_csv_path=ATTRIBUTES_CSV,
        eval_csv_path=EVAL_PARTITION_CSV,
        image_size=INPUT_IMAGE_SIZE,
        batch_size=32  # Keep it batched for efficient iteration
    )
    print("Dataset loaded.")

    # Get attribute indices
    try:
        mustache_attr_idx = get_attribute_index(
            'Mustache', celeba_attribute_names)
        wearing_hat_attr_idx = get_attribute_index(
            'Wearing_Hat', celeba_attribute_names)
        print(f"Found 'Mustache' attribute at index: {mustache_attr_idx}")
        print(
            f"Found 'Wearing_Hat' attribute at index: {wearing_hat_attr_idx}")
    except ValueError as e:
        print(f"Error: {e}")
        exit()

    # 3. Select two images with specific attribute combinations
    print("\nSelecting one image with 'Mustache' and 'No Hat'...")
    try:
        # Mustache=1, Wearing_Hat=0
        mustache_no_hat_images, _ = select_images_with_specific_attributes(
            train_ds,
            attribute_indices=[mustache_attr_idx, wearing_hat_attr_idx],
            desired_values=[1, 0],
            num_samples=1
        )
        image1_mustache_no_hat = tf.expand_dims(
            mustache_no_hat_images[0], axis=0)  # Add batch dim
        plot_images([image1_mustache_no_hat[0]], title="Original: Mustache, No Hat",
                    filename="original_mustache_no_hat.png")
    except ValueError as e:
        print(f"Error selecting Mustache & No Hat image: {e}")
        exit()

    print("\nSelecting one image with 'Mustache' and 'Wearing Hat'...")
    try:
        # Mustache=1, Wearing_Hat=1
        mustache_with_hat_images, _ = select_images_with_specific_attributes(
            train_ds,
            attribute_indices=[mustache_attr_idx, wearing_hat_attr_idx],
            desired_values=[1, 1],
            num_samples=1
        )
        image2_mustache_with_hat = tf.expand_dims(
            mustache_with_hat_images[0], axis=0)  # Add batch dim
        plot_images([image2_mustache_with_hat[0]], title="Original: Mustache, With Hat",
                    filename="original_mustache_with_hat.png")
    except ValueError as e:
        print(f"Error selecting Mustache & With Hat image: {e}")
        exit()

    # Combine original images for a single plot
    plot_images([image1_mustache_no_hat[0], image2_mustache_with_hat[0]],
                title="Original Samples for Interpolation",
                filename="original_interpolation_samples.png")

    # 4. Encode the selected images
    print("\nEncoding images into latent space...")
    z_mean1, _, _ = vae_model.encode(image1_mustache_no_hat)
    z_mean2, _, _ = vae_model.encode(image2_mustache_with_hat)
    print(f"Latent vector 1 (Mustache, No Hat) shape: {z_mean1.shape}")
    print(f"Latent vector 2 (Mustache, With Hat) shape: {z_mean2.shape}")

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
                title=f"Latent Space Interpolation: Mustache (No Hat -> With Hat) (Epoch {EPOCH})",
                filename="mustache_hat_interpolation_samples.png")

    print("--- Interpolation complete ---")
