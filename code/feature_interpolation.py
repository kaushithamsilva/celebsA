import tensorflow as tf
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# Import your custom modules
from vae import VAE, Sampling  # Assuming VAE and Sampling classes are in vae.py
from preprocess import get_celeba_datasets_with_splits
import model_utils  # Your model_utils.py with save_model and load_model
# from hyperplane import Hyperplane  # Not strictly needed for this script's logic

# --- Configuration Paths ---
SAVE_PATH = '../../models/ci_vae/'
CHECKPOINT_PATH = os.path.join(SAVE_PATH, 'checkpoints/')
EPOCH = 500  # The epoch number for the saved VAE model

VAE_MODEL_PATH = CHECKPOINT_PATH
VAE_MODEL_NAME = f'vae-e{EPOCH}'  # Load the VAE from the specified epoch

IMAGE_DIR = '../data/img_align_celeba/img_align_celeba/'
ATTRIBUTES_CSV = '../data/list_attr_celeba.csv'
EVAL_PARTITION_CSV = '../data/list_eval_partition.csv'

# Base output directory for all attribute interpolations
BASE_OUTPUT_DIR = '../figures/attribute_interpolation/'


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

    # Ensure desired_values are integers for comparison
    desired_values = [int(v) for v in desired_values]

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
        # Provide more context on what was being searched for in the error message
        attr_search_str = ", ".join([f"{attr_names_list[a_idx]}={d_val}" for a_idx, d_val in zip(
            attribute_indices, desired_values)])
        raise ValueError(
            f"Could not find {num_samples} images with attributes: {attr_search_str}.")
    return selected_images[:num_samples], selected_attributes[:num_samples]


def plot_images(images, title="", filename="", output_dir=""):
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

    if filename and output_dir:
        full_path = os.path.join(output_dir, filename)
        plt.savefig(full_path)
        print(f"Saved plot to {full_path}")
    plt.show()

# --- Main Interpolation Function ---


def perform_interpolation(
    vae_model,
    train_ds,
    celeba_attribute_names,
    attr_1_name,
    attr_2_name,
    epoch,
    base_output_dir=BASE_OUTPUT_DIR
):
    print(
        f"\n--- Starting Interpolation: {attr_1_name} (No {attr_2_name}) to {attr_1_name} (With {attr_2_name}) ---")

    # Get attribute indices
    try:
        attr_1_idx = get_attribute_index(attr_1_name, celeba_attribute_names)
        attr_2_idx = get_attribute_index(attr_2_name, celeba_attribute_names)
        print(f"Found '{attr_1_name}' attribute at index: {attr_1_idx}")
        print(f"Found '{attr_2_name}' attribute at index: {attr_2_idx}")
    except ValueError as e:
        print(f"Error getting attribute indices: {e}")
        return

    # Create a specific output directory for this interpolation pair
    # Clean attribute names for folder creation (replace spaces and other problematic chars)
    clean_attr_1_name = attr_1_name.replace(' ', '_').replace('/', '_')
    clean_attr_2_name = attr_2_name.replace(' ', '_').replace('/', '_')
    current_output_dir = os.path.join(
        base_output_dir, f"{clean_attr_1_name}_to_{clean_attr_2_name}")
    os.makedirs(current_output_dir, exist_ok=True)
    print(f"Output will be saved to: {current_output_dir}")

    # 1. Select the two images with specific attribute combinations
    print(
        f"\nSelecting one image with '{attr_1_name}' and 'No {attr_2_name}'...")
    try:
        # Desired: attr_1=1, attr_2=0
        img1_samples, _ = select_images_with_specific_attributes(
            train_ds,
            attribute_indices=[attr_1_idx, attr_2_idx],
            desired_values=[1, 0],
            num_samples=1
        )
        image1 = tf.expand_dims(img1_samples[0], axis=0)  # Add batch dim
        plot_images([image1[0]], title=f"Original: {attr_1_name}, No {attr_2_name}",
                    filename=f"original_{clean_attr_1_name}_no_{clean_attr_2_name}.png",
                    output_dir=current_output_dir)
    except ValueError as e:
        print(f"Error selecting {attr_1_name} & No {attr_2_name} image: {e}")
        print("Skipping this interpolation.")
        return

    print(
        f"\nSelecting one image with '{attr_1_name}' and 'With {attr_2_name}'...")
    try:
        # Desired: attr_1=1, attr_2=1
        img2_samples, _ = select_images_with_specific_attributes(
            train_ds,
            attribute_indices=[attr_1_idx, attr_2_idx],
            desired_values=[1, 1],
            num_samples=1
        )
        image2 = tf.expand_dims(img2_samples[0], axis=0)  # Add batch dim
        plot_images([image2[0]], title=f"Original: {attr_1_name}, With {attr_2_name}",
                    filename=f"original_{clean_attr_1_name}_with_{clean_attr_2_name}.png",
                    output_dir=current_output_dir)
    except ValueError as e:
        print(f"Error selecting {attr_1_name} & With {attr_2_name} image: {e}")
        print("Skipping this interpolation.")
        return

    # Combine original images for a single plot
    plot_images([image1[0], image2[0]],
                title=f"Original Samples for Interpolation ({attr_1_name} -> {attr_2_name})",
                filename=f"original_{clean_attr_1_name}_{clean_attr_2_name}_interpolation_samples.png",
                output_dir=current_output_dir)

    # 2. Encode the selected images
    print("\nEncoding images into latent space...")
    z_mean1, _, _ = vae_model.encode(image1)
    z_mean2, _, _ = vae_model.encode(image2)
    print(
        f"Latent vector 1 ({attr_1_name}, No {attr_2_name}) shape: {z_mean1.shape}")
    print(
        f"Latent vector 2 ({attr_1_name}, With {attr_2_name}) shape: {z_mean2.shape}")

    # 3. Interpolate in latent space
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

    # 4. Decode and visualize
    print("Decoding interpolated latent vectors...")
    decoded_images = vae_model.decode(interpolated_latent_vectors_batch)

    plot_images(decoded_images,
                title=f"Interpolation: {attr_1_name} (No {attr_2_name} -> With {attr_2_name}) (Epoch {epoch})",
                filename=f"{clean_attr_1_name}_{clean_attr_2_name}_interpolation_sequence.png",
                output_dir=current_output_dir)

    print(
        f"--- Interpolation for {attr_1_name} (No {attr_2_name}) to {attr_1_name} (With {attr_2_name}) complete ---")


if __name__ == "__main__":
    print("--- Initializing Latent Space Interpolation Script ---")

    # 1. Load VAE model
    print(f"Loading VAE model from {VAE_MODEL_PATH}...")
    vae_model = model_utils.load_model(VAE_MODEL_PATH, VAE_MODEL_NAME)
    if vae_model is None:
        print("Failed to load VAE model. Exiting.")
        exit()
    print("VAE model loaded successfully.")

    # 2. Load CelebA dataset (only need train_ds for selecting images)
    print("Loading CelebA training dataset for image selection...")
    attr_df_for_names = pd.read_csv(ATTRIBUTES_CSV, index_col=0)
    celeba_attribute_names = attr_df_for_names.columns.tolist()

    _, _, test_ds = get_celeba_datasets_with_splits(
        image_dir=IMAGE_DIR,
        attr_csv_path=ATTRIBUTES_CSV,
        eval_csv_path=EVAL_PARTITION_CSV,
        image_size=INPUT_IMAGE_SIZE,
        batch_size=32  # Small batch size for iterating to find samples
    )
    print("Dataset loaded.")

    # --- Example Interpolations ---

    # Example 1: Mustache (No Hat -> With Hat) - Your previous case
    perform_interpolation(vae_model, test_ds, celeba_attribute_names,
                          attr_1_name='Mustache', attr_2_name='Wearing_Hat', epoch=EPOCH)

    # Example 2: Male (No Smiling -> With Smiling)
    perform_interpolation(vae_model, test_ds, celeba_attribute_names,
                          attr_1_name='Male', attr_2_name='Smiling', epoch=EPOCH)

    # Example 3: Young (No Eyeglasses -> With Eyeglasses)
    perform_interpolation(vae_model, test_ds, celeba_attribute_names,
                          attr_1_name='Young', attr_2_name='Eyeglasses', epoch=EPOCH)

    print("\n--- All requested interpolations complete ---")
