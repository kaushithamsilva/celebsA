import tensorflow as tf
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# Import your custom modules
from vae import VAE, Sampling  # Assuming VAE and Sampling classes are in vae.py
from preprocess import get_celeba_datasets_with_splits
import model_utils  # Your model_utils.py with save_model and load_model
# Hyperplane is not used in interpolation, so we can comment it out or remove it if desired
# from hyperplane import Hyperplane

# --- Configuration Paths ---
SAVE_PATH = '../../models/ci_vae/'
CHECKPOINT_PATH = os.path.join(SAVE_PATH, 'checkpoints/')
EPOCH = 500  # The epoch number for your saved VAE model

VAE_MODEL_PATH = CHECKPOINT_PATH
VAE_MODEL_NAME = f'vae-e{EPOCH}'

IMAGE_DIR = '../data/img_align_celeba/img_align_celeba/'
ATTRIBUTES_CSV = '../data/list_attr_celeba.csv'
EVAL_PARTITION_CSV = '../data/list_eval_partition.csv'

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

        # Ensure attr_batch has the same dtype as desired_values for comparison
        # Assuming desired_values are typically int/float 0 or 1
        # Cast to int for comparison with 0/1
        attr_batch_cast = tf.cast(attr_batch, tf.int32)

        for idx, desired_val in zip(attribute_indices, desired_values):
            # Create a mask for each attribute and combine them
            attr_mask = (attr_batch_cast[:, idx] == desired_val)
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
        # Ensure OUTPUT_DIR exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        print(f"Saved plot to {os.path.join(OUTPUT_DIR, filename)}")
    plt.show()


def perform_interpolation(
    vae_model,
    train_ds,
    celeba_attribute_names,
    attribute_name1,
    attribute_value1_start,
    attribute_value1_end,
    attribute_name2=None,  # Optional second attribute to keep constant
    attribute_value2=None,  # Desired value for the second attribute
    num_interpolations=10,
    output_dir_suffix=""  # For specific output folders
):
    """
    Performs latent space interpolation between images defined by specified attributes.
    """
    print(
        f"\n--- Starting Latent Space Interpolation: {attribute_name1} {attribute_value1_start} to {attribute_name1} {attribute_value1_end} ---")

    # Get attribute indices
    try:
        attr1_idx = get_attribute_index(
            attribute_name1, celeba_attribute_names)
        print(f"Found '{attribute_name1}' attribute at index: {attr1_idx}")

        attribute_indices_start = [attr1_idx]
        desired_values_start = [attribute_value1_start]

        attribute_indices_end = [attr1_idx]
        desired_values_end = [attribute_value1_end]

        interpolation_title_base = f"{attribute_name1} ({attribute_value1_start} -> {attribute_value1_end})"
        filename_base = f"{attribute_name1.replace(' ', '_')}_{attribute_value1_start}_to_{attribute_value1_end}"

        if attribute_name2:
            attr2_idx = get_attribute_index(
                attribute_name2, celeba_attribute_names)
            print(f"Found '{attribute_name2}' attribute at index: {attr2_idx}")

            # Add the second attribute to the conditions for both start and end images
            attribute_indices_start.append(attr2_idx)
            desired_values_start.append(attribute_value2)
            attribute_indices_end.append(attr2_idx)
            desired_values_end.append(attribute_value2)

            interpolation_title_base += f" (Keeping {attribute_name2}={attribute_value2})"
            filename_base += f"_keeping_{attribute_name2.replace(' ', '_')}_{attribute_value2}"

    except ValueError as e:
        print(f"Error getting attribute indices: {e}")
        return

    # Set up specific output directory for this interpolation
    global OUTPUT_DIR  # Access the global variable
    original_output_dir = OUTPUT_DIR  # Store original
    if output_dir_suffix:
        OUTPUT_DIR = os.path.join(os.path.dirname(
            original_output_dir), output_dir_suffix)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Set output directory to: {OUTPUT_DIR}")

    # 1. Select the start image
    print(
        f"\nSelecting one image for start point ({attribute_name1}={attribute_value1_start}{f', {attribute_name2}={attribute_value2}' if attribute_name2 else ''})...")
    try:
        start_images, _ = select_images_with_specific_attributes(
            train_ds,
            attribute_indices=attribute_indices_start,
            desired_values=desired_values_start,
            num_samples=1
        )
        image_start = tf.expand_dims(start_images[0], axis=0)  # Add batch dim
        plot_images([image_start[0]], title=f"Original Start: {interpolation_title_base.split(' ')[0]} {attribute_value1_start}",
                    filename=f"original_start_{filename_base}.png")
    except ValueError as e:
        print(f"Error selecting start image: {e}")
        OUTPUT_DIR = original_output_dir  # Revert output dir
        return

    # 2. Select the end image
    print(
        f"\nSelecting one image for end point ({attribute_name1}={attribute_value1_end}{f', {attribute_name2}={attribute_value2}' if attribute_name2 else ''})...")
    try:
        end_images, _ = select_images_with_specific_attributes(
            train_ds,
            attribute_indices=attribute_indices_end,
            desired_values=desired_values_end,
            num_samples=1
        )
        image_end = tf.expand_dims(end_images[0], axis=0)  # Add batch dim
        plot_images([image_end[0]], title=f"Original End: {interpolation_title_base.split(' ')[0]} {attribute_value1_end}",
                    filename=f"original_end_{filename_base}.png")
    except ValueError as e:
        print(f"Error selecting end image: {e}")
        OUTPUT_DIR = original_output_dir  # Revert output dir
        return

    # Combine original images for a single plot
    plot_images([image_start[0], image_end[0]],
                title=f"Original Samples for Interpolation ({interpolation_title_base})",
                filename=f"original_samples_{filename_base}.png")

    # 3. Encode the selected images
    print("\nEncoding images into latent space...")
    z_mean_start, _, _ = vae_model.encode(image_start)
    z_mean_end, _, _ = vae_model.encode(image_end)
    print(f"Latent vector START shape: {z_mean_start.shape}")
    print(f"Latent vector END shape: {z_mean_end.shape}")

    # 4. Interpolate in latent space
    interpolation_factors = np.linspace(
        0, 1, num_interpolations)  # 0 to 1 in N steps

    interpolated_latent_vectors = []
    for alpha in interpolation_factors:
        interpolated_z = (1 - alpha) * z_mean_start + alpha * z_mean_end
        interpolated_latent_vectors.append(interpolated_z)

    # Concatenate to a single tensor for batch decoding
    interpolated_latent_vectors_batch = tf.concat(
        interpolated_latent_vectors, axis=0)
    print(f"Generated {num_interpolations} interpolated latent vectors.")

    # 5. Decode and visualize
    print("Decoding interpolated latent vectors...")
    decoded_images = vae_model.decode(interpolated_latent_vectors_batch)

    plot_images(decoded_images,
                title=f"Latent Space Interpolation: {interpolation_title_base} (Epoch {EPOCH})",
                filename=f"interpolation_sequence_{filename_base}.png")

    print(f"--- Interpolation complete for {interpolation_title_base} ---")

    OUTPUT_DIR = original_output_dir  # Revert output dir to original
    print(f"Output directory reverted to: {OUTPUT_DIR}")


if __name__ == "__main__":
    print("--- Starting General Latent Space Interpolation Script ---")

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

    # --- EXAMPLE 1: Mustache (No Hat -> With Hat) ---
    # print("\n" + "="*50)
    # print("Running Example 1: Mustache (No Hat -> With Hat)")
    # print("="*50)
    # perform_interpolation(
    #     vae_model=vae_model,
    #     train_ds=train_ds,
    #     celeba_attribute_names=celeba_attribute_names,
    #     attribute_name1='Wearing_Hat',       # The attribute that changes
    #     attribute_value1_start=0,            # Start with No Hat
    #     attribute_value1_end=1,              # End with With Hat
    #     attribute_name2='Mustache',          # Keep Mustache constant
    #     attribute_value2=1,                  # Keep Mustache present
    #     num_interpolations=10,
    #     output_dir_suffix="interpolated_mustache_hat_samples" # Specific folder for this example
    # )

    # --- EXAMPLE 2: Smiling (Not Smiling -> Smiling) ---
    print("\n" + "="*50)
    print("Running Example 2: Smiling (Not Smiling -> Smiling) while keeping Male")
    print("="*50)
    perform_interpolation(
        vae_model=vae_model,
        train_ds=train_ds,
        celeba_attribute_names=celeba_attribute_names,
        attribute_name1='Smiling',           # The attribute that changes
        attribute_value1_start=0,            # Start with Not Smiling
        attribute_value1_end=1,              # End with Smiling
        attribute_name2='Male',              # Keep Male constant
        # Keep Male present (Female is -1 in CelebA, so 1 is Male)
        attribute_value2=1,
        num_interpolations=10,
        # Specific folder for this example
        output_dir_suffix="interpolated_smiling_male_samples"
    )

    # --- EXAMPLE 3: Bangs (No Bangs -> With Bangs) while keeping Young ---
    print("\n" + "="*50)
    print("Running Example 3: Bangs (No Bangs -> With Bangs) while keeping Young")
    print("="*50)
    perform_interpolation(
        vae_model=vae_model,
        train_ds=train_ds,
        celeba_attribute_names=celeba_attribute_names,
        attribute_name1='Bangs',             # The attribute that changes
        attribute_value1_start=0,            # Start with No Bangs
        attribute_value1_end=1,              # End with With Bangs
        attribute_name2='Young',             # Keep Young constant
        attribute_value2=1,                  # Keep Young
        num_interpolations=10,
        # Specific folder for this example
        output_dir_suffix="interpolated_bangs_young_samples"
    )

    print("\n--- All Interpolation Examples Complete ---")
