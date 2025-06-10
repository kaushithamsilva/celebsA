import tensorflow as tf
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import random  # For random selection of attributes and values

# Import your custom modules
from vae import VAE, Sampling
from preprocess import get_celeba_datasets_with_splits
import model_utils

# --- Configuration Paths ---
SAVE_PATH = '../../models/ci_vae/'
CHECKPOINT_PATH = os.path.join(SAVE_PATH, 'checkpoints/')
EPOCH = 500  # The epoch number for the saved VAE model

VAE_MODEL_PATH = CHECKPOINT_PATH
VAE_MODEL_NAME = f'vae-e{EPOCH}'

IMAGE_DIR = '../data/img_align_celeba/img_align_celeba/'
ATTRIBUTES_CSV = '../data/list_attr_celeba.csv'
EVAL_PARTITION_CSV = '../data/list_eval_partition.csv'

# Base output directory for all attribute interpolations
# New base output directory for N-attribute interpolations
BASE_OUTPUT_DIR = '../figures/n_attribute_interpolation/'


# --- Model Parameters (MUST MATCH TRAINING) ---
INPUT_IMAGE_SIZE = (64, 64)
INPUT_IMAGE_CHANNELS = 3
LATENT_DIM = 128
HIDDEN_DIM = 64


def get_attribute_index(attribute_name, attr_names_list):
    """
    Helper to get the index of a specific attribute.
    """
    try:
        return attr_names_list.index(attribute_name)
    except ValueError:
        raise ValueError(
            f"Attribute '{attribute_name}' not found in attribute list. Available: {attr_names_list}")


def select_images_with_specific_attributes(dataset, attribute_indices, desired_values, attr_names_list, num_samples=1):
    """
    Selects a specified number of images that match ALL desired attribute values.
    Args:
        dataset (tf.data.Dataset): The dataset to search.
        attribute_indices (list): List of attribute indices to check.
        desired_values (list): List of desired binary values (0 or 1) corresponding to attribute_indices.
                               E.g., [1, 0] for (mustache=1, hat=0)
        attr_names_list (list): Full list of all CelebA attribute names (used for informative error messages).
        num_samples (int): Number of samples to find.
    Returns:
        tuple: (list of selected images, list of selected attribute tensors)
    Raises:
        ValueError: If a sufficient number of matching images cannot be found.
    """
    selected_images = []
    selected_attributes = []

    # Ensure desired_values are integers
    desired_values = [int(v) for v in desired_values]

    for image_batch, attr_batch in dataset:
        batch_mask = tf.constant(True, shape=(tf.shape(image_batch)[0],))

        for idx, desired_val in zip(attribute_indices, desired_values):
            attr_mask = (tf.cast(attr_batch[:, idx], tf.int32) == desired_val)
            batch_mask = tf.logical_and(batch_mask, attr_mask)

        images_matching = tf.boolean_mask(image_batch, batch_mask)
        attrs_matching = tf.boolean_mask(attr_batch, batch_mask)

        for i in range(tf.shape(images_matching)[0]):
            selected_images.append(images_matching[i])
            selected_attributes.append(attrs_matching[i])

            if len(selected_images) >= num_samples:
                return selected_images[:num_samples], selected_attributes[:num_samples]

    if len(selected_images) < num_samples:
        attr_search_str = ", ".join([f"{attr_names_list[a_idx]}={'Positive' if d_val == 1 else 'Negative'}"
                                     for a_idx, d_val in zip(attribute_indices, desired_values)])
        raise ValueError(
            f"Could not find {num_samples} images with attributes: [{attr_search_str}].")
    return selected_images[:num_samples], selected_attributes[:num_samples]


def plot_images(images, title="", filename="", output_dir=""):
    """
    Plots a list of images in a grid and saves them.
    """
    num_images = len(images)
    if num_images == 0:
        print("No images to plot.")
        return

    fig_cols = min(num_images, 5)
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
    plt.close()  # Close figure to free memory and prevent too many pop-up windows


def clean_name(name):
    """Helper to clean attribute names for filenames."""
    return name.replace(' ', '_').replace('/', '_').replace('__', '_')


# --- N-Attribute Interpolation Function ---
def perform_n_attribute_interpolation(
    vae_model,
    dataset,
    celeba_attribute_names,
    num_n_attributes=10,
    epoch=EPOCH,
    base_output_dir=BASE_OUTPUT_DIR,
    max_retries=5
):
    """
    Performs latent space interpolation between two images that differ by only one of N randomly selected attributes.

    Args:
        vae_model (tf.keras.Model): The loaded VAE model.
        dataset (tf.data.Dataset): The dataset to sample images from.
        celeba_attribute_names (list): List of all attribute names in CelebA.
        num_n_attributes (int): The number of attributes (N) to consider for the interpolation.
        epoch (int): The epoch of the VAE model for plot titles.
        base_output_dir (str): Base directory to save interpolation figures.
        max_retries (int): Maximum attempts to find a suitable pair of images.
    """
    print(
        f"\n--- Starting N-Attribute Interpolation (N={num_n_attributes}) ---")

    # A curated list of visually distinct attributes to increase chances of finding samples
    candidate_attributes = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Bags_Under_Eyes', 'Big_Lips', 'Big_Nose',
        'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
        'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Mouth_Slightly_Open',
        'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
        'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
        'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie'
    ]
    # Filter to ensure we only pick attributes present in the actual dataset's attribute list
    available_candidate_attributes = [
        attr for attr in candidate_attributes if attr in celeba_attribute_names]

    if len(available_candidate_attributes) < num_n_attributes:
        print(
            f"Error: Not enough distinct visual attributes ({len(available_candidate_attributes)}) available for N={num_n_attributes}.")
        print("Please reduce N or adjust the candidate_attributes list.")
        return

    for attempt in range(max_retries):
        try:
            # 1. Randomly select N distinct attributes from the candidate list
            chosen_attr_names = random.sample(
                available_candidate_attributes, num_n_attributes)
            chosen_attr_indices = [get_attribute_index(
                name, celeba_attribute_names) for name in chosen_attr_names]

            print(
                f"\nAttempt {attempt + 1}/{max_retries}: Chosen N attributes: {chosen_attr_names}")

            # 2. Randomly choose one attribute to be the changing one
            changing_attr_name = random.choice(chosen_attr_names)
            changing_attr_idx = get_attribute_index(
                changing_attr_name, celeba_attribute_names)
            print(f"Attribute chosen to change: '{changing_attr_name}'")

            # 3. Define fixed attributes and their random states (for the N-1 attributes)
            fixed_attr_names = [
                name for name in chosen_attr_names if name != changing_attr_name]
            fixed_attr_indices = [get_attribute_index(
                name, celeba_attribute_names) for name in fixed_attr_names]

            # Randomly assign 0 or 1 to each fixed attribute
            fixed_attr_values = [random.randint(
                0, 1) for _ in fixed_attr_indices]

            # Create a compact string for logging/folder names based on fixed attributes
            fixed_attr_str_list = [f"{clean_name(name)}_{val}" for name, val in zip(
                fixed_attr_names, fixed_attr_values)]
            fixed_attrs_combined_str = "_".join(fixed_attr_str_list)
            # Handle case where N=1 (no fixed attributes)
            if not fixed_attrs_combined_str:
                fixed_attrs_combined_str = "no_fixed_attrs"

            print(
                f"Fixed attributes and their values: {list(zip(fixed_attr_names, fixed_attr_values))}")

            # 4. Construct Search Criteria for Image A (changing_attr=0) and Image B (changing_attr=1)
            # Image A: fixed_attributes as determined, changing_attribute = 0
            image_A_attr_indices = fixed_attr_indices + [changing_attr_idx]
            image_A_desired_values = fixed_attr_values + \
                [0]  # Assuming 0 is 'absence' or 'negative'

            # Image B: fixed_attributes as determined, changing_attribute = 1
            image_B_attr_indices = fixed_attr_indices + [changing_attr_idx]
            image_B_desired_values = fixed_attr_values + \
                [1]  # Assuming 1 is 'presence' or 'positive'

            # Create a specific output directory for this interpolation pair
            clean_changing_attr_name = clean_name(changing_attr_name)

            # Ensure a unique directory name for each attempt, incorporating fixed attributes
            current_output_dir = os.path.join(
                base_output_dir,
                f"N{num_n_attributes}_changing_{clean_changing_attr_name}_fixed_{fixed_attrs_combined_str}_run{attempt+1}"
            )
            os.makedirs(current_output_dir, exist_ok=True)
            print(f"Output will be saved to: {current_output_dir}")

            # 5. Find Images A and B
            print(
                f"\nSearching for Image A (changing_attr='{changing_attr_name}'=0) with {num_n_attributes-1} fixed attributes...")
            img_A_samples, _ = select_images_with_specific_attributes(
                dataset,
                attribute_indices=image_A_attr_indices,
                desired_values=image_A_desired_values,
                # Pass full attribute list for error messages
                attr_names_list=celeba_attribute_names,
                num_samples=1
            )
            image_A = tf.expand_dims(
                img_A_samples[0], axis=0)  # Add batch dimension

            print(
                f"Searching for Image B (changing_attr='{changing_attr_name}'=1) with {num_n_attributes-1} fixed attributes...")
            img_B_samples, _ = select_images_with_specific_attributes(
                dataset,
                attribute_indices=image_B_attr_indices,
                desired_values=image_B_desired_values,
                # Pass full attribute list for error messages
                attr_names_list=celeba_attribute_names,
                num_samples=1
            )
            image_B = tf.expand_dims(
                img_B_samples[0], axis=0)  # Add batch dimension

            # If we reach here, both images are found successfully
            plot_images([image_A[0], image_B[0]],
                        title=f"Original Samples (N={num_n_attributes}) \nChanging '{changing_attr_name}' (0 -> 1)",
                        filename=f"original_N{num_n_attributes}_interpolation_samples.png",
                        output_dir=current_output_dir)

            # 6. Encode the selected images
            print("\nEncoding images into latent space...")
            z_mean_A, _, _ = vae_model.encode(image_A)
            z_mean_B, _, _ = vae_model.encode(image_B)
            print(f"Latent vector A (changing_attr=0) shape: {z_mean_A.shape}")
            print(f"Latent vector B (changing_attr=1) shape: {z_mean_B.shape}")

            # 7. Interpolate in latent space
            num_interpolations = 10
            interpolation_factors = np.linspace(0, 1, num_interpolations)

            interpolated_latent_vectors = []
            for alpha in interpolation_factors:
                interpolated_z = (1 - alpha) * z_mean_A + alpha * z_mean_B
                interpolated_latent_vectors.append(interpolated_z)

            interpolated_latent_vectors_batch = tf.concat(
                interpolated_latent_vectors, axis=0)
            print(
                f"Generated {num_interpolations} interpolated latent vectors.")

            # 8. Decode and visualize
            print("Decoding interpolated latent vectors...")
            decoded_images = vae_model.decode(
                interpolated_latent_vectors_batch)

            plot_images(decoded_images,
                        title=f"Interpolation: Changing '{changing_attr_name}' (N={num_n_attributes}) (Epoch {epoch})",
                        filename=f"N{num_n_attributes}_interpolation_sequence_{clean_changing_attr_name}.png",
                        output_dir=current_output_dir)

            print(
                f"--- N-Attribute Interpolation (N={num_n_attributes}, changing '{changing_attr_name}') complete ---")
            return  # Successfully found and interpolated, exit the retry loop

        except ValueError as e:
            print(
                f"Failed to find suitable images for N-attribute interpolation (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                print(
                    "Max retries reached. Could not find a suitable pair of images. Consider reducing N or increasing max_retries.")
            # Continue to next attempt
        except Exception as e:
            print(
                f"An unexpected error occurred during N-attribute interpolation (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                print("Max retries reached due to unexpected errors.")
            # Continue to next attempt


if __name__ == "__main__":
    print("--- Initializing Latent Space Interpolation Script ---")

    # 1. Load VAE model
    print(f"Loading VAE model from {VAE_MODEL_PATH}...")
    vae_model = model_utils.load_model(VAE_MODEL_PATH, VAE_MODEL_NAME)
    if vae_model is None:
        print("Failed to load VAE model. Exiting.")
        exit()
    print("VAE model loaded successfully.")

    # 2. Load CelebA dataset (using the training dataset for broader sample finding)
    print("Loading CelebA training dataset for image selection...")
    attr_df_for_names = pd.read_csv(ATTRIBUTES_CSV, index_col=0)
    celeba_attribute_names = attr_df_for_names.columns.tolist()

    train_ds, _, _ = get_celeba_datasets_with_splits(
        image_dir=IMAGE_DIR,
        attr_csv_path=ATTRIBUTES_CSV,
        eval_csv_path=EVAL_PARTITION_CSV,
        image_size=INPUT_IMAGE_SIZE,
        batch_size=64  # Increased batch size for potentially faster sample finding
    )
    print("Dataset loaded.")

    # --- Perform N-Attribute Interpolation ---
    # This will attempt to find a pair of images with N=10 attributes where only one changes.
    # It will retry up to 10 times with different random attribute sets if it fails.
    perform_n_attribute_interpolation(vae_model, train_ds, celeba_attribute_names,
                                      num_n_attributes=10, epoch=EPOCH, max_retries=10)

    # You can add more calls here with different N values to explore
    # For example, to try with N=5 attributes:
    # perform_n_attribute_interpolation(vae_model, train_ds, celeba_attribute_names,
    #                                   num_n_attributes=5, epoch=EPOCH, max_retries=10)

    print("\n--- All requested interpolations complete ---")
