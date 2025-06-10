import tensorflow as tf
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# Import your custom modules
from vae import VAE, Sampling  # Assuming VAE and Sampling classes are in vae.py
from preprocess import get_celeba_datasets_with_splits
import model_utils  # Your model_utils.py with save_model and load_model
# from hyperplane import Hyperplane  # No longer directly needed for this method

# --- Configuration Paths (MUST BE CORRECTED) ---
# Base directory where your models are saved
SAVE_PATH = '../../models/ci_vae/'
CHECKPOINT_PATH = os.path.join(SAVE_PATH, 'checkpoints/')
EPOCH = 500  # The epoch number for the saved VAE and discriminator checkpoints

VAE_MODEL_PATH = CHECKPOINT_PATH
VAE_MODEL_NAME = f'vae-e{EPOCH}'  # Load the VAE from the specified epoch

IMAGE_DIR = '../data/img_align_celeba/img_align_celeba/'
ATTRIBUTES_CSV = '../data/list_attr_celeba.csv'
EVAL_PARTITION_CSV = '../data/list_eval_partition.csv'

# Output directory for extrapolated images
# Changed output directory
OUTPUT_DIR = '../figures/extrapolated_average_direction_samples/'
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


def select_images_with_specific_attributes(dataset, attribute_indices, desired_values, num_samples):
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
        # Ensure image data type is float for imshow (it expects float values in [0,1])
        plt.imshow(img.numpy() if tf.is_tensor(img) else img)
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if filename:
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        print(f"Saved plot to {os.path.join(OUTPUT_DIR, filename)}")
    plt.show()


if __name__ == "__main__":
    print("--- Starting Latent Space Extrapolation using Average Direction ---")

    # 1. Load VAE model
    print(f"Loading VAE model from {VAE_MODEL_PATH}...")
    vae_model = model_utils.load_model(VAE_MODEL_PATH, VAE_MODEL_NAME)
    if vae_model is None:
        print("Failed to load VAE model. Exiting.")
        exit()
    print("VAE model loaded successfully.")

    # 2. Load CelebA dataset for image selection and attribute names
    print("Loading CelebA training dataset for image selection...")
    attr_df_for_names = pd.read_csv(ATTRIBUTES_CSV, index_col=0)
    celeba_attribute_names = attr_df_for_names.columns.tolist()

    train_ds, _, _ = get_celeba_datasets_with_splits(
        image_dir=IMAGE_DIR,
        attr_csv_path=ATTRIBUTES_CSV,
        eval_csv_path=EVAL_PARTITION_CSV,
        image_size=INPUT_IMAGE_SIZE,
        batch_size=32  # Small batch size for iterating to find samples
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

    # 3. Collect samples for calculating the average direction
    # Number of samples to average for direction calculation
    num_samples_for_direction = 100

    print(
        f"\nCollecting {num_samples_for_direction} 'Mustache, No Hat' samples for direction calculation...")
    try:
        mustache_no_hat_samples, _ = select_images_with_specific_attributes(
            train_ds,
            attribute_indices=[mustache_attr_idx, wearing_hat_attr_idx],
            desired_values=[1, 0],
            num_samples=num_samples_for_direction
        )
    except ValueError as e:
        print(f"Error collecting Mustache & No Hat samples: {e}")
        print("Consider reducing num_samples_for_direction or checking your dataset.")
        exit()
    print(
        f"Collected {len(mustache_no_hat_samples)} Mustache, No Hat samples.")

    print(
        f"\nCollecting {num_samples_for_direction} 'Mustache, With Hat' samples for direction calculation...")
    try:
        mustache_with_hat_samples, _ = select_images_with_specific_attributes(
            train_ds,
            attribute_indices=[mustache_attr_idx, wearing_hat_attr_idx],
            desired_values=[1, 1],
            num_samples=num_samples_for_direction
        )
    except ValueError as e:
        print(f"Error collecting Mustache & With Hat samples: {e}")
        print("Consider reducing num_samples_for_direction or checking your dataset.")
        exit()
    print(
        f"Collected {len(mustache_with_hat_samples)} Mustache, With Hat samples.")

    # 4. Encode samples and calculate average latent vectors
    print("\nEncoding samples to calculate average direction...")
    # Encode 'mustache, no hat' samples
    z_means_mustache_no_hat = []
    for img in mustache_no_hat_samples:
        z_mean, _, _ = vae_model.encode(tf.expand_dims(img, axis=0))
        z_means_mustache_no_hat.append(z_mean)
    mean_z_mustache_no_hat = tf.reduce_mean(
        tf.concat(z_means_mustache_no_hat, axis=0), axis=0)

    # Encode 'mustache, with hat' samples
    z_means_mustache_with_hat = []
    for img in mustache_with_hat_samples:
        z_mean, _, _ = vae_model.encode(tf.expand_dims(img, axis=0))
        z_means_mustache_with_hat.append(z_mean)
    mean_z_mustache_with_hat = tf.reduce_mean(
        tf.concat(z_means_mustache_with_hat, axis=0), axis=0)

    # Calculate the average direction vector (from no hat to with hat)
    average_hat_direction_vector = mean_z_mustache_with_hat - mean_z_mustache_no_hat
    # Normalize the direction vector (optional, but good for consistent step sizes)
    average_hat_direction_vector = average_hat_direction_vector / \
        tf.norm(average_hat_direction_vector)

    # Ensure it's (1, latent_dim) for consistent addition later
    average_hat_direction_vector = tf.expand_dims(
        average_hat_direction_vector, axis=0)
    print(
        f"Average Hat Direction Vector shape: {average_hat_direction_vector.shape}")

    # 5. Select a random starting image (Mustache, No Hat) for traversal
    print("\nSelecting a random 'Mustache, No Hat' image to start traversal...")
    try:
        random_mustache_no_hat_image_list, _ = select_images_with_specific_attributes(
            train_ds,
            attribute_indices=[mustache_attr_idx, wearing_hat_attr_idx],
            desired_values=[1, 0],
            num_samples=1
        )
        random_original_image = tf.expand_dims(
            random_mustache_no_hat_image_list[0], axis=0)
        plot_images([random_original_image[0]], title="Random Start Image: Mustache, No Hat",
                    filename="random_start_mustache_no_hat.png")
    except ValueError as e:
        print(f"Error selecting random starting image: {e}")
        exit()

    # Encode the random starting image
    initial_z_for_traversal, _, _ = vae_model.encode(random_original_image)
    # Start manipulation from here
    current_z = tf.identity(initial_z_for_traversal)

    # 6. Traverse (Extrapolate) in the calculated direction
    num_extrapolation_steps = 10  # How many steps to take
    # How much to move along the average direction per step (tune this!)
    step_size = 1.0

    # Start with the original image
    generated_images = [random_original_image[0]]

    print("\nStarting latent space traversal using average direction...")
    for i in range(num_extrapolation_steps):
        # Move in the calculated average 'Wearing_Hat' direction
        current_z = current_z + step_size * average_hat_direction_vector

        # Decode the new latent vector
        decoded_img = vae_model.decode(current_z)
        generated_images.append(decoded_img[0])

        print(f"Step {i+1}: Traversed along average hat direction.")

    print(f"Traversal completed. Generated {len(generated_images)} images.")

    # 7. Visualize results
    plot_images(generated_images,
                title=f"Extrapolation: Mustache (No Hat -> With Hat) by Average Direction (Epoch {EPOCH})",
                filename="extrapolation_mustache_to_hat_avg_direction.png")

    print("--- Extrapolation complete ---")
