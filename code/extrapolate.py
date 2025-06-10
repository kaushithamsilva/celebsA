import tensorflow as tf
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# Import your custom modules
from vae import VAE, Sampling
from preprocess import get_celeba_datasets_with_splits
import model_utils
from hyperplane import Hyperplane

# --- Configuration Paths (MUST BE CORRECTED) ---
# Base directory where your models are saved
SAVE_PATH = '../../models/ci_vae/'
CHECKPOINT_PATH = os.path.join(SAVE_PATH, 'checkpoints/')
DISCRIMINATOR_SAVE_PATH = CHECKPOINT_PATH
EPOCH = 400  # The epoch number for the saved VAE and discriminator checkpoints

VAE_MODEL_PATH = CHECKPOINT_PATH
VAE_MODEL_NAME = f'vae-e{EPOCH}'  # Load the VAE from the specified epoch

IMAGE_DIR = '../data/img_align_celeba/img_align_celeba/'
ATTRIBUTES_CSV = '../data/list_attr_celeba.csv'
EVAL_PARTITION_CSV = '../data/list_eval_partition.csv'

# Output directory for extrapolated images
# Updated output directory for new attributes
OUTPUT_DIR = '../figures/extrapolated_male_bald_to_hair_samples/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
            attr_mask = (tf.cast(attr_batch[:, idx], tf.int32) == desired_val)
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
        attr_search_str = ", ".join([
            f"{attr_names_list[a_idx]}={'Positive' if d_val == 1 else 'Negative'}"
            for a_idx, d_val in zip(attribute_indices, desired_values)
        ])
        raise ValueError(
            f"Could not find {num_samples} images with attributes: {attr_search_str}. Available: {attr_names_list}")
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
    plt.close()  # Close the figure to free memory


if __name__ == "__main__":
    print("--- Starting Latent Space Extrapolation ---")

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
        batch_size=32
    )
    print("Dataset loaded.")

    # Get attribute indices for 'Male' and 'Bald'
    try:
        male_attr_idx = get_attribute_index('Male', celeba_attribute_names)
        bald_attr_idx = get_attribute_index('Bald', celeba_attribute_names)

        print(f"Found 'Male' attribute at index: {male_attr_idx}")
        print(f"Found 'Bald' attribute at index: {bald_attr_idx}")
    except ValueError as e:
        print(f"Error: {e}")
        exit()

    # 3. Select one initial image: Male and Bald
    print(f"Selecting one image that is 'Male' and 'Bald'...")
    try:
        # Desired: Male=1, Bald=1
        initial_images, _ = select_images_with_specific_attributes(
            train_ds,
            attribute_indices=[male_attr_idx, bald_attr_idx],
            desired_values=[1, 1],
            num_samples=1
        )
        original_image = tf.expand_dims(initial_images[0], axis=0)
        plot_images([original_image[0]], title="Original Male, Bald Image",
                    filename="original_male_bald_image.png")
    except ValueError as e:
        print(f"Error selecting initial image: {e}")
        print("Could not find a 'Male' and 'Bald' image. Please ensure such images exist and are accessible in your dataset.")
        exit()

    # 4. Load the discriminators for the attributes we care about (Male, Bald)
    print("Loading attribute discriminators...")
    male_disc_name = f"{celeba_attribute_names[male_attr_idx].replace(' ', '_')}_discriminator-e{EPOCH}"
    bald_disc_name = f"{celeba_attribute_names[bald_attr_idx].replace(' ', '_')}_discriminator-e{EPOCH}"

    male_discriminator_model = model_utils.load_model(
        CHECKPOINT_PATH, male_disc_name)
    bald_discriminator_model = model_utils.load_model(
        CHECKPOINT_PATH, bald_disc_name)

    if male_discriminator_model is None or bald_discriminator_model is None:
        print("Failed to load one or both discriminator models. Exiting.")
        exit()
    print("Discriminator models loaded successfully.")

    # 5. Extract hyperplane parameters for 'Bald' attribute (this will be our direction)
    bald_hyperplane = Hyperplane(bald_discriminator_model)
    # The normal vector 'w' for 'Bald' attribute will point in the direction of 'Baldness'
    # To go from Bald (1) to Not Bald (0), we need to move in the *opposite* direction of the baldness vector.
    bald_direction_vector, _ = bald_hyperplane.get_hyplerplane_params()

    # We want to go from Bald (positive score) to Not Bald (negative score),
    # so we'll move in the negative direction of the 'Bald' hyperplane normal vector.
    # Negate the vector to move towards 'Not Bald'
    direction_vector = -tf.expand_dims(bald_direction_vector, axis=0)
    print(
        f"Direction vector (towards Not Bald) shape: {direction_vector.shape}")

    # 6. Encode the original image (Male and Bald)
    initial_z_mean, _, _ = vae_model.encode(original_image)
    current_z = tf.identity(initial_z_mean)

    # 7. Extrapolate with pull-to-center
    num_extrapolation_steps = 40
    step_size = 0.2  # Controls how much attribute changes per step
    # Controls how strongly latent vector is pulled towards center (0.01 to 0.1 is a good starting range)
    pull_strength = 0.01

    # Define a threshold for the latent vector norm (distance from origin)
    max_latent_norm_threshold = 10 * np.sqrt(LATENT_DIM)
    print(f"Maximum allowed latent norm: {max_latent_norm_threshold:.2f}")
    print(f"Pull-to-center strength (beta): {pull_strength}")

    # Start with the original image
    generated_images = [original_image[0]]

    print("\nStarting extrapolation with pull-to-center (Male, Bald -> Male, Not Bald)...")
    for i in range(num_extrapolation_steps):
        # Calculate current 'Bald' score (this is the attribute we are changing)
        bald_score_current = bald_discriminator_model(current_z).numpy()[0, 0]
        # Calculate current 'Male' score (this should remain stable)
        male_score_current = male_discriminator_model(current_z).numpy()[0, 0]

        # Calculate current latent vector norm
        current_latent_norm = tf.norm(current_z).numpy()

        # Check conditions to stop extrapolation
        # Condition 1: 'Bald' attribute changes (we want to reach a 'Not Bald' state)
        # Assuming a negative score implies 'Not Bald'
        if bald_score_current < -30.0:  # Tuned threshold for 'Bald' to 'Not Bald' transition
            print(
                f"Stopped at step {i+1}: Latent vector reached 'Not Bald' region. Bald Score: {bald_score_current:.2f}")
            break

        # Condition 2: Latent vector goes too far from the origin (to maintain quality/validity)
        if current_latent_norm > max_latent_norm_threshold:
            print(
                f"Stopped at step {i+1}: Latent vector norm ({current_latent_norm:.2f}) exceeded threshold ({max_latent_norm_threshold:.2f}).")
            break

        # Condition 3: Ensure 'Male' attribute is preserved (optional, but good for control)
        # You might want to adjust the threshold for 'Male' score, e.g., > 0.0 or > -0.5
        # If 'Male' score drops significantly, you might be changing gender, which is undesirable.
        if male_score_current < -0.5:  # Assuming positive score means male, negative means not male
            print(
                f"Stopped at step {i+1}: Latent vector moved out of 'Male' region. Male Score: {male_score_current:.2f}")
            break

        # Apply the pull-to-center logic and move along the direction vector
        current_z = (1 - pull_strength) * current_z + \
            step_size * direction_vector

        # Decode the new latent vector
        decoded_img = vae_model.decode(current_z)
        generated_images.append(decoded_img[0])

        # Calculate new scores for monitoring
        bald_score_new = bald_discriminator_model(current_z).numpy()[0, 0]
        male_score_new = male_discriminator_model(current_z).numpy()[0, 0]
        print(
            f"Step {i+1}: Bald Score = {bald_score_new:.2f}, Male Score = {male_score_new:.2f}, Latent Norm = {tf.norm(current_z).numpy():.2f}")

    print(
        f"Extrapolation completed. Generated {len(generated_images)} images.")

    # 8. Visualize results
    plot_images(generated_images,
                title=f"Extrapolation: Male, Bald to Male, Hair (Epoch {EPOCH}, Beta={pull_strength})",
                filename="extrapolation_male_bald_to_hair_pull_to_center.png")

    print("--- Extrapolation complete ---")
