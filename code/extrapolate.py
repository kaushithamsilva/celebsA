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
# DISCRIMINATOR_SAVE_PATH = os.path.join(SAVE_PATH, 'discriminators/')
DISCRIMINATOR_SAVE_PATH = CHECKPOINT_PATH
EPOCH = 400  # The epoch number for the saved VAE and discriminator checkpoints

VAE_MODEL_PATH = CHECKPOINT_PATH
VAE_MODEL_NAME = f'vae-e{EPOCH}'  # Load the VAE from the specified epoch

IMAGE_DIR = '../data/img_align_celeba/img_align_celeba/'
ATTRIBUTES_CSV = '../data/list_attr_celeba.csv'
EVAL_PARTITION_CSV = '../data/list_eval_partition.csv'

# Output directory for extrapolated images
OUTPUT_DIR = '../figures/extrapolated_hat_mustache_samples/'
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


def select_images_with_attribute(dataset, attribute_index, num_samples=1):
    """
    Selects a specified number of images that have the given attribute.
    Iterates over batches and collects samples.
    """
    selected_images = []
    selected_attributes = []

    for image_batch, attr_batch in dataset:
        has_attribute_mask = (attr_batch[:, attribute_index] == 1)

        images_with_attribute = tf.boolean_mask(
            image_batch, has_attribute_mask)
        attrs_with_attribute = tf.boolean_mask(attr_batch, has_attribute_mask)

        for i in range(tf.shape(images_with_attribute)[0]):
            selected_images.append(images_with_attribute[i])
            selected_attributes.append(attrs_with_attribute[i])

            if len(selected_images) >= num_samples:
                return selected_images[:num_samples], selected_attributes[:num_samples]

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
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if filename:
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        print(f"Saved plot to {os.path.join(OUTPUT_DIR, filename)}")
    plt.show()


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

    # 3. Select one image with 'Mustache'
    print(f"Selecting one image with 'Mustache' attribute...")
    try:
        mustache_images, _ = select_images_with_attribute(
            train_ds, mustache_attr_idx, num_samples=1)
        original_mustache_image = tf.expand_dims(
            mustache_images[0], axis=0)  # Add batch dim
        plot_images([original_mustache_image[0]], title="Original Mustache Image",
                    filename="original_mustache_image.png")
    except ValueError as e:
        print(f"Error selecting original image: {e}")
        exit()

    # 4. Load the discriminators
    print("Loading attribute discriminators...")
    mustache_disc_name = f"{celeba_attribute_names[mustache_attr_idx].replace(' ', '_')}_discriminator-e{EPOCH}"
    wearing_hat_disc_name = f"{celeba_attribute_names[wearing_hat_attr_idx].replace(' ', '_')}_discriminator-e{EPOCH}"

    mustache_discriminator_model = model_utils.load_model(
        CHECKPOINT_PATH, mustache_disc_name)
    wearing_hat_discriminator_model = model_utils.load_model(
        CHECKPOINT_PATH, wearing_hat_disc_name)

    if mustache_discriminator_model is None or wearing_hat_discriminator_model is None:
        print("Failed to load one or both discriminator models. Exiting.")
        exit()
    print("Discriminator models loaded successfully.")

    # 5. Extract hyperplane parameters
    mustache_hyperplane = Hyperplane(mustache_discriminator_model)
    wearing_hat_hyperplane = Hyperplane(wearing_hat_discriminator_model)

    # Get the direction vector for 'Wearing_Hat'
    # The normal vector 'w' of the hyperplane points in the direction of increasing probability for the positive class.
    hat_direction_vector, _ = wearing_hat_hyperplane.get_hyplerplane_params()
    # Ensure it's a (1, latent_dim) tensor for addition later
    hat_direction_vector = tf.expand_dims(hat_direction_vector, axis=0)
    print(f"Hat direction vector shape: {hat_direction_vector.shape}")

    # 6. Encode the original mustache image
    initial_z_mean, _, _ = vae_model.encode(original_mustache_image)
    current_z = tf.identity(initial_z_mean)  # Start manipulation from here

    # 7. Extrapolate
    num_extrapolation_steps = 10  # How many steps to take
    # How much to move in the hat direction per step (tune this!)
    step_size = 0.5

    # Start with the original image
    generated_images = [original_mustache_image[0]]

    print("\nStarting extrapolation...")
    for i in range(num_extrapolation_steps):
        # Calculate current 'Mustache' score
        mustache_score_current = mustache_discriminator_model(current_z).numpy()[
            0, 0]

        # Check if still in 'mustache' region (positive score implies mustache present)
        # Assuming 0 is the decision boundary for binary classification.
        # A positive logit implies "present" and negative implies "absent".
        if mustache_score_current < -0.5:  # Tune this threshold, e.g., -0.5 or 0.0
            print(
                f"Stopped at step {i+1}: Latent vector moved out of 'Mustache' region. Score: {mustache_score_current:.2f}")
            break

        # Move in the 'Wearing_Hat' direction
        current_z = current_z + step_size * hat_direction_vector

        # Decode the new latent vector
        decoded_img = vae_model.decode(current_z)
        generated_images.append(decoded_img[0])

        # Calculate new 'Mustache' and 'Wearing_Hat' scores for monitoring
        mustache_score_new = mustache_discriminator_model(current_z).numpy()[
            0, 0]
        hat_score_new = wearing_hat_discriminator_model(current_z).numpy()[
            0, 0]
        print(
            f"Step {i+1}: Mustache Score = {mustache_score_new:.2f}, Hat Score = {hat_score_new:.2f}")

    print(
        f"Extrapolation completed. Generated {len(generated_images)} images.")

    # 8. Visualize results
    plot_images(generated_images,
                title=f"Extrapolation: Mustache to Hat (Epoch {EPOCH})",
                filename="extrapolation_mustache_to_hat.png")

    print("--- Extrapolation complete ---")
