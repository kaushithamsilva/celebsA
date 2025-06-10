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

# Base output directory for extrapolated images
BASE_OUTPUT_DIR = '../figures/extrapolated_samples/'
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

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


def select_images_with_specific_attributes(dataset, attr_names_list, attribute_criteria, num_samples=1):
    """
    Selects a specified number of images that match ALL desired attribute criteria.
    Args:
        dataset (tf.data.Dataset): The dataset to search.
        attr_names_list (list): List of all attribute names for index lookup.
        attribute_criteria (dict): A dictionary where keys are attribute names (str)
                                   and values are their desired binary states (0 or 1).
                                   E.g., {"Male": 1, "Bald": 1}
        num_samples (int): Number of samples to find.
    Returns:
        tuple: (list of selected images, list of selected attribute tensors)
    """
    selected_images = []
    selected_attributes_full = []  # Store full attribute vectors for debugging if needed

    # Prepare attribute indices and desired values from criteria
    attribute_indices_to_check = []
    desired_values_for_check = []
    for attr_name, attr_value in attribute_criteria.items():
        attr_idx = get_attribute_index(attr_name, attr_names_list)
        attribute_indices_to_check.append(attr_idx)
        desired_values_for_check.append(int(attr_value))

    if not attribute_indices_to_check:
        raise ValueError("No attribute criteria provided for image selection.")

    for image_batch, attr_batch in dataset:
        batch_mask = tf.constant(True, shape=(tf.shape(image_batch)[0],))

        for idx, desired_val in zip(attribute_indices_to_check, desired_values_for_check):
            attr_mask = (tf.cast(attr_batch[:, idx], tf.int32) == desired_val)
            batch_mask = tf.logical_and(batch_mask, attr_mask)

        images_matching = tf.boolean_mask(image_batch, batch_mask)
        attrs_matching = tf.boolean_mask(attr_batch, batch_mask)

        for i in range(tf.shape(images_matching)[0]):
            selected_images.append(images_matching[i])
            # Store full attribute vector
            selected_attributes_full.append(attrs_matching[i])

            if len(selected_images) >= num_samples:
                print(
                    f"Found {len(selected_images)} images matching criteria.")
                return selected_images[:num_samples], selected_attributes_full[:num_samples]

    if len(selected_images) < num_samples:
        criteria_str = ", ".join(
            [f"{name}={val}" for name, val in attribute_criteria.items()])
        raise ValueError(
            f"Could not find {num_samples} images with attributes: {criteria_str}.")
    return selected_images[:num_samples], selected_attributes_full[:num_samples]


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

    if filename:
        full_path = os.path.join(output_dir, filename)
        plt.savefig(full_path)
        print(f"Saved plot to {full_path}")
    plt.show()
    plt.close()


def run_extrapolation_experiment(
    vae_model,
    celeba_attribute_names,
    train_ds,
    experiment_params
):
    """
    Runs a generalized latent space extrapolation experiment.

    Args:
        vae_model (tf.keras.Model): The loaded VAE model.
        celeba_attribute_names (list): List of all attribute names from CelebA.
        train_ds (tf.data.Dataset): The training dataset for image selection.
        experiment_params (dict): A dictionary containing experiment configuration:
            - 'initial_image_criteria' (dict): E.g., {"Male": 1, "Bald": 1}
            - 'attribute_to_change' (str): The name of the attribute to change.
            - 'change_direction_towards_positive' (bool): True to increase score, False to decrease.
            - 'target_attr_stop_threshold' (float): Discriminator score to stop at for the changing attribute.
            - 'fixed_attribute' (str, optional): Name of attribute to preserve.
            - 'fixed_attr_stability_threshold' (float, optional): Discriminator score threshold for fixed attribute.
            - 'num_extrapolation_steps' (int): Number of steps for extrapolation.
            - 'step_size' (float): Step size for latent walk.
            - 'pull_strength' (float): Beta parameter for pull-to-center.
            - 'output_filename_suffix' (str): Suffix for output filenames.
            - 'title' (str): Title for the plot.
    """
    print(f"\n--- Starting Experiment: {experiment_params['title']} ---")

    # Set up experiment-specific output directory
    experiment_output_dir = os.path.join(
        BASE_OUTPUT_DIR, experiment_params['output_filename_suffix'])
    os.makedirs(experiment_output_dir, exist_ok=True)
    # Temporarily update global OUTPUT_DIR for plot_images if needed, or pass it explicitly
    global OUTPUT_DIR
    OUTPUT_DIR = experiment_output_dir  # Update global for plot_images usage

    # Get attribute indices
    try:
        attr_to_change_idx = get_attribute_index(
            experiment_params['attribute_to_change'], celeba_attribute_names)

        fixed_attr_idx = None
        if experiment_params.get('fixed_attribute'):
            fixed_attr_idx = get_attribute_index(
                experiment_params['fixed_attribute'], celeba_attribute_names)
    except ValueError as e:
        print(f"Error: {e}")
        return  # Exit experiment if attributes not found

    # 3. Select one initial image based on criteria
    print(
        f"Selecting one image with criteria: {experiment_params['initial_image_criteria']}...")
    try:
        initial_images, _ = select_images_with_specific_attributes(
            train_ds,
            celeba_attribute_names,
            experiment_params['initial_image_criteria'],
            num_samples=1
        )
        original_image = tf.expand_dims(initial_images[0], axis=0)
        plot_images([original_image[0]], title=f"Original Image: {experiment_params['title']}",
                    filename="original_image.png", output_dir=experiment_output_dir)
    except ValueError as e:
        print(f"Error selecting initial image: {e}")
        print("Please ensure such images exist and are accessible in your dataset.")
        return

    # 4. Load relevant discriminators
    print("Loading attribute discriminators...")
    attr_to_change_disc_name = f"{experiment_params['attribute_to_change'].replace(' ', '_')}_discriminator-e{EPOCH}"
    attr_to_change_discriminator_model = model_utils.load_model(
        CHECKPOINT_PATH, attr_to_change_disc_name)

    fixed_discriminator_model = None
    if fixed_attr_idx is not None:
        fixed_disc_name = f"{experiment_params['fixed_attribute'].replace(' ', '_')}_discriminator-e{EPOCH}"
        fixed_discriminator_model = model_utils.load_model(
            CHECKPOINT_PATH, fixed_disc_name)

    if attr_to_change_discriminator_model is None or (fixed_attr_idx is not None and fixed_discriminator_model is None):
        print("Failed to load one or more discriminator models. Exiting experiment.")
        return
    print("Discriminator models loaded successfully.")

    # 5. Extract hyperplane parameters for the changing attribute
    attr_to_change_hyperplane = Hyperplane(attr_to_change_discriminator_model)
    base_direction_vector, _ = attr_to_change_hyperplane.get_hyplerplane_params()

    # Determine the actual direction for extrapolation
    if experiment_params['change_direction_towards_positive']:
        # Move in the positive direction of the attribute's normal vector
        direction_vector = tf.expand_dims(base_direction_vector, axis=0)
    else:
        # Move in the negative direction (e.g., Bald to Not Bald, which means less baldness)
        direction_vector = -tf.expand_dims(base_direction_vector, axis=0)

    print(
        f"Extrapolation direction vector (for {experiment_params['attribute_to_change']}) shape: {direction_vector.shape}")

    # 6. Encode the original image
    initial_z_mean, _, _ = vae_model.encode(original_image)
    current_z = tf.identity(initial_z_mean)

    # 7. Extrapolate with pull-to-center
    num_extrapolation_steps = experiment_params['num_extrapolation_steps']
    step_size = experiment_params['step_size']
    pull_strength = experiment_params['pull_strength']

    max_latent_norm_threshold = 10 * np.sqrt(LATENT_DIM)  # General safeguard
    print(f"Maximum allowed latent norm: {max_latent_norm_threshold:.2f}")
    print(f"Pull-to-center strength (beta): {pull_strength}")

    generated_images = [original_image[0]]

    print("\nStarting extrapolation...")
    for i in range(num_extrapolation_steps):
        attr_to_change_score_current = attr_to_change_discriminator_model(
            current_z).numpy()[0, 0]
        fixed_attr_score_current = None
        if fixed_discriminator_model:
            fixed_attr_score_current = fixed_discriminator_model(current_z).numpy()[
                0, 0]

        current_latent_norm = tf.norm(current_z).numpy()

        # Stop conditions
        # Condition 1: Target attribute state reached
        if (experiment_params['change_direction_towards_positive'] and attr_to_change_score_current > experiment_params['target_attr_stop_threshold']) or \
           (not experiment_params['change_direction_towards_positive'] and attr_to_change_score_current < experiment_params['target_attr_stop_threshold']):
            print(
                f"Stopped at step {i+1}: Target attribute '{experiment_params['attribute_to_change']}' state reached. Score: {attr_to_change_score_current:.2f}")
            break

        # Condition 2: Latent vector too far from origin
        if current_latent_norm > max_latent_norm_threshold:
            print(
                f"Stopped at step {i+1}: Latent vector norm ({current_latent_norm:.2f}) exceeded threshold ({max_latent_norm_threshold:.2f}).")
            break

        # Condition 3: Fixed attribute not preserved (only if a fixed_attribute is specified)
        # Ensure fixed_attribute key exists
        if fixed_discriminator_model and experiment_params.get('fixed_attribute'):
            fixed_attr_name = experiment_params['fixed_attribute']
            stability_threshold = experiment_params['fixed_attr_stability_threshold']

            # Determine the desired state of the fixed attribute based on initial_image_criteria
            # Default to 1 if not explicitly found (e.g., if it was implicitly positive)
            desired_fixed_attr_initial_val = experiment_params['initial_image_criteria'].get(
                fixed_attr_name, 1)

            should_stop = False
            stop_reason = ""

            # We want to keep the attribute in a positive state (e.g., Male=1, Bald=1)
            if desired_fixed_attr_initial_val == 1:
                # Stop if the current score drops *below* the threshold
                if fixed_attr_score_current < stability_threshold:
                    should_stop = True
                    stop_reason = f"Fixed attribute '{fixed_attr_name}' (desired positive) not preserved. Score: {fixed_attr_score_current:.2f} < Threshold: {stability_threshold:.2f}"
            # desired_fixed_attr_initial_val == 0. We want to keep the attribute in a negative state (e.g., Male=0 for female)
            else:
                # Stop if the current score *rises above* the threshold
                if fixed_attr_score_current > stability_threshold:
                    should_stop = True
                    stop_reason = f"Fixed attribute '{fixed_attr_name}' (desired negative) not preserved. Score: {fixed_attr_score_current:.2f} > Threshold: {stability_threshold:.2f}"

            if should_stop:
                print(f"Stopped at step {i+1}: {stop_reason}")
                break

        # Apply the pull-to-center logic and move along the direction vector
        current_z = (1 - pull_strength) * current_z + \
            step_size * direction_vector

        # Decode the new latent vector
        decoded_img = vae_model.decode(current_z)
        generated_images.append(decoded_img[0])

        # Log scores
        log_str = f"Step {i+1}: {experiment_params['attribute_to_change']} Score = {attr_to_change_discriminator_model(current_z).numpy()[0,0]:.2f}"
        if fixed_discriminator_model:
            log_str += f", {experiment_params['fixed_attribute']} Score = {fixed_discriminator_model(current_z).numpy()[0,0]:.2f}"
        log_str += f", Latent Norm = {tf.norm(current_z).numpy():.2f}"
        print(log_str)

    print(
        f"Extrapolation completed. Generated {len(generated_images)} images.")

    # 8. Visualize results
    plot_images(generated_images,
                title=f"{experiment_params['title']} (Epoch {EPOCH}, Beta={pull_strength})",
                filename=f"{experiment_params['output_filename_suffix']}.png",
                output_dir=experiment_output_dir)

    print(f"--- Experiment '{experiment_params['title']}' Complete ---")


if __name__ == "__main__":
    print("--- Initializing Extrapolation Script ---")

    # 1. Load VAE model
    print(f"Loading VAE model from {VAE_MODEL_PATH}...")
    vae_model = model_utils.load_model(VAE_MODEL_PATH, VAE_MODEL_NAME)
    if vae_model is None:
        print("Failed to load VAE model. Exiting.")
        exit()
    print("VAE model loaded successfully.")

    # 2. Load CelebA dataset and attribute names
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

    # Example 7: Female, No Eyeglasses to Female, With Eyeglasses (Corrected for 'Male' attribute)
    female_to_glasses_experiment_corrected = {
        # Start with a female (Male: 0) who does NOT have eyeglasses
        "initial_image_criteria": {"Male": 0, "Eyeglasses": 0},
        "attribute_to_change": "Eyeglasses",
        # Moving from No Eyeglasses (0) to Eyeglasses (1)
        "change_direction_towards_positive": True,
        # Target: 'Eyeglasses' score very positive (has eyeglasses)
        "target_attr_stop_threshold": 40.0,
        "fixed_attribute": "Male",  # Now correctly references 'Male'
        # Stop if 'Male' score *rises above* 0.5 (becomes more male)
        # THIS SHOULD BE POSITIVE, e.g., 0.5, to stop if it becomes male
        "fixed_attr_stability_threshold": 0.5,
        "num_extrapolation_steps": 25,
        "step_size": 0.3,
        "pull_strength": 0.01,
        "output_filename_suffix": "female_to_glasses_corrected",
        "title": "Extrapolation: Female, No Eyeglasses to Female, With Eyeglasses"
    }

    # Don't forget to call this new experiment!
    run_extrapolation_experiment(
        vae_model, celeba_attribute_names, train_ds, female_to_glasses_experiment_corrected)

    print("\n--- All Extrapolation Experiments Complete ---")
