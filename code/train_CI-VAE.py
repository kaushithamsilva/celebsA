import tensorflow as tf
import pandas as pd
import numpy as np
import os
import datetime

# Assuming model_utils.py is in the same directory or accessible in PYTHONPATH
# It should contain a function like `save_model`
import model_utils  # You'll need to ensure model_utils.py is correctly set up

# Assuming preprocess.py is in the same directory or accessible in PYTHONPATH
from preprocess import get_celeba_datasets_with_splits

# Assuming vae.py is in the same directory or accessible in PYTHONPATH
from vae import VAE, Sampling

# This is a placeholder for your GPU initialization script if you have one
# For standard TF setup, this might not be strictly necessary if TF automatically finds GPUs
# import init_gpu
# init_gpu.initialize_gpus()

# --- Configuration Constants ---
SAVE_PATH = '../../models/ci_vae/'
CHECKPOINT_PATH = os.path.join(SAVE_PATH, 'checkpoints/')
DISCRIMINATOR_SAVE_PATH = os.path.join(SAVE_PATH, 'discriminators/')
EPOCH_CHECKPOINT_INTERVAL = 50  # Save checkpoints every 50 epochs
KL_WEIGHT = 0.01  # Weight for KL divergence loss
# Define a weight for the classification loss. This is crucial for balancing objectives.
# You might need to tune this. Start with something small.
CLASSIFICATION_LOSS_WEIGHT = 1.0


# Ensure save directories exist
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
os.makedirs(DISCRIMINATOR_SAVE_PATH, exist_ok=True)

# --- Loss Functions ---

# Use BinaryCrossentropy for binary attributes, from_logits=True because Dense layer has no activation
attribute_classification_loss_fn = tf.keras.losses.BinaryCrossentropy(
    from_logits=True, reduction=tf.keras.losses.Reduction.NONE
)

# --- Model Definition Helpers ---


def linear_discriminator(input_dim):
    """
    Creates a single linear discriminator for a binary attribute.
    Output is a single logit.
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation=None, input_shape=(input_dim,))
    ])

# --- Training Step ---


@tf.function
def train_step_ci(vae_model, discriminators, x, y, optimizer):
    """
    Performs one training step for the CI-VAE.

    Args:
        vae_model (VAE): The VAE model.
        discriminators (list): A list of 40 linear discriminator models.
        x (tf.Tensor): Batch of input images.
        y (tf.Tensor): Batch of attribute labels (shape: [batch_size, 40]).
        optimizer (tf.keras.optimizers.Optimizer): The optimizer for training.

    Returns:
        dict: Dictionary of loss values for the current step.
    """
    with tf.GradientTape() as tape:
        # 1. VAE forward pass
        reconstructed, z_mean, z_log_var = vae_model(
            x, training=True)  # Ensure training=True for BN

        # 2. VAE Reconstruction Loss (MSE)
        reconstruction_loss = tf.reduce_mean(tf.square(x - reconstructed))

        # 3. VAE KL Divergence Loss
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )

        # Get the latent representation z (using z_mean for discriminator input)
        z = z_mean

        # 4. Attribute Classification Losses
        attribute_classification_losses = []
        for i, discriminator in enumerate(discriminators):
            # y[:, i] gets the labels for the i-th attribute across the batch
            # Reshape labels to [batch_size, 1] for BinaryCrossentropy
            attribute_labels = tf.expand_dims(y[:, i], axis=-1)

            # Get logits from the current discriminator
            # Ensure training=True for BN if discriminators used BN
            attribute_preds = discriminator(z, training=True)

            # Calculate binary cross-entropy loss for this attribute
            # We use reduction=NONE then mean to ensure correct averaging over batch
            loss_i = attribute_classification_loss_fn(
                attribute_labels, attribute_preds)
            attribute_classification_losses.append(tf.reduce_mean(
                loss_i))  # Average over batch for this attribute

        # Sum all attribute classification losses
        total_attribute_classification_loss = tf.add_n(
            attribute_classification_losses)

        # 5. Total Loss
        total_loss = (reconstruction_loss +
                      KL_WEIGHT * kl_loss +
                      CLASSIFICATION_LOSS_WEIGHT * total_attribute_classification_loss)

    # 6. Compute gradients
    # Combine trainable variables from VAE and all discriminators
    trainable_variables = vae_model.trainable_variables
    for disc in discriminators:
        trainable_variables.extend(disc.trainable_variables)

    grads = tape.gradient(total_loss, trainable_variables)

    # 7. Apply gradients
    optimizer.apply_gradients(zip(grads, trainable_variables))

    return {
        "total_loss": total_loss,
        "reconstruction_loss": reconstruction_loss,
        "kl_loss": kl_loss,
        "attribute_classification_loss": total_attribute_classification_loss,
        # You might want to return individual attribute losses for monitoring too
        # e.g., "attr_0_loss": attribute_classification_losses[0]
    }

# --- Training Loop ---


def train_ci_vae(vae_model, discriminators, train_dataset, val_dataset, optimizer, epochs, attribute_names=None):
    """
    Trains the CI-VAE model with multiple attribute discriminators.

    Args:
        vae_model (VAE): The VAE model.
        discriminators (list): A list of 40 linear discriminator models.
        train_dataset (tf.data.Dataset): The training dataset.
        val_dataset (tf.data.Dataset): The validation dataset.
        optimizer (tf.keras.optimizers.Optimizer): The optimizer.
        epochs (int): Number of training epochs.
        attribute_names (list, optional): List of attribute names for saving.
    """
    if attribute_names is None:
        attribute_names = [
            f"attribute_{i}" for i in range(len(discriminators))]

    print(f"Starting CI-VAE training for {epochs} epochs...")
    print(f"KL_WEIGHT: {KL_WEIGHT}")
    print(f"CLASSIFICATION_LOSS_WEIGHT: {CLASSIFICATION_LOSS_WEIGHT}")
    print("-" * 50)

    for epoch in range(epochs):
        start_time = datetime.datetime.now()

        # Metrics for training
        train_metrics = {loss: tf.keras.metrics.Mean()
                         for loss in ["total_loss", "reconstruction_loss", "kl_loss", "attribute_classification_loss"]}

        # Metrics for validation
        val_metrics = {loss: tf.keras.metrics.Mean()
                       for loss in ["total_loss", "reconstruction_loss", "kl_loss", "attribute_classification_loss"]}

        # --- Training Phase ---
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            train_losses = train_step_ci(
                vae_model, discriminators, x_batch_train, y_batch_train, optimizer
            )
            for key, value in train_losses.items():
                train_metrics[key].update_state(value)

        # --- Validation Phase ---
        for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            # No gradient calculation for validation
            reconstructed, z_mean, z_log_var = vae_model(
                x_batch_val, training=False)
            reconstruction_loss_val = tf.reduce_mean(
                tf.square(x_batch_val - reconstructed))
            kl_loss_val = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )
            z_val = z_mean

            attribute_classification_losses_val = []
            for i, discriminator in enumerate(discriminators):
                attribute_labels_val = tf.expand_dims(
                    y_batch_val[:, i], axis=-1)
                attribute_preds_val = discriminator(z_val, training=False)
                loss_i_val = attribute_classification_loss_fn(
                    attribute_labels_val, attribute_preds_val)
                attribute_classification_losses_val.append(
                    tf.reduce_mean(loss_i_val))

            total_attribute_classification_loss_val = tf.add_n(
                attribute_classification_losses_val)

            total_loss_val = (reconstruction_loss_val +
                              KL_WEIGHT * kl_loss_val +
                              CLASSIFICATION_LOSS_WEIGHT * total_attribute_classification_loss_val)

            val_metrics["total_loss"].update_state(total_loss_val)
            val_metrics["reconstruction_loss"].update_state(
                reconstruction_loss_val)
            val_metrics["kl_loss"].update_state(kl_loss_val)
            val_metrics["attribute_classification_loss"].update_state(
                total_attribute_classification_loss_val)

        end_time = datetime.datetime.now()
        epoch_duration = end_time - start_time

        # --- Print Results ---
        print(
            f"Epoch {epoch+1}/{epochs} ({epoch_duration.total_seconds():.2f}s):")
        print("  Train Metrics:")
        for key in train_metrics:
            print(f"    {key}: {train_metrics[key].result().numpy():.4f}")
        print("  Validation Metrics:")
        for key in val_metrics:
            print(f"    {key}: {val_metrics[key].result().numpy():.4f}")
        print("-" * 50)

        # --- Checkpoint Saving ---
        if (epoch + 1) % EPOCH_CHECKPOINT_INTERVAL == 0:
            print(f"Saving checkpoint at epoch {epoch+1}...")
            model_utils.save_model(
                vae_model, CHECKPOINT_PATH, f'vae-e{epoch+1}')
            for i, disc in enumerate(discriminators):
                disc_name = f"{attribute_names[i].replace(' ', '_')}_discriminator-e{epoch+1}"
                # Save to checkpoint for recovery
                model_utils.save_model(disc, CHECKPOINT_PATH, disc_name)

    # --- Final Save ---
    print("\nTraining complete. Saving final models...")
    model_utils.save_model(vae_model, SAVE_PATH, f'vae-final')
    for i, disc in enumerate(discriminators):
        disc_name = f"{attribute_names[i].replace(' ', '_')}_discriminator.keras"
        # Save to the dedicated discriminator folder
        model_utils.save_model(disc, DISCRIMINATOR_SAVE_PATH, disc_name)
    print("Models saved.")


# --- Main Execution Block ---
if __name__ == '__main__':
    import init_gpu
    init_gpu.initialize_gpus()

    # --- Data Loading ---
    # Set your dataset paths correctly
    # Replace with your actual paths
    # Directory containing images
    IMAGE_DIR = '../data/img_align_celeba/img_align_celeba/'
    ATTRIBUTES_CSV = '../data/list_attr_celeba.csv'
    EVAL_PARTITION_CSV = '../data/list_eval_partition.csv'

    # Get CelebA attribute names (optional, but good for saving)
    # You'll need to load the CSV just to get the column names
    attr_df_for_names = pd.read_csv(ATTRIBUTES_CSV, index_col=0)
    celeba_attribute_names = attr_df_for_names.columns.tolist()
    NUM_ATTRIBUTES = len(celeba_attribute_names)  # Should be 40

    # Define image size and batch size
    INPUT_IMAGE_SIZE = (64, 64)  # Height, Width
    INPUT_IMAGE_CHANNELS = 3
    BATCH_SIZE = 256

    print("Loading CelebA datasets...")
    train_ds, val_ds, test_ds = get_celeba_datasets_with_splits(
        image_dir=IMAGE_DIR,
        attr_csv_path=ATTRIBUTES_CSV,
        eval_csv_path=EVAL_PARTITION_CSV,
        image_size=INPUT_IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )
    print("Datasets loaded.")

    # --- Model Initialization ---
    input_dim = (INPUT_IMAGE_SIZE[0],
                 INPUT_IMAGE_SIZE[1], INPUT_IMAGE_CHANNELS)
    latent_dim = 128  # Size of the latent space
    hidden_dim = 64  # Base number of filters for VAE convolutional layers

    print("Initializing VAE model...")
    vae_model = VAE(input_dim, latent_dim, hidden_dim)
    # Build VAE model to get shapes correct before creating discriminators
    # Take one batch from the training dataset to infer shapes for the VAE encoder.
    for x_batch, _ in train_ds.take(1):
        _ = vae_model(x_batch)
        break
    vae_model.summary()
    print("VAE model initialized.")

    print(f"Initializing {NUM_ATTRIBUTES} linear discriminators...")
    # Create 40 discriminators
    discriminators = [linear_discriminator(
        latent_dim) for _ in range(NUM_ATTRIBUTES)]
    print("Discriminators initialized.")

    # --- Optimizer Initialization ---
    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4)

    # --- Training ---
    epochs = 1000
    train_ci_vae(vae_model, discriminators,
                 train_ds, val_ds, optimizer, epochs=epochs,
                 attribute_names=celeba_attribute_names)

    # Note: Final models are saved inside train_ci_vae function after completion.
