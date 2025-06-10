import tensorflow as tf
from preprocess import get_celeba_datasets_with_splits

# Assuming you have these paths
image_directory = '..data/img_align_celeba/img_align_celeba/'
attributes_csv = '../data/list_attr_celeba.csv'
eval_partition_csv = "../data/list_eval_partition.csv"

# Define image size and batch size
img_h, img_w, img_c = 64, 64, 3
image_dims = (img_h, img_w)
batch_size = 32

# Get the datasets
train_ds, val_ds, test_ds = get_celeba_datasets_with_splits(
    image_dir=image_directory,
    attr_csv_path=attributes_csv,
    eval_csv_path=eval_partition_csv,
    image_size=image_dims,
    batch_size=batch_size
)

# Example of iterating through a dataset
print("Training dataset example:")
for image_batch, attr_batch in train_ds.take(1):
    print(f"Image batch shape: {image_batch.shape}")
    print(f"Attribute batch shape: {attr_batch.shape}")
    # Print a small portion of first image
    print(f"First image (normalized): {image_batch[0, :5, :5, 0].numpy()}")
    print(f"First image's attributes: {attr_batch[0].numpy()}")

print("\nValidation dataset example:")
for image_batch, attr_batch in val_ds.take(1):
    print(f"Image batch shape: {image_batch.shape}")
    print(f"Attribute batch shape: {attr_batch.shape}")

print("\nTest dataset example:")
for image_batch, attr_batch in test_ds.take(1):
    print(f"Image batch shape: {image_batch.shape}")
    print(f"Attribute batch shape: {attr_batch.shape}")
