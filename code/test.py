import tensorflow as tf
import os
from preprocess import get_celeba_dataset_with_attributes

# Define your paths
image_directory = '../data/img_align_celeba/img_align_celeba'
attributes_csv_path = '../data/list_attr_celeba.csv'

# Create the dataset
celeba_ds = get_celeba_dataset_with_attributes(
    image_dir=image_directory,
    attr_csv_path=attributes_csv_path,
    image_size=(128, 128),
    batch_size=32,
    shuffle=True
)

# Iterate and verify (optional)
for images, attributes in celeba_ds.take(1):
    print("Batch of images shape:", images.shape)  # (batch_size, 128, 128, 3)
    print("Batch of attributes shape:", attributes.shape)  # (batch_size, 40)
    # You can also inspect the first image and its attributes
    # print("First image attributes:", attributes[0].numpy())
