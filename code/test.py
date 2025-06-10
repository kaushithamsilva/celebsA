from preprocess import get_celeba_dataset
import tensorflow as tf

dataset = get_celeba_dataset('../data/img_align_celeba/img_align_celeba')

for batch in dataset.take(1):
    print(batch.shape)  # Should print (batch_size, 128, 128, 3)
