import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize API
api = KaggleApi()
api.authenticate()

# Download dataset
dataset = 'jessicali9530/celeba-dataset'
api.dataset_download_files(dataset, path='data/', unzip=True)
