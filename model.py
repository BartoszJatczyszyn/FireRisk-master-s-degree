# ALT+SHIFT+E - LINE RUN

# IMPORT
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
import segmentation_models as sm
from tensorflow.keras.preprocessing.image import ImageDataGenerator

############################################

train_images_dir = "Land-cover_dataset/data_for_training_and_testing/train/images/"
train_masks_dir = "Land-cover_dataset/data_for_training_and_testing/train/masks/"

seed = 24
batch_size = 16
n_classes = 4

preprocess_input = sm.get_preprocessing('resnet34')

# DON'T UNDERSTAND YET


def preprocess_data(img, mask, num_class):
    # Initialize scaler only once to avoid fitting it repeatedly
    scaler = MinMaxScaler()
    # Scale images
    img = img.astype('float32')
    original_shape = img.shape
    img = img.reshape((-1, img.shape[-1]))
    img = scaler.fit_transform(img).reshape(original_shape)
    img = preprocess_input(img)  # Preprocess based on the pretrained backbone
    # Convert mask to one-hot encoding
    mask = mask.astype('float32')
    mask = to_categorical(mask, num_class)
    return img, mask
