# ALT+SHIFT+E - LINE RUN

# IMPORT
import os
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from patchify import patchify
import splitfolders

############################################

# Show data
img_show = cv2.imread("Land-cover_dataset/images/M-33-48-A-c-4-4.tif")
#plt.imshow(img_show)
mask_show = cv2.imread("Land-cover_dataset/masks/M-33-48-A-c-4-4.tif")
#plt.imshow(mask_show[:, :, 2])

# Compare image and mask on one plot
# Do this in function
plt.figure(figsize=(12, 8))
plt.subplot(121)
plt.imshow(img_show)
plt.title('Image')
plt.subplot(122)
plt.imshow(mask_show[:, :, 2], 'gray')
plt.title('Mask')
plt.show()

labels, count_pixels = np.unique(mask_show[:, :, 2], True)
print(labels)  # [0 1 2 3 4] (other, building, woodland, water, road)
print(count_pixels)  # how many pixels

##############################################################

# MAIN PROGRAM
# Make directories
os.makedirs('Land-cover_dataset/256_patches/images/', True)
os.makedirs('Land-cover_dataset/256_patches/masks/', True)

# Crop image into patches of 256x256 and save them into a directory
patch_size = 256


def patch_image(data_types):  # data_types - images or masks
    type_path = os.path.join("Land-cover_dataset", data_types)
    # print(type_path) # Land-cover_dataset\images or Land-cover_dataset\masks
    # path - current directory path,
    # subdir - list of the names of the subdirectories in the current directory (not used here),
    # files - list of the names of the non-directory files in the current directory
    for path, subdir, files in os.walk(type_path):
        # print(path) # Land-cover_dataset/images/
        images = [f for f in files if f.endswith(".tif")]  # All files whose name ends with .tir
        # print(images)
        for image_name in images:
            image_path = os.path.join(type_path, image_name)
            # print(image_path)  # Example Land-cover_dataset/images/M-33-20-D-c-4-2.tif
            image = cv2.imread(image_path, 1)  # 1, because RGB
            # print(image.shape, image_name)
            if image is not None:
                # Calculate the nearest size divisible by patch_size
                image_height = (image.shape[1] // patch_size) * patch_size
                image_width = (image.shape[0] // patch_size) * patch_size
                # Converts the NumPy array to a PIL Image object (good for operations like cropping, resizing)
                image = Image.fromarray(image)
                # Crop the image to the calculated size from the top-left corner
                image = image.crop((0, 0, image_height, image_width))
                # Converts the PIL Image back to a NumPy array
                image = np.array(image)
                patches_img = patchify(image, (patch_size, patch_size, 3), patch_size)  # 3, because channel RGB
                output_dir = os.path.join("Land-cover_dataset", "256_patches", data_types)
                # print(output_dir) # Land-cover_dataset\256_patches\images
                for i in range(patches_img.shape[0]):
                    for j in range(patches_img.shape[1]):
                        single_patch = patches_img[i, j, 0, :, :, :]  # (num_patches_x, num_patches_y, 1, patch_size,
                        # patch_size, num_channels)
                        patch_filename = f"{image_name}_patch_{i}_{j}.tif"
                        patch_filepath = os.path.join(output_dir, patch_filename)
                        cv2.imwrite(patch_filepath, single_patch)


patch_image("images")
patch_image("masks")

# SHOW 256x256
img_show = cv2.imread("Land-cover_dataset/256_patches/images/N-33-104-A-c-1-1.tif_patch_31_1.tif")
mask_show = cv2.imread("Land-cover_dataset/256_patches/masks/N-33-104-A-c-1-1.tif_patch_31_1.tif")

plt.figure(figsize=(12, 8))
plt.subplot(121)
plt.imshow(img_show)
plt.title('Image')
plt.subplot(122)
plt.imshow(mask_show[:, :, 2], 'gray')
plt.title('Mask')
plt.show()

# DON'T HAVE A LOT OF TIME, SPEED
#######################################

# I do this in function

# REMOVE USELESS IMAGES
train_images_dir = "Land-cover_dataset/256_patches/images/"
train_masks_dir = "Land-cover_dataset/256_patches/masks/"

img_list = os.listdir(train_images_dir)
msk_list = os.listdir(train_masks_dir)

os.makedirs('Land-cover_dataset/256_patches/images_with_useful_info/images/', True)
os.makedirs('Land-cover_dataset/256_patches/images_with_useful_info/masks/', True)

useless = 0

# If mask has useless information, remove it
for img in range(len(img_list)):
    img_name = img_list[img]
    mask_name = msk_list[img]

    temp_image = cv2.imread(train_images_dir + img_list[img], 1)  # Read image
    temp_mask = cv2.imread(train_masks_dir + msk_list[img], 0)  # Read mask
    val, counts = np.unique(temp_mask, True)  # Get unique values and their counts in the mask

    if (1 - (counts[0] / counts.sum())) > 0.05:  # Check if more than 5% of the mask is not zero
        cv2.imwrite('Land-cover_dataset/256_patches/images_with_useful_info/images/' + img_name, temp_image)
        cv2.imwrite('Land-cover_dataset/256_patches/images_with_useful_info/masks/' + mask_name, temp_mask)
    else:
        useless += 1

print("Total useful images are: ", len(img_list) - useless)
print("Total useless images are: ", useless)

input_folder = 'Land-cover_dataset/256_patches/images_with_useful_info/'
output_folder = 'Land-cover_dataset/data_for_training_and_testing/'

# SPLIT DATA to TRAINING AND VALIDATION
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.75, .25), group_prefix=None)

# MOVE FILES for PYTORCH
# I move them manually
os.makedirs('Land-cover_dataset/keras_data/train_images/train/', True)
os.makedirs('Land-cover_dataset/keras_data/train_masks/train/', True)
os.makedirs('Land-cover_dataset/keras_data/val_images/val/', True)
os.makedirs('Land-cover_dataset/keras_data/val_masks/val/', True)
# END prepare data
###################################################################################################
