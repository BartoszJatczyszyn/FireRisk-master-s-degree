# ALT+SHIFT+E - LINE RUN

# IMPORT
import os
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from patchify import patchify
import splitfolders
import shutil

############################################

# Show data
img = cv2.imread("Land-cover_dataset/images/M-33-48-A-c-4-4.tif")
#plt.imshow(img)
mask = cv2.imread("Land-cover_dataset/masks/M-33-48-A-c-4-4.tif")
#plt.imshow(mask[:, :, 2])


def show_image_and_mask(image_path, mask_path):
    # Read the image and mask
    image_show = cv2.imread(image_path)
    mask_show = cv2.imread(mask_path)

    # Convert the image from BGR to RGB
    img_show_rgb = cv2.cvtColor(image_show, cv2.COLOR_BGR2RGB)

    # Create a figure to display the image and mask side by side
    plt.figure(figsize=(12, 8))

    # Display the image
    plt.subplot(121)
    plt.imshow(img_show_rgb)
    plt.title('Image')

    # Display the mask (assuming the mask is a single channel image)
    plt.subplot(122)
    plt.imshow(mask_show[:, :, 2], cmap='gray')  # Display the third channel of the mask
    plt.title('Mask')

    plt.show()


image_path = "Land-cover_dataset/images/M-33-48-A-c-4-4.tif"
mask_path = "Land-cover_dataset/masks/M-33-48-A-c-4-4.tif"
show_image_and_mask(image_path, mask_path)

labels, count_pixels = np.unique(mask[:, :, 2], True)
print(labels)  # [0 1 2 3 4] (other, building, woodland, water, road)
print(count_pixels)  # how many pixels


def aggregate_pixel_counts(mask_dir):
    # Initialize a dictionary to hold the aggregated pixel counts
    aggregated_counts = {}

    # Loop through each file in the mask directory
    for filename in os.listdir(mask_dir):
        if filename.endswith(".tif"):  # Ensure we are only processing .tif files
            # Read the image
            img_path = os.path.join(mask_dir, filename)
            img = cv2.imread(img_path)

            # Ensure the image was read properly
            if img is not None:
                # Extract the label channel (assuming it's the third channel, index 2)
                labels = img[:, :, 2]

                # Get the unique labels and their counts in the current image
                unique_labels, counts = np.unique(labels, return_counts=True)

                # Aggregate the counts into the aggregated_counts dictionary
                for label, count in zip(unique_labels, counts):
                    if label in aggregated_counts:
                        aggregated_counts[label] += count
                    else:
                        aggregated_counts[label] = count

    return aggregated_counts


def plot_pixel_counts(aggregated_counts):
    # Print the aggregated counts
    for label, count in aggregated_counts.items():
        print(f'Label {label}: {count} pixels')

    # Plot the results
    labels = list(aggregated_counts.keys())
    pixel_counts = list(aggregated_counts.values())

    plt.figure(figsize=(12, 8))
    plt.bar(labels, pixel_counts, color='skyblue')
    plt.xlabel('Label')
    plt.ylabel('Pixel Count')
    plt.title('Pixel Count per Label in Land-cover Dataset')
    plt.xticks(labels)  # Set x-ticks to be the labels
    plt.show()


mask_dir = 'Land-cover_dataset/masks/'
aggregated_counts = aggregate_pixel_counts(mask_dir)
plot_pixel_counts(aggregated_counts)

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
image_path = "Land-cover_dataset/256_patches/images/N-33-104-A-c-1-1.tif_patch_31_1.tif"
mask_path = "Land-cover_dataset/256_patches/masks/N-33-104-A-c-1-1.tif_patch_31_1.tif"
show_image_and_mask(image_path, mask_path)


# DON'T HAVE A LOT OF TIME, SPEED
#######################################

# REMOVE USELESS IMAGES
def filter_and_copy_images_masks(train_images_dir, train_masks_dir):
    # Create directories for filtered images and masks
    os.makedirs('Land-cover_dataset/256_patches/images_with_useful_info/images/', exist_ok=True)
    os.makedirs('Land-cover_dataset/256_patches/images_with_useful_info/masks/', exist_ok=True)
    os.makedirs('Land-cover_dataset/256_patches/images_with_useless_info/images/', exist_ok=True)
    os.makedirs('Land-cover_dataset/256_patches/images_with_useless_info/masks/', exist_ok=True)

    img_list = os.listdir(train_images_dir)
    msk_list = os.listdir(train_masks_dir)
    useless = 0

    # Filter and copy images and masks
    for img in range(len(img_list)):
        img_name = img_list[img]
        mask_name = msk_list[img]

        temp_image = cv2.imread(os.path.join(train_images_dir, img_name), 1)  # Read image
        temp_mask = cv2.imread(os.path.join(train_masks_dir, mask_name), 0)  # Read mask

        if temp_image is None or temp_mask is None:
            print(f"Error reading {img_name} or {mask_name}. Skipping these files.")
            continue

        val, counts = np.unique(temp_mask, return_counts=True)  # Get unique values and their counts in the mask

        if (1 - (counts[0] / counts.sum())) > 0.05:  # Check if more than 5% of the mask is not zero
            cv2.imwrite(os.path.join('Land-cover_dataset/256_patches/images_with_useful_info/images/', img_name),
                        temp_image)
            cv2.imwrite(os.path.join('Land-cover_dataset/256_patches/images_with_useful_info/masks/', mask_name),
                        temp_mask)
        else:
            useless += 1
            if img % 1000 == 0:
                cv2.imwrite(os.path.join('Land-cover_dataset/256_patches/images_with_useless_info/images/', img_name),
                            temp_image)
                cv2.imwrite(os.path.join('Land-cover_dataset/256_patches/images_with_useless_info/masks/', mask_name),
                            temp_mask)

    print("Total useful images are:", len(img_list) - useless)
    print("Total useless images are:", useless)

    # SHOW LABELS AND COUNT_PIXELS AFTER REMOVE USELESS
    aggregated_counts = aggregate_pixel_counts('Land-cover_dataset/256_patches/images_with_useful_info/masks')
    plot_pixel_counts(aggregated_counts)


def split_data(input_folder, output_folder):
    # Split data into training and validation sets
    splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.75, .25), group_prefix=None)


def move_files_for_organize(source_folder, destination_folder):
    # Create destination directories
    os.makedirs(destination_folder, exist_ok=True)
    for subdir in ['train_images/train', 'train_masks/train', 'val_images/val', 'val_masks/val']:
        os.makedirs(os.path.join(destination_folder, subdir), exist_ok=True)

    # Define subdirectories
    subdirs = ['train', 'val']
    for subdir in subdirs:
        # Move images
        src_images = os.path.join(source_folder, subdir, 'images')
        dst_images = os.path.join(destination_folder, f'{subdir}_images', subdir)
        for file_name in os.listdir(src_images):
            shutil.move(os.path.join(src_images, file_name), dst_images)

        # Move masks
        src_masks = os.path.join(source_folder, subdir, 'masks')
        dst_masks = os.path.join(destination_folder, f'{subdir}_masks', subdir)
        for file_name in os.listdir(src_masks):
            shutil.move(os.path.join(src_masks, file_name), dst_masks)

    # Move useless info files
    useless_info_images_src = 'Land-cover_dataset/256_patches/images_with_useless_info/images/'
    useless_info_masks_src = 'Land-cover_dataset/256_patches/images_with_useless_info/masks/'
    useless_info_images_dst = os.path.join(destination_folder, 'useless_info/images/')
    useless_info_masks_dst = os.path.join(destination_folder, 'useless_info/masks/')

    os.makedirs(useless_info_images_dst, exist_ok=True)
    os.makedirs(useless_info_masks_dst, exist_ok=True)

    for file_name in os.listdir(useless_info_images_src):
        shutil.move(os.path.join(useless_info_images_src, file_name), useless_info_images_dst)

    for file_name in os.listdir(useless_info_masks_src):
        shutil.move(os.path.join(useless_info_masks_src, file_name), useless_info_masks_dst)


train_images_dir = "Land-cover_dataset/256_patches/images/"
train_masks_dir = "Land-cover_dataset/256_patches/masks/"
input_folder = 'Land-cover_dataset/256_patches/images_with_useful_info/'
output_folder = 'Land-cover_dataset/data_for_training_and_testing/'
destination_folder = 'Land-cover_dataset/keras_data/'

# Filter and copy images and masks
filter_and_copy_images_masks(train_images_dir, train_masks_dir)

# Split data into training and validation sets
split_data(input_folder, output_folder)

# Move files for organize
move_files_for_organize(output_folder, destination_folder)
