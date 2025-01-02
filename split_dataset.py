import os
import shutil
from sklearn.model_selection import train_test_split
from collections import Counter

# Paths
data_dir = 'food-101/images'  # Adjusted to point to the 'images' folder
train_dir = 'food-101/train'  # Train destination
test_dir = 'food-101/test'    # Test destination

# Get all class names (subdirectories in images/)
classes = [cls for cls in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, cls))]


# Create train/test directories for each class
for cls in classes:
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

    # Get all image files for the current class
    images = os.listdir(os.path.join(data_dir, cls))
    print(f"{cls}: {len(images)} images")
    images = [os.path.join(data_dir, cls, img) for img in images]

    # Split into 80% train and 20% test
    train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)

    # Copy the images to the respective folders
    for img in train_imgs:
        shutil.copy(img, os.path.join(train_dir, cls))
    for img in test_imgs:
        shutil.copy(img, os.path.join(test_dir, cls))

print("Dataset successfully split into train and test sets.")
