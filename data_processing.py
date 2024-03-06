import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from PIL import Image
import os
import random

# Define paths to your data (modify as needed)
base_data_dir = 'C:\Users\asus\Documents\GitHub\Thesis-Lettuce\thesis image processing\Data for geometry-based greenness index comparisons in lettuce biophysical estimations'  # Replace with actual path
data_labels_xlsx = 'C:\Users\asus\Documents\GitHub\Thesis-Lettuce\thesis image processing\Data for geometry-based greenness index comparisons in lettuce biophysical estimations\CEA_Numerical_Data.xlsx'  # Replace with actual path

# Define training and validation ratios (adjust as needed)
train_ratio = 0.8
val_ratio = 0.2

# Create training and validation directories (optional)
train_data_dir = os.path.join(base_data_dir, 'training')
val_data_dir = os.path.join(base_data_dir, 'validation')
os.makedirs(train_data_dir, exist_ok=True)
os.makedirs(val_data_dir, exist_ok=True)


def load_data(data_dir):
  """
  Loads images and actual heights from a directory.

  Args:
      data_dir: Path to the directory containing image files.

  Returns:
      images: A NumPy array of preprocessed images.
      actual_heights: A list of actual height labels (floats).
  """

  images = []
  actual_heights = []

  for filename in os.listdir(data_dir):
      # Extract image name and extension
      name, ext = os.path.splitext(filename)

      # Skip non-image files
      if ext not in [".jpg", ".png"]:
          continue

      # Load image (assuming RGB format)
      image = np.array(Image.open(os.path.join(data_dir, filename)).convert('RGB'))

      # Check for height information in filename (modify based on your data format)
      try:
          actual_height = float(name.split('_')[-1].replace('cm', ''))  # Assuming format "lettuce_<height>cm.jpg"
      except ValueError:
          # Adjust based on your height data storage (e.g., load from a separate file)
          actual_height = pd.read_excel(data_labels_xlsx, index_col='filename').loc[filename].height  # Example using a separate Excel file

      images.append(image)
      actual_heights.append(actual_height)

  return np.array(images), actual_heights  # Return preprocessed images and actual heights


# Function to split images from a folder into training and validation sets
def split_folder_images(folder_name):
  """
  Splits images from a subfolder into training and validation sets.

  Args:
      folder_name: Name of the subfolder containing images.

  Returns:
      None (images are moved to respective directories).
  """

  image_paths = [os.path.join(base_data_dir, folder_name, image_filename)
                 for image_filename in os.listdir(os.path.join(base_data_dir, folder_name))]
  random.shuffle(image_paths)  # Shuffle images for random split

  num_images = len(image_paths)
  train_split = int(num_images * train_ratio)

  for i, image_path in enumerate(image_paths):
      if i < train_split:
          # Move image to training directory
          new_path = os.path.join(train_data_dir, folder_name, os.path.basename(image_path))
          os.makedirs(os.path.dirname(new_path), exist_ok=True)  # Create subfolders if needed
          os.rename(image_path, new_path)
      else:
          # Move image to validation directory
          new_path = os.path.join(val_data_dir, folder_name, os.path.basename(image_path))
          os.makedirs(os.path.dirname(new_path), exist_ok=True)  # Create subfolders if needed
          os.rename(image_path, new_path)


# Process each folder in base_data_dir (assuming class subfolders)
folder_names = os.listdir(base_data_dir)
for folder_name in folder_names:
  split_folder_images(folder_name)

# (Optional) Create ImageDataGenerators for training and validation sets
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip = True)  # Add data augmentation as needed
val_datagen = ImageDataGenerator(rescale=1./255)

# Use train_datagen and val_datagen in your model training process (replace with your code)
# ... your model building and training code using train_datagen and val_datagen ...
