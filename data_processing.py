import tensorflow as tf
from tensorflow.keras.applications import VGG16  # Replace with your pre-trained model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd  # Import pandas for reading XLSX files

# Define paths to your training and validation data folders (images)
train_data_dir = 'path/to/your/training/images'
val_data_dir = 'path/to/your/validation/images'

# Define path to your XLSX file containing labels (replace with actual path)
data_labels_xlsx = 'Data for geometry-based greenness index comparisons in lettuce biophysical estimations/CEA_Numerical_Data.xlsx'

# Create ImageDataGenerator objects for data augmentation and preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

def load_data(data_dir):
    """
    Loads images and actual heights from a directory (assuming height in filenames or separate files).

    Args:
        data_dir: Path to the directory containing image files.

    Returns:
        images: A list of preprocessed images (NumPy arrays).
        actual_heights: A list of actual height labels (NumPy arrays).
        desired_heights: A placeholder list for desired heights (currently ignored).
    """

    images = []
    actual_heights = []
    desired_heights = []

    for filename in os.listdir(data_dir):
        # Extract image name and extension
        name, ext = os.path.splitext(filename)

        # Skip non-image files
        if ext not in [".jpg", ".png"]:
            continue

        # Load image (assuming RGB format)
        image = np.array(Image.open(os.path.join(data_dir, filename)).convert('RGB'))

        # Check for height information in filename (optional)
        try:
            actual_height = float(name.split('_')[-1].replace('cm', ''))  # Assuming format "lettuce_<height>cm.jpg"
        except ValueError:
            # If height not in filename, load from separate file (optional)
            actual_height_path = os.path.join(data_dir, name + "_actual_height.txt")
            try:
                with open(actual_height_path, "r") as f:
                    actual_height = float(f.read())
            except (IOError, ValueError) as e:
                print(f"Error loading actual height for {filename}: {e}")
                actual_height = np.nan  # Use a placeholder value (e.g., NaN) for missing data

        # Desired heights are currently ignored (modify if needed)
        desired_heights.append(np.nan)  # Placeholder, replace with desired height loading if applicable

        images.append(image)
        actual_heights.append(actual_height)

    return images, actual_heights, desired_heights  # Return all three lists for potential future use

# Load data labels from XLSX file
try:
    data_df = pd.read_excel(data_labels_xlsx)  # Read data from XLSX using pandas
    # Assuming image filenames match entries in a specific column (modify as needed)
    image_filenames = data_df['Image Filename'].tolist()  # Extract image filenames
except Exception as e:
    print(f"Error loading data labels: {e}")
    exit(1)  # Exit with an error code if data loading fails

# Process for using the loaded data for training (replace with your model building and training code)
# ... your model building and training code here ...
