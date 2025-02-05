# File: scripts/data_preprocessing.py

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths
data_path = r'C:\Users\PranithaReddy\Desktop\mini project\theft detection\datasets'  # Adjust this path based on where your dataset is located
preprocessed_data_path = "../preprocessed_data"  # Path to save preprocessed data if needed

# Parameters
img_height, img_width = 224, 224  # Resize images to this resolution
batch_size = 32

# Function to preprocess an image
def preprocess_image(image_path):
    """
    Loads and preprocesses an image.
    Args:
        image_path (str): Path to the image.
    Returns:
        numpy array: Preprocessed image array.
    """
    # Load image
    image = cv2.imread(image_path)
    # Resize and normalize
    image = cv2.resize(image, (img_height, img_width))
    image = image / 255.0  # Normalize pixel values to [0, 1]
    return image

# Function to load the dataset
def load_dataset():
    """
    Loads and preprocesses the dataset into arrays for training.
    Returns:
        (np.array, np.array): Preprocessed images and corresponding labels.
    """
    images = []
    labels = []

    # Iterate over "normal" and "abnormal" folders
    for label, category in enumerate(['normal', 'abnormal']):
        category_path = os.path.join(data_path, category)
        for file_name in os.listdir(category_path):
            file_path = os.path.join(category_path, file_name)
            # Preprocess the image
            preprocessed_image = preprocess_image(file_path)
            images.append(preprocessed_image)
            labels.append(label)  # 0 for normal, 1 for abnormal

    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    return images, labels

# If needed, split the dataset and save to disk
def split_and_save_data():
    # Ensure the preprocessed data directory exists
    if not os.path.exists(preprocessed_data_path):
        os.makedirs(preprocessed_data_path)

    images, labels = load_dataset()
    # Split into training and validation sets (80% train, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Optionally save preprocessed data
    np.save(os.path.join(preprocessed_data_path, 'X_train.npy'), X_train)
    np.save(os.path.join(preprocessed_data_path, 'y_train.npy'), y_train)
    np.save(os.path.join(preprocessed_data_path, 'X_val.npy'), X_val)
    np.save(os.path.join(preprocessed_data_path, 'y_val.npy'), y_val)

    print("Preprocessed data saved to", preprocessed_data_path)

# Run the script if executed directly
if __name__ == "__main__":
    split_and_save_data()