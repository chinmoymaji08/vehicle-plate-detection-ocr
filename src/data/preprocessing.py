"""
Data preprocessing functions for license plate recognition.
"""

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from models.recognition import create_char_mappings

def load_detection_data(csv_path, image_dir, img_size=(224, 224)):
    """
    Load and preprocess license plate detection dataset.
    
    Args:
        csv_path: Path to CSV file with annotations
        image_dir: Directory containing images
        img_size: Target image size (width, height)
        
    Returns:
        images: Numpy array of preprocessed images
        boxes: Numpy array of normalized bounding boxes [xmin, ymin, xmax, ymax]
        filenames: List of filenames
    """
    # Load CSV file
    df = pd.read_csv(csv_path)
    
    images = []
    boxes = []
    filenames = []
    
    for i, row in df.iterrows():
        img_path = os.path.join(image_dir, row['img_id'])
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
            
        h, w, _ = img.shape
        
        # Normalize coordinates
        xmin = row['xmin'] / w
        ymin = row['ymin'] / h
        xmax = row['xmax'] / w
        ymax = row['ymax'] / h
        
        # Resize image
        img = cv2.resize(img, img_size)
        img = img / 255.0  # Normalize pixel values
        
        images.append(img)
        boxes.append([xmin, ymin, xmax, ymax])
        filenames.append(row['img_id'])
    
    return np.array(images), np.array(boxes), filenames

def load_recognition_data(csv_path, image_dir, img_size=(128, 64), max_length=None):
    """
    Load and preprocess license plate recognition dataset.
    
    Args:
        csv_path: Path to CSV file with annotations
        image_dir: Directory containing images
        img_size: Target image size (width, height)
        max_length: Maximum text length (if None, derive from data)
        
    Returns:
        images: Numpy array of preprocessed images
        labels: Numpy array of padded label sequences
        max_length: Maximum text length
        char_to_idx: Character to index mapping
        filenames: List of filenames
    """
    # Load CSV file
    df = pd.read_csv(csv_path)
    
    # Derive max_length if not provided
    if max_length is None:
        max_length = df['text'].str.len().max()
    
    images = []
    labels = []
    filenames = []
    
    # Create character mappings
    char_to_idx, _ = create_char_mappings()
    
    for i, row in df.iterrows():
        img_path = os.path.join(image_dir, row['img_id'])
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
            
        # Preprocess image
        img = cv2.resize(img, img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255.0  # Normalize
        img = np.expand_dims(img, axis=-1)  # Add channel dimension
        
        # Convert text to sequence of indices
        text = row['text']
        text_indices = [char_to_idx.get(c, 0) for c in text]  # Use 0 for unknown chars
        
        # Pad sequence to max_length
        padded_indices = text_indices + [0] * (max_length - len(text_indices))
        
        images.append(img)
        labels.append(padded_indices)
        filenames.append(row['img_id'])
    
    return np.array(images), np.array(labels), max_length, char_to_idx, filenames

def split_data(images, labels, test_size=0.2, random_state=42):
    """
    Split data into training and validation sets.
    
    Args:
        images: Numpy array of images
        labels: Numpy array of labels
        test_size: Proportion of data to use for validation
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_val, y_train, y_val: Split datasets
    """
    return train_test_split(images, labels, test_size=test_size, random_state=random_state)

def preprocess_single_car_image(img, target_size=(224, 224)):
    """
    Preprocess a single car image for detection.
    
    Args:
        img: Input image (BGR format)
        target_size: Target size for resizing
        
    Returns:
        Preprocessed image ready for the detection model
    """
    # Store original dimensions
    original_h, original_w = img.shape[:2]
    
    # Resize image
    img_resized = cv2.resize(img, target_size)
    
    # Normalize pixel values
    img_normalized = img_resized / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch, (original_h, original_w)

def preprocess_single_plate_image(img, target_size=(128, 64)):
    """
    Preprocess a single license plate image for recognition.
    
    Args:
        img: Input image (BGR format)
        target_size: Target size for resizing
        
    Returns:
        Preprocessed image ready for the recognition model
    """
    # Validate input
    if img is None or img.size == 0:
        raise ValueError("Empty or invalid image passed to preprocess_single_plate_image.")
    
    # Resize image
    img_resized = cv2.resize(img, target_size)
    
    # Convert to grayscale
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # Normalize pixel values
    img_normalized = img_gray / 255.0
    
    # Add channel dimension
    img_expanded = np.expand_dims(img_normalized, axis=-1)
    
    # Add batch dimension
    img_batch = np.expand_dims(img_expanded, axis=0)
    
    return img_batch

def save_processed_data(output_dir, **data):
    """
    Save processed data to disk.
    
    Args:
        output_dir: Directory to save data
        **data: Data to save as keyword arguments
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for name, array in data.items():
        np.save(os.path.join(output_dir, f"{name}.npy"), array)
    
    print(f"Saved processed data to {output_dir}")

def load_processed_data(input_dir, *keys):
    """
    Load processed data from disk.
    
    Args:
        input_dir: Directory containing saved data
        *keys: Names of arrays to load
        
    Returns:
        Dictionary of loaded arrays
    """
    result = {}
    
    for key in keys:
        path = os.path.join(input_dir, f"{key}.npy")
        if os.path.exists(path):
            result[key] = np.load(path, allow_pickle=True)
        else:
            print(f"Warning: Could not find {path}")
    
    return result
