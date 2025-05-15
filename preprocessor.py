import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def load_and_preprocess_images(image_paths, target_size=(224, 224)):
    """
    Load and preprocess images from a list of file paths.
    
    Args:
        image_paths: List of paths to image files
        target_size: Tuple of (height, width) to resize images to
        
    Returns:
        numpy array of preprocessed images, list of successfully loaded indices
    """
    images = []
    valid_indices = []
    for i, path in enumerate(image_paths):
        try:
            img = load_img(path, target_size=target_size)
            img_array = img_to_array(img)
            img_array = img_array / 255.0  # Normalize pixel values
            images.append(img_array)
            valid_indices.append(i)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # We don't add a placeholder image anymore, just track which ones worked
    
    return np.array(images), valid_indices


def load_data_from_csv(csv_path):
    """
    Load image paths and labels from a CSV file.
    
    Args:
        csv_path: Path to CSV file containing image_path and label columns
        
    Returns:
        Tuple of (image_paths, labels)
    """
    df = pd.read_csv(csv_path)
    
    # Assuming the CSV has columns: 'file_name' and 'label' (0 for human, 1 for AI)
    image_paths = df['file_name'].values
    labels = df['label'].values
    
    # Check if all image paths exist
    missing_files = [path for path in image_paths if not os.path.exists(path)]
    if missing_files:
        print(f"Warning: {len(missing_files)} image files not found")
        print(f"First few missing: {missing_files[:5]}")
    
    return image_paths, labels


def split_data(image_paths, labels, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        image_paths: List of paths to image files
        labels: List of labels corresponding to image_paths
        test_size: Proportion of the dataset to include in the test split
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train_paths, X_test_paths, y_train, y_test)
    """
    return train_test_split(
        image_paths, labels, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=labels
    )


def load_and_split_data(csv_path, test_size=0.2, random_state=42):
    """
    Load data from CSV and split into training and testing sets.
    
    Args:
        csv_path: Path to CSV file containing image_path and label columns
        test_size: Proportion of the dataset to include in the test split
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train_paths, X_test_paths, y_train, y_test)
    """
    # Load data
    image_paths, labels = load_data_from_csv(csv_path)
    
    # Split data
    return split_data(image_paths, labels, test_size, random_state)


def preprocess_dataset(X_train_paths, X_test_paths, target_size=(224, 224)):
    """
    Load and preprocess training and testing images.
    
    Args:
        X_train_paths: List of paths to training images
        X_test_paths: List of paths to testing images
        target_size: Tuple of (height, width) to resize images to
        
    Returns:
        Tuple of (X_train, X_test) preprocessed image arrays, and (y_train_filtered, y_test_filtered)
    """
    print("Loading and preprocessing training images...")
    X_train, train_valid_indices = load_and_preprocess_images(X_train_paths, target_size)
    
    print("Loading and preprocessing test images...")
    X_test, test_valid_indices = load_and_preprocess_images(X_test_paths, target_size)
    
    return X_train, X_test, train_valid_indices, test_valid_indices

def evaluate_model(model, X_test, X_test_features, y_test, config):
    """
    Evaluate model performance and generate classification report.
    
    Args:
        model: Trained model
        X_test: Test images
        X_test_features: Test image features
        y_test: True labels
        
    Returns:
        Dictionary with evaluation metrics
    """
     # Create test dataset input format based on available inputs
    if X_test_features is None:
        # If no feature input is available, only use image input
        test_input = X_test
        # Or if your model expects a dictionary with only 'image_input':
        # test_input = {'image_input': X_test}
    else:
        # If both inputs are available
        test_input = {
            'image_input': X_test,
            'feature_input': X_test_features
        }
    
    # Create dataset based on your model's input requirements
    test_dataset = tf.data.Dataset.from_tensor_slices((test_input, y_test))
    test_dataset = test_dataset.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)

    # Evaluate model
    test_loss, test_acc = model.evaluate(test_dataset)
    
    # Make predictions
    y_pred_prob = model.predict(test_input)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Print classification report
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=['Human', 'AI'], output_dict=True)
    print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Return metrics
    return {
        'accuracy': test_acc,
        'loss': test_loss,
        'classification_report': report,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_prob': y_pred_prob
    }


def load_and_preprocess_single_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess a single image.
    
    Args:
        image_path: Path to image file
        target_size: Tuple of (height, width) to resize image to
        
    Returns:
        Preprocessed image array with batch dimension
    """
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


def save_model_artifacts(model, best_mask, history, metrics, 
                         model_path='ai_detector_model.h5',
                         mask_path='optimized_patch_mask.npy'):
    """
    Save model and related artifacts.
    
    Args:
        model: Trained model
        best_mask: Optimized patch mask
        history: Training history
        metrics: Evaluation metrics
        model_path: Path to save model
        mask_path: Path to save mask
    """
    # Save model
    model.save(model_path)
    
    # Save patch mask
    np.save(mask_path, best_mask)
    
    print(f"Model saved to {model_path}")
    print(f"Optimized patch mask saved to {mask_path}")
    
    return {
        'model_path': model_path,
        'mask_path': mask_path
    }