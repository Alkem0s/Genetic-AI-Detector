import numpy as np
import functools
from functools import lru_cache
from PIL import Image

import global_config


@functools.lru_cache(maxsize=16)
def get_edge_detection(image, method='canny', **kwargs):
    """Cached edge detection to avoid redundant computation.
    
    Args:
        image: Input image (numpy array)
        method: 'canny', 'sobel', or 'scharr'
        **kwargs: Parameters for edge detection
    
    Returns:
        Edge detection result
    """
    import cv2
    import numpy as np
    
    if isinstance(image, np.ndarray) and image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    if method == 'canny':
        low = kwargs.get('low', 100)
        high = kwargs.get('high', 200)
        return cv2.Canny(image, low, high)
    elif method == 'sobel':
        dx = kwargs.get('dx', 1)
        dy = kwargs.get('dy', 0)
        ksize = kwargs.get('ksize', 3)
        return cv2.Sobel(image, cv2.CV_64F, dx, dy, ksize=ksize)
    elif method == 'scharr':
        dx = kwargs.get('dx', 1)
        dy = kwargs.get('dy', 0)
        return cv2.Scharr(image, cv2.CV_64F, dx, dy)
    else:
        raise ValueError(f"Unknown edge detection method: {method}")


def generate_dynamic_mask(self, patch_features, rule_set):
    """
    Generate a dynamic mask using genetic algorithm rules.
    
    Args:
        patch_features (np.ndarray): Features for each patch in the image
        
    Returns:
        np.ndarray: Binary patch mask of shape (n_patches_h, n_patches_w)
    """
    
    # Initialize the patch mask
    patch_mask = np.zeros((self.n_patches_h, self.n_patches_w), dtype=np.int8)
    
    # Feature names used in genetic rules
    feature_names = global_config.feature_weights.keys()
    
    # For each patch, apply the rules
    for h in range(min(self.n_patches_h, patch_features.shape[0])):
        for w in range(min(self.n_patches_w, patch_features.shape[1])):
            # Create a dictionary of feature values for this patch
            feature_dict = {}
            for i, feature_name in enumerate(feature_names):
                if i < patch_features.shape[2]:  # Make sure the feature index is valid
                    feature_dict[feature_name] = patch_features[h, w, i]
            
            # Apply each rule to the patch
            include_patch = False
            
            for rule in rule_set:
                feature = rule["feature"]
                if feature not in feature_dict:
                    continue  # Skip rules for unavailable features
                
                value = feature_dict[feature]
                threshold = rule["threshold"]
                operator = rule["operator"]
                action = rule["action"]
                
                # Evaluate the rule
                condition_met = False
                if operator == ">":
                    condition_met = value > threshold
                else:  # operator == "<"
                    condition_met = value < threshold
                
                # If the condition is met, apply the action
                if condition_met and action == 1:
                    include_patch = True
                    break  # One matching rule with action=1 is enough
            
            # Set the patch mask value
            patch_mask[h, w] = 1 if include_patch else 0
    
    return patch_mask


@staticmethod
def remove_black_bars(img, threshold=10):
    """
    Remove black padding bars from an image while maintaining content aspect ratio.
    Handles both PIL Images and numpy arrays.
    
    Args:
        img: Input image (PIL Image or numpy array)
        threshold: Intensity threshold for considering pixels as black (0-255)
    
    Returns:
        Cropped image (same type as input) without black bars
    """
    # Convert to numpy array if PIL Image
    if isinstance(img, Image.Image):
        pil_mode = img.mode
        img_np = np.array(img)
        return_pil = True
    else:
        img_np = img.copy()
        return_pil = False

    # Convert to grayscale for bar detection
    if img_np.ndim == 3:
        gray = np.mean(img_np, axis=2)
    else:
        gray = img_np

    # Find content boundaries
    mask = gray > threshold
    coords = np.argwhere(mask)
    
    if coords.size == 0:  # Entire image is black
        return img
    
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    
    # Crop image
    cropped = img_np[y0:y1, x0:x1]
    
    # Convert back to PIL if original was PIL
    if return_pil:
        return Image.fromarray(cropped, mode=pil_mode)
    return cropped


@functools.lru_cache(maxsize=32)
def cached_fft2(image_hash):
    """Cache for FFT computations"""
    import numpy as np
    # Convert hash back to image (implementation depends on your hashing method)
    image = image_hash_to_array(image_hash)
    return np.fft.fft2(image)

@functools.lru_cache(maxsize=32)
def cached_glcm(image_hash, distances, angles, levels=256, symmetric=True, normed=True):
    """Cache for GLCM computations"""
    from skimage.feature import graycomatrix
    # Convert hash back to image
    image = image_hash_to_array(image_hash)
    return graycomatrix(image, distances, angles, levels, symmetric, normed)

def image_hash_to_array(image_hash):
    """Helper to convert image hash back to numpy array"""
    # Implementation depends on how you hash the image
    # For example, if using tuple hash:
    return np.array(image_hash).reshape((-1, image_hash[1]))

def is_natural_scene(image):
    """Detect if image likely contains natural elements like sky, water, etc."""
    import cv2
    import numpy as np
    
    if len(image.shape) > 2 and image.shape[2] > 1:
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    else:
        # Convert grayscale to RGB then to HSV
        rgb = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    
    h, w = image.shape[:2]
    
    # Check for sky
    top_quarter = hsv[:h//4, :, :]
    hue_top = np.mean(top_quarter[:, :, 0])
    sat_top = np.mean(top_quarter[:, :, 1])
    val_top = np.mean(top_quarter[:, :, 2])
    is_sky_top = (85 < hue_top < 130) and sat_top < 80 and val_top > 150
    
    # Check for water
    bottom_quarter = hsv[3*h//4:, :, :]
    hue_bottom = np.mean(bottom_quarter[:, :, 0])
    sat_bottom = np.mean(bottom_quarter[:, :, 1])
    is_water_bottom = (85 < hue_bottom < 150) and 50 < sat_bottom < 180
    
    # Check for straight line architecture
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) if len(image.shape) > 2 else (image * 255).astype(np.uint8)
    sobelx = get_edge_detection(gray, method='sobel', dx=1, dy=0, ksize=3)
    sobely = get_edge_detection(gray, method='sobel', dx=0, dy=1, ksize=3)
    edges = get_edge_detection(gray, method='canny', low=100, high=200)
    
    edge_ratio = np.sum(edges > 0) / edges.size
    horizontal_edges = np.sum(np.abs(sobelx) > 30) / edges.size
    vertical_edges = np.sum(np.abs(sobely) > 30) / edges.size
    directionality = abs(horizontal_edges - vertical_edges) / max(horizontal_edges + vertical_edges, 0.001)
    has_straight_lines = directionality > 0.4 and edge_ratio > 0.05
    
    return {
        'is_sky': is_sky_top,
        'is_water': is_water_bottom,
        'has_straight_lines': has_straight_lines,
        'is_natural': is_sky_top or is_water_bottom or edge_ratio < 0.02,
        'is_architectural': has_straight_lines and edge_ratio > 0.05
    }

def convert_patch_mask_to_pixel_mask(patch_mask, 
                                   image_shape=None, 
                                   patch_size=None):
    """
    Convert a 2D patch mask to a pixel-level mask.
    Handles both fixed-size patches and variable patches based on image dimensions.
    
    Args:
        patch_mask (np.ndarray): 2D array where 1 indicates selected patches
        image_shape (tuple): Optional (height, width) of original image for variable patches
        patch_size (int or tuple): Optional fixed patch size (h, w) or single int for square
    
    Returns:
        np.ndarray: Pixel-level mask with 1s in selected patch areas
    """
    if len(patch_mask.shape) != 2:
        raise ValueError("Patch mask must be a 2D array")

    # Calculate patch dimensions
    if patch_size is not None:
        # Handle fixed patch size
        if isinstance(patch_size, int):
            patch_h = patch_w = patch_size
        else:
            patch_h, patch_w = patch_size
    elif image_shape is not None:
        # Calculate variable patch size from image dimensions
        img_h, img_w = image_shape[:2]
        patch_h = img_h // patch_mask.shape[0]
        patch_w = img_w // patch_mask.shape[1]
    else:
        raise ValueError("Must provide either image_shape or patch_size")

    # Initialize pixel mask
    if image_shape is not None:
        pixel_mask = np.zeros(image_shape[:2])
    else:
        # Calculate total size from patch dimensions
        total_h = patch_mask.shape[0] * patch_h
        total_w = patch_mask.shape[1] * patch_w
        pixel_mask = np.zeros((total_h, total_w))

    # Populate pixel mask
    for i in range(patch_mask.shape[0]):
        for j in range(patch_mask.shape[1]):
            if patch_mask[i, j] == 1:
                h_start = i * patch_h
                h_end = (i+1) * patch_h
                w_start = j * patch_w
                w_end = (j+1) * patch_w
                
                # Handle edge cases where image dimensions don't divide evenly
                if image_shape is not None:
                    h_end = min(h_end, image_shape[0])
                    w_end = min(w_end, image_shape[1])
                
                pixel_mask[h_start:h_end, w_start:w_end] = 1

    return pixel_mask