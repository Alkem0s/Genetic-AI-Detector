import numpy as np
import functools
from functools import lru_cache


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

@staticmethod
def process_large_image(image, config=None, image_path=None, max_size=1024):
    from feature_extractor import AIFeatureExtractor
    """Process large images using pyramid approach for memory efficiency"""
    import cv2
    import numpy as np
    
    if config is None:
        from ai_detection_config import AIDetectionConfig
        config = AIDetectionConfig()
    
    h, w = image.shape[:2]
    
    # If image is already small enough, process directly
    if max(h, w) <= max_size:
        return AIFeatureExtractor.extract_all_features(image, config, image_path)
    
    # Calculate scaling factor
    scale = max_size / max(h, w)
    small_h, small_w = int(h * scale), int(w * scale)
    
    # Resize image
    small_image = cv2.resize(image, (small_w, small_h))
    
    # Process at smaller scale
    small_map, small_features, scores = AIFeatureExtractor.extract_all_features(
        small_image, config, image_path
    )
    
    # Upscale feature maps to original size
    feature_map = cv2.resize(small_map, (w, h))
    
    # Update scores with scale info
    scores['processing_scale'] = scale
    
    return feature_map, None, scores  # No need to upscale feature stack

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