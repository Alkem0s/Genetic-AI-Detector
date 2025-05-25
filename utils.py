import tensorflow as tf
import functools
from functools import lru_cache

# Assuming global_config is accessible and defines feature_weights
import global_config 


@functools.lru_cache(maxsize=16)
def get_edge_detection(image, method='canny', **kwargs):
    """Cached edge detection to avoid redundant computation.
    
    Args:
        image: Input image (tf.Tensor)
        method: 'canny', 'sobel', or 'scharr'
        **kwargs: Parameters for edge detection
    
    Returns:
        Edge detection result (tf.Tensor)
    """
    if image.dtype != tf.uint8:
        image = tf.cast(image * 255, tf.uint8)

    if method == 'canny':
        # TensorFlow does not have a direct Canny implementation.
        # This is a placeholder or requires a custom Canny implementation using TF ops.
        # For a true conversion, you'd implement Canny's steps (grayscale, Gaussian blur, gradients, non-max suppression, hysteresis)
        # using tf.nn.conv2d, tf.math.atan2, etc.
        # For now, let's return a basic edge detection using Sobel as a conceptual replacement if Canny is not directly available.
        # In a real-world scenario, you might use a pre-trained model or a more complex TF implementation of Canny.
        
        # Approximate Canny with Sobel magnitude for demonstration
        sobel_x = tf.image.sobel_edges(tf.expand_dims(tf.cast(image, tf.float32), axis=-1))[:,:,:,0]
        sobel_y = tf.image.sobel_edges(tf.expand_dims(tf.cast(image, tf.float32), axis=-1))[:,:,:,1]
        edges = tf.sqrt(sobel_x**2 + sobel_y**2)
        
        low = kwargs.get('low', 100)
        high = kwargs.get('high', 200)
        
        # Simple thresholding to simulate Canny's output
        edges = tf.cast(edges > tf.cast(low, tf.float32), tf.uint8) * 255
        return edges
    elif method == 'sobel':
        # tf.image.sobel_edges expects a 4D tensor [batch, height, width, channels]
        # and returns a tensor of shape [batch, height, width, channels, 2]
        # where the last dimension contains the x and y gradients.
        
        # Ensure image has a channel dimension
        if len(image.shape) == 2: # Grayscale
            image = tf.expand_dims(image, axis=-1)
        elif len(image.shape) == 3 and image.shape[-1] == 3: # RGB
            image = tf.image.rgb_to_grayscale(image) # Sobel usually on grayscale
            
        image = tf.cast(image, tf.float32)
        sobel_combined = tf.image.sobel_edges(tf.expand_dims(image, axis=0)) # Add batch dimension
        
        dx = kwargs.get('dx', 1)
        dy = kwargs.get('dy', 0)

        if dx == 1 and dy == 0:
            result = sobel_combined[0, :, :, 0, 0] # x-gradient for the first (and only) channel
        elif dx == 0 and dy == 1:
            result = sobel_combined[0, :, :, 0, 1] # y-gradient for the first (and only) channel
        else:
            raise ValueError("Sobel method in TensorFlow only supports dx=1, dy=0 or dx=0, dy=1 directly via tf.image.sobel_edges for single direction.")
        
        return result
    elif method == 'scharr':
        # TensorFlow does not have a direct Scharr implementation.
        # Scharr is a type of Sobel operator with different kernel values.
        # You would need to implement the Scharr kernel as a custom convolution.
        
        # Scharr kernels:
        # G_x = [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]
        # G_y = [[-3, -10, -3], [0, 0, 0], [3, 10, 3]]

        if len(image.shape) == 2: # Grayscale
            image = tf.expand_dims(tf.expand_dims(tf.cast(image, tf.float32), axis=-1), axis=0) # Add batch and channel
        elif len(image.shape) == 3 and image.shape[-1] == 3: # RGB
            image = tf.expand_dims(tf.image.rgb_to_grayscale(tf.cast(image, tf.float32)), axis=0) # Add batch, convert to gray

        kernel_x = tf.constant([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]], dtype=tf.float32)
        kernel_x = tf.expand_dims(tf.expand_dims(kernel_x, axis=-1), axis=-1) # Add in_channels, out_channels

        kernel_y = tf.constant([[-3., -10., -3.], [0., 0., 0.], [3., 10., 3.]], dtype=tf.float32)
        kernel_y = tf.expand_dims(tf.expand_dims(kernel_y, axis=-1), axis=-1)

        dx = kwargs.get('dx', 1)
        dy = kwargs.get('dy', 0)

        if dx == 1 and dy == 0:
            result = tf.nn.conv2d(image, kernel_x, strides=[1, 1, 1, 1], padding='SAME')[0,:,:,0]
        elif dx == 0 and dy == 1:
            result = tf.nn.conv2d(image, kernel_y, strides=[1, 1, 1, 1], padding='SAME')[0,:,:,0]
        else:
            raise ValueError("Scharr method in TensorFlow only supports dx=1, dy=0 or dx=0, dy=1 for now.")
        
        return result
    else:
        raise ValueError(f"Unknown edge detection method: {method}")


def generate_dynamic_mask(patch_features, n_patches_h, n_patches_w, rule_set):
    """
    Generate a dynamic mask using genetic algorithm rules.
    
    Args:
        patch_features (tf.Tensor): Features for each patch in the image
        n_patches_h (int): Number of patches in height
        n_patches_w (int): Number of patches in width
        rule_set (list): List of rules to apply, each rule is a dict with keys:
                             - 'feature': feature name
                             - 'threshold': threshold value
                             - 'operator': comparison operator ('>' or '<')
                             - 'action': 1 to include patch, 0 to exclude patch
    Returns:
        tf.Tensor: Binary patch mask of shape (n_patches_h, n_patches_w)
    """
    
    # Initialize the patch mask
    patch_mask = tf.zeros((n_patches_h, n_patches_w), dtype=tf.int8)
    
    # Feature names used in genetic rules
    feature_names = global_config.feature_weights.keys()
    # Use tf.TensorArray for dynamic writes in a loop, then convert to tf.Tensor
    mask_list = tf.TensorArray(tf.int8, size=n_patches_h * n_patches_w, dynamic_size=False)

    idx = tf.constant(0)

    # For each patch, apply the rules
    # TensorFlow's tf.range and tf.while_loop are preferred for graph mode compatibility
    h_limit = tf.minimum(n_patches_h, tf.shape(patch_features)[0])
    w_limit = tf.minimum(n_patches_w, tf.shape(patch_features)[1])

    # Loop over height
    for h in tf.range(h_limit):
        # Loop over width
        for w in tf.range(w_limit):
            current_patch_features = patch_features[h, w, :]
            
            # This part needs to be handled carefully as TensorFlow doesn't have direct dict-like access for dynamic feature names easily in graph mode.
            # We'll map feature names to indices, assuming a consistent order.
            include_patch = tf.constant(False)
            
            for rule in rule_set:
                feature_name = rule["feature"]
                threshold = rule["threshold"]
                operator = rule["operator"]
                action = rule["action"]
                
                # Find the index of the feature_name
                try:
                    feature_idx = list(feature_names).index(feature_name)
                except ValueError:
                    continue # Skip rule for unavailable feature
                
                if feature_idx >= tf.shape(current_patch_features)[0]:
                    continue # Skip if feature index is out of bounds for this patch's features

                value = current_patch_features[feature_idx]
                
                # Evaluate the rule
                condition_met = tf.constant(False)
                if operator == ">":
                    condition_met = value > threshold
                else:  # operator == "<"
                    condition_met = value < threshold
                
                # If the condition is met and action is 1, include the patch
                if condition_met and action == 1:
                    include_patch = tf.constant(True)
                    break # One matching rule with action=1 is enough
            
            mask_val = tf.cond(include_patch, lambda: tf.constant(1, dtype=tf.int8), lambda: tf.constant(0, dtype=tf.int8))
            mask_list = mask_list.write(idx, mask_val)
            idx += 1

    # Convert TensorArray back to Tensor and reshape
    patch_mask = tf.reshape(mask_list.stack(), (h_limit, w_limit))
    
    return patch_mask

@tf.function
def remove_black_bars(img, threshold=10):
    """
    Remove black padding bars from an image while maintaining content aspect ratio.
    Handles TensorFlow Tensors.
    
    Args:
        img: Input image (tf.Tensor)
        threshold: Intensity threshold for considering pixels as black (0-255)
    
    Returns:
        Cropped image (tf.Tensor) without black bars
    """
    # Ensure image is float for calculations, then convert to uint8 for consistency if needed
    original_dtype = img.dtype
    img_float = tf.cast(img, tf.float32)

    # Convert to grayscale for bar detection
    if len(img_float.shape) == 3 and img_float.shape[-1] == 3: # RGB
        gray = tf.image.rgb_to_grayscale(img_float)
    else: # Grayscale or already processed
        gray = img_float

    # Squeeze the channel dimension for 2D mask operations
    gray = tf.squeeze(gray)

    # Find content boundaries
    mask = gray > tf.cast(threshold, tf.float32)
    
    # Check if the entire image is black
    if tf.reduce_all(tf.logical_not(mask)):
        return img

    # Find the indices of True values (non-black pixels)
    coords = tf.where(mask)
    
    # Get min/max coordinates
    y0 = tf.reduce_min(coords[:, 0])
    x0 = tf.reduce_min(coords[:, 1])
    y1 = tf.reduce_max(coords[:, 0]) + 1
    x1 = tf.reduce_max(coords[:, 1]) + 1
    
    # Crop image using tf.slice
    cropped = tf.slice(img, [y0, x0, 0] if len(img.shape) == 3 else [y0, x0], 
                             [y1 - y0, x1 - x0, img.shape[-1]] if len(img.shape) == 3 else [y1 - y0, x1 - x0])
    
    return tf.cast(cropped, original_dtype)


@functools.lru_cache(maxsize=32)
def cached_fft2(image_hash):
    """Cache for FFT computations"""
    # Convert hash back to image (implementation depends on your hashing method)
    # This part remains conceptually the same, as image_hash_to_array is a helper.
    image = image_hash_to_array(image_hash) 
    
    # tf.signal.fft2d expects complex input, or float that it converts to complex
    # Ensure image is float32 for FFT
    image = tf.cast(image, tf.float32)
    return tf.signal.fft2d(tf.cast(image, tf.complex64))

@functools.lru_cache(maxsize=32)
def cached_glcm(image_hash, distances, angles, levels=256, symmetric=True, normed=True):
    """Cache for GLCM computations"""
    # TensorFlow does not have a direct `skimage.feature.graycomatrix` equivalent.
    # GLCM computation is complex and would require a custom TensorFlow implementation
    # involving binning pixels, calculating co-occurrence matrices for each distance/angle, etc.
    # For a direct conversion, this function would either need to be removed,
    # or a dedicated TensorFlow-based GLCM library/custom op would be required.
    
    # Placeholder for GLCM:
    # If `graycomatrix` is critical and must be pure TF, you'd need a custom implementation.
    # For now, raising an error to indicate it's not directly convertible.
    raise NotImplementedError("GLCM computation (skimage.feature.graycomatrix) has no direct TensorFlow equivalent and requires a custom implementation.")

def image_hash_to_array(image_hash):
    """Helper to convert image hash back to tensorflow array"""
    # This implementation depends on how you hash the image.
    # If the hash is designed to be directly convertible to a TensorFlow tensor shape,
    # you would reconstruct the tensor here.
    # Example (assuming hash is a tuple of (flattened_array, height, width)):
    if isinstance(image_hash, tuple) and len(image_hash) == 3:
        flattened_array, h, w = image_hash
        return tf.reshape(tf.constant(flattened_array, dtype=tf.float32), (h, w))
    else:
        # Fallback if hash format is different or not directly reversible
        raise ValueError("Unsupported image hash format for conversion to TensorFlow tensor.")

@tf.function
def is_natural_scene(image):
    """Detect if image likely contains natural elements like sky, water, etc."""
    # Ensure image is float [0,1] or uint8 [0,255] for color conversions.
    # Assuming input 'image' is already normalized to [0, 1] or is uint8.
    # Convert to uint8 for operations that expect 0-255 range.
    image_uint8 = tf.cast(image * 255, tf.uint8) if image.dtype != tf.uint8 else image
    image_float = tf.cast(image, tf.float32) # Keep a float version for other calcs

    h, w = tf.shape(image)[0], tf.shape(image)[1]
    
    if len(image.shape) > 2 and image.shape[2] > 1: # RGB image
        hsv = tf.image.rgb_to_hsv(image_float)
    else: # Grayscale image
        # Convert grayscale to RGB then to HSV
        rgb = tf.image.grayscale_to_rgb(tf.expand_dims(image_float, axis=-1))
        hsv = tf.image.rgb_to_hsv(rgb)
    
    # Check for sky
    top_quarter = hsv[:h//4, :, :]
    hue_top = tf.reduce_mean(top_quarter[:, :, 0]) * 360 # Scale hue to 0-360
    sat_top = tf.reduce_mean(top_quarter[:, :, 1]) * 255 # Scale sat to 0-255
    val_top = tf.reduce_mean(top_quarter[:, :, 2]) * 255 # Scale val to 0-255
    is_sky_top = tf.logical_and(tf.logical_and(hue_top > 85, hue_top < 130),
                                tf.logical_and(sat_top < 80, val_top > 150))
    
    # Check for water
    bottom_quarter = hsv[3*h//4:, :, :]
    hue_bottom = tf.reduce_mean(bottom_quarter[:, :, 0]) * 360
    sat_bottom = tf.reduce_mean(bottom_quarter[:, :, 1]) * 255
    is_water_bottom = tf.logical_and(tf.logical_and(hue_bottom > 85, hue_bottom < 150),
                                     tf.logical_and(sat_bottom > 50, sat_bottom < 180))
    
    # Check for straight line architecture
    if len(image.shape) > 2 and image.shape[2] > 1:
        gray = tf.image.rgb_to_grayscale(image_float)
    else:
        gray = tf.expand_dims(image_float, axis=-1) # Add channel dimension for consistency
    
    # get_edge_detection function returns 2D tensor, so need to squeeze channel after sobel_edges
    sobel_x_edges = get_edge_detection(tf.squeeze(gray), method='sobel', dx=1, dy=0)
    sobely_edges = get_edge_detection(tf.squeeze(gray), method='sobel', dx=0, dy=1)
    
    # Canny approximation from get_edge_detection
    edges = get_edge_detection(tf.squeeze(gray), method='canny', low=100, high=200) 
    
    edge_ratio = tf.reduce_sum(tf.cast(edges > 0, tf.float32)) / tf.cast(tf.size(edges), tf.float32)
    horizontal_edges = tf.reduce_sum(tf.cast(tf.abs(sobel_x_edges) > 30, tf.float32)) / tf.cast(tf.size(edges), tf.float32)
    vertical_edges = tf.reduce_sum(tf.cast(tf.abs(sobely_edges) > 30, tf.float32)) / tf.cast(tf.size(edges), tf.float32)
    
    # Add a small epsilon to avoid division by zero
    directionality = tf.abs(horizontal_edges - vertical_edges) / tf.maximum(horizontal_edges + vertical_edges, 1e-6)
    has_straight_lines = tf.logical_and(directionality > 0.4, edge_ratio > 0.05)
    
    return {
        'is_sky': is_sky_top,
        'is_water': is_water_bottom,
        'has_straight_lines': has_straight_lines,
        'is_natural': tf.logical_or(tf.logical_or(is_sky_top, is_water_bottom), edge_ratio < 0.02),
        'is_architectural': tf.logical_and(has_straight_lines, edge_ratio > 0.05)
    }

@tf.function
def convert_patch_mask_to_pixel_mask(patch_mask, 
                                   image_shape=None, 
                                   patch_size=None):
    """
    Convert a 2D patch mask to a pixel-level mask.
    Handles both fixed-size patches and variable patches based on image dimensions.
    
    Args:
        patch_mask (tf.Tensor): 2D array where 1 indicates selected patches
        image_shape (tuple/tf.TensorShape): Optional (height, width) of original image for variable patches
        patch_size (int or tuple): Optional fixed patch size (h, w) or single int for square
    
    Returns:
        tf.Tensor: Pixel-level mask with 1s in selected patch areas
    """
    if len(patch_mask.shape) != 2:
        raise ValueError("Patch mask must be a 2D array")

    # Calculate patch dimensions
    if patch_size is not None:
        # Handle fixed patch size
        if isinstance(patch_size, int):
            patch_h = tf.constant(patch_size, dtype=tf.int32)
            patch_w = tf.constant(patch_size, dtype=tf.int32)
        else:
            patch_h = tf.constant(patch_size[0], dtype=tf.int32)
            patch_w = tf.constant(patch_size[1], dtype=tf.int32)
    elif image_shape is not None:
        # Calculate variable patch size from image dimensions
        # Ensure image_shape is a tf.Tensor or can be converted
        img_h, img_w = tf.cast(image_shape[0], tf.int32), tf.cast(image_shape[1], tf.int32)
        patch_h = img_h // tf.cast(patch_mask.shape[0], tf.int32)
        patch_w = img_w // tf.cast(patch_mask.shape[1], tf.int32)
    else:
        raise ValueError("Must provide either image_shape or patch_size")

    # Initialize pixel mask
    if image_shape is not None:
        pixel_mask = tf.zeros(tf.cast(image_shape[:2], tf.int32), dtype=tf.float32)
    else:
        # Calculate total size from patch dimensions
        total_h = tf.cast(patch_mask.shape[0], tf.int32) * patch_h
        total_w = tf.cast(patch_mask.shape[1], tf.int32) * patch_w
        pixel_mask = tf.zeros((total_h, total_w), dtype=tf.float32)

    # Populate pixel mask using tf.where and tf.tensor_scatter_nd_update for efficiency
    # Create a grid of patch indices
    patch_indices_h, patch_indices_w = tf.where(patch_mask == 1)[:, 0], tf.where(patch_mask == 1)[:, 1]

    # Calculate start and end pixel coordinates for each selected patch
    h_starts = patch_indices_h * patch_h
    w_starts = patch_indices_w * patch_w
    h_ends = (patch_indices_h + 1) * patch_h
    w_ends = (patch_indices_w + 1) * patch_w

    if image_shape is not None:
        img_h, img_w = tf.cast(image_shape[0], tf.int32), tf.cast(image_shape[1], tf.int32)
        h_ends = tf.minimum(h_ends, img_h)
        w_ends = tf.minimum(w_ends, img_w)

    # Use a loop with tf.tensor_scatter_nd_update or equivalent for sparse updates
    # A more efficient way would be to create a large tensor of ones and then slice and add.
    # For simplicity and direct mapping of the original loop logic (though less efficient in TF):
    
    # Create an empty pixel mask to build upon
    if image_shape is not None:
        final_pixel_mask = tf.zeros(tf.cast(image_shape[:2], tf.int32), dtype=tf.float32)
    else:
        total_h = tf.cast(patch_mask.shape[0], tf.int32) * patch_h
        total_w = tf.cast(patch_mask.shape[1], tf.int32) * patch_w
        final_pixel_mask = tf.zeros((total_h, total_w), dtype=tf.float32)

    num_selected_patches = tf.shape(patch_indices_h)[0]

    for i in tf.range(num_selected_patches):
        h_start = h_starts[i]
        h_end = h_ends[i]
        w_start = w_starts[i]
        w_end = w_ends[i]

        # Create a slice of ones for the current patch area
        patch_area = tf.ones([h_end - h_start, w_end - w_start], dtype=tf.float32)

        # Pad the patch area to the full pixel mask size
        paddings = [[h_start, tf.shape(final_pixel_mask)[0] - h_end],
                    [w_start, tf.shape(final_pixel_mask)[1] - w_end]]
        padded_patch_area = tf.pad(patch_area, paddings, "CONSTANT")

        # Add to the existing pixel mask. This implies patches might overlap and sum up, 
        # but since patch_mask is binary, it would result in 1 for selected regions.
        # tf.add is fine as we are summing 0s and 1s, resulting in 1 where a patch exists.
        final_pixel_mask = tf.add(final_pixel_mask, padded_patch_area)
    
    # Ensure the final mask is binary (1s where patches are, 0s otherwise)
    final_pixel_mask = tf.cast(final_pixel_mask > 0, dtype=tf.float32)

    return final_pixel_mask