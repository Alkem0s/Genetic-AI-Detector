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

    # Feature names used in genetic rules - convert to a list for indexing
    feature_names = list(global_config.feature_weights.keys())
    
    # Use tf.TensorArray for dynamic writes in a loop, then convert to tf.Tensor
    total_patches = n_patches_h * n_patches_w
    mask_list = tf.TensorArray(tf.int8, size=total_patches, dynamic_size=False, clear_after_read=False)

    # Initial loop variables for tf.while_loop
    h = tf.constant(0)
    w = tf.constant(0)
    idx = tf.constant(0)

    # Condition for the outer while loop (height)
    def cond_h(h, w, idx, mask_list):
        return h < n_patches_h

    # Body for the outer while loop (height)
    def body_h(h, w, idx, mask_list):
        # Condition for the inner while loop (width)
        def cond_w(h, w, idx, mask_list):
            return w < n_patches_w

        # Body for the inner while loop (width)
        def body_w(h, w, idx, mask_list):
            # Ensure patch_features indexing is safe (min is already handled by h_limit/w_limit if they were used)
            # Assuming patch_features shape is (H, W, Features)
            current_patch_features = patch_features[h, w, :]
            
            include_patch = tf.constant(False)
            
            # Use tf.while_loop for iterating through rules if rule_set can be converted to a Tensor
            # For simplicity with Python list `rule_set`, we'll keep the Python for loop here,
            # but be aware this implies `rule_set` values are known at graph construction time.
            # If `rule_set` can change dynamically, a tf.TensorArray for rules would be needed.
            
            # This part remains a Python for loop because rule_set is a Python list of dicts.
            # This means the rules are fixed at graph trace time.
            for rule in rule_set:
                feature_name = rule["feature"]
                threshold = rule["threshold"]
                operator = rule["operator"]
                action = rule["action"]
                
                try:
                    feature_idx = feature_names.index(feature_name)
                except ValueError:
                    continue # Skip rule for unavailable feature

                # Check if feature_idx is within bounds of current_patch_features
                if feature_idx >= tf.shape(current_patch_features)[0]:
                    continue

                value = current_patch_features[feature_idx]
                
                condition_met = tf.constant(False)
                if operator == ">":
                    condition_met = value > threshold
                else:  # operator == "<"
                    condition_met = value < threshold
                
                # If the condition is met and action is 1, include the patch
                # Use tf.cond for conditional assignment in graph mode
                include_patch = tf.cond(tf.logical_and(condition_met, tf.equal(action, 1)), 
                                        lambda: tf.constant(True), 
                                        lambda: include_patch) # Keep current include_patch status if not True

                # If include_patch becomes True, we can theoretically break.
                # However, tf.while_loop does not support direct `break` like Python for loops.
                # The `include_patch` will accumulate results across rules.
                # If "one matching rule with action=1 is enough", we need to structure it differently
                # For now, let's assume multiple rules can contribute to include_patch.
                # If only one rule is sufficient, the `include_patch = tf.cond(...)` needs careful consideration.
                # The original code's `break` won't translate directly.
                # A more "TensorFlow way" would be to compute all conditions and then reduce.
                
                # To replicate the "break" behavior if one rule is enough:
                # We can use tf.reduce_any on a list of booleans if we refactor how rules are applied.
                # For now, let's adjust the logic slightly:
                # If any rule sets `include_patch` to True, it stays True.
            
            # The current logic will set `include_patch` to True if ANY rule matches with action=1.
            # If no rules match with action=1, it remains False.
            
            mask_val = tf.cond(include_patch, lambda: tf.constant(1, dtype=tf.int8), lambda: tf.constant(0, dtype=tf.int8))
            
            # Write to TensorArray
            mask_list = mask_list.write(idx, mask_val)
            
            # Increment width and global index
            return h, w + 1, idx + 1, mask_list

        # Execute inner while loop for current row (h)
        _, final_w, final_idx, mask_list = tf.while_loop(
            cond_w, body_w, loop_vars=[h, tf.constant(0), idx, mask_list]
        )
        
        # Reset w for the next iteration of the outer loop, update idx
        return h + 1, tf.constant(0), final_idx, mask_list

    # Execute outer while loop
    _, _, final_idx, mask_list = tf.while_loop(
        cond_h, body_h, loop_vars=[h, w, idx, mask_list]
    )

    # Convert TensorArray back to Tensor and reshape
    patch_mask = tf.reshape(mask_list.stack(), (n_patches_h, n_patches_w))
    
    return patch_mask

@tf.function
def create_content_mask(image: tf.Tensor) -> tf.Tensor:
    """
    Creates a binary mask where 1 indicates non-black (content) pixels
    and 0 indicates black (border) pixels. Assumes black borders are (0,0,0) or 0.
    
    Args:
        image: Input image tf.Tensor (H, W, C or H, W, dtype=tf.float32, normalized to 0-1 or 0-255).
               It's important that black borders are exactly 0.

    Returns:
        tf.Tensor: A 2D binary mask (H, W, dtype=tf.float32).
    """
    # Sum absolute values across channels to detect any non-zero pixel
    # This handles both grayscale (no channel dim or 1 channel) and color images.
    # tf.abs ensures it works even if images have negative values or are centered around 0.
    if tf.rank(image) == 3:
        # For color images, sum across the channel dimension
        pixel_intensity_sum = tf.reduce_sum(tf.abs(image), axis=-1)
    else:
        # For grayscale images (2D), just take the absolute value
        pixel_intensity_sum = tf.abs(image)
        
    # Create a binary mask: 1 where sum > 0 (non-black), 0 where sum == 0 (black)
    content_mask = tf.cast(pixel_intensity_sum > 0, tf.float32)
    
    return content_mask

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

# utils.py - Corrected function
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

    # Initialize pixel mask - this part looks fine
    if image_shape is not None:
        pixel_mask_shape = tf.cast(image_shape[:2], tf.int32)
        pixel_mask = tf.zeros(pixel_mask_shape, dtype=tf.float32)
    else:
        # Calculate total size from patch dimensions
        total_h = tf.cast(patch_mask.shape[0], tf.int32) * patch_h
        total_w = tf.cast(patch_mask.shape[1], tf.int32) * patch_w
        pixel_mask = tf.zeros((total_h, total_w), dtype=tf.float32)

    # Populate pixel mask using tf.where and tf.tensor_scatter_nd_update for efficiency
    # Create a grid of patch indices
    # *** FIX HERE: Cast indices to tf.int32 ***
    selected_indices = tf.where(patch_mask == 1)
    patch_indices_h = tf.cast(selected_indices[:, 0], tf.int32)
    patch_indices_w = tf.cast(selected_indices[:, 1], tf.int32)

    # Calculate start and end pixel coordinates for each selected patch
    h_starts = patch_indices_h * patch_h
    w_starts = patch_indices_w * patch_w
    h_ends = (patch_indices_h + 1) * patch_h
    w_ends = (patch_indices_w + 1) * patch_w

    if image_shape is not None:
        img_h, img_w = tf.cast(image_shape[0], tf.int32), tf.cast(image_shape[1], tf.int32)
        h_ends = tf.minimum(h_ends, img_h)
        w_ends = tf.minimum(w_ends, img_w)

    # Create an empty pixel mask to build upon
    if image_shape is not None:
        final_pixel_mask = tf.zeros(tf.cast(image_shape[:2], tf.int32), dtype=tf.float32)
    else:
        total_h = tf.cast(patch_mask.shape[0], tf.int32) * patch_h
        total_w = tf.cast(patch_mask.shape[1], tf.int32) * patch_w
        final_pixel_mask = tf.zeros((total_h, total_w), dtype=tf.float32)

    num_selected_patches = tf.shape(patch_indices_h)[0]

    # Use tf.TensorArray for efficient accumulation of patches in graph mode
    # This is a more performant way than repeated tf.add on large tensors in a loop
    # We will build a list of sparse updates.
    
    # Define a TensorArray to store the patches to be scattered
    updates_ta = tf.TensorArray(tf.float32, size=num_selected_patches, dynamic_size=False, clear_after_read=False)
    indices_ta = tf.TensorArray(tf.int32, size=num_selected_patches, dynamic_size=False, clear_after_read=False)

    i = tf.constant(0)
    
    # Use tf.while_loop for the iteration
    def cond(i, updates_ta, indices_ta):
        return i < num_selected_patches

    def body(i, updates_ta, indices_ta):
        h_start = h_starts[i]
        h_end = h_ends[i]
        w_start = w_starts[i]
        w_end = w_ends[i]

        # Create a slice of ones for the current patch area
        patch_area = tf.ones([h_end - h_start, w_end - w_start], dtype=tf.float32)

        # For `tf.tensor_scatter_nd_update`, we need the indices and values for the update.
        # This approach is generally more efficient than padding and adding
        # which can create large intermediate tensors and multiple copies.
        
        # Get all pixel coordinates within this patch
        # This creates a meshgrid of pixel coordinates for the current patch
        h_coords = tf.range(h_start, h_end, dtype=tf.int32)
        w_coords = tf.range(w_start, w_end, dtype=tf.int32)
        grid_h, grid_w = tf.meshgrid(h_coords, w_coords)
        
        # Stack coordinates to get [[h1, w1], [h2, w2], ...]
        patch_pixel_indices = tf.stack([tf.reshape(grid_h, [-1]), tf.reshape(grid_w, [-1])], axis=-1)
        
        # The values to scatter are all 1.0 for these pixels
        patch_pixel_values = tf.ones(tf.shape(patch_pixel_indices)[0], dtype=tf.float32)
        
        updates_ta = updates_ta.write(i, patch_pixel_values)
        indices_ta = indices_ta.write(i, patch_pixel_indices)

        return i + 1, updates_ta, indices_ta

    _, final_updates_ta, final_indices_ta = tf.while_loop(cond, body, [i, updates_ta, indices_ta])

    # Stack the collected updates and indices
    all_pixel_values = final_updates_ta.concat()
    all_pixel_indices = final_indices_ta.concat()

    # Use tf.tensor_scatter_nd_add to accumulate values at the specified indices.
    # Since values are 1.0, this will effectively mark the pixels as active.
    final_pixel_mask = tf.tensor_scatter_nd_add(final_pixel_mask, all_pixel_indices, all_pixel_values)
    
    # Ensure the final mask is binary (1s where patches are, 0s otherwise)
    final_pixel_mask = tf.cast(final_pixel_mask > 0, dtype=tf.float32)

    return final_pixel_mask