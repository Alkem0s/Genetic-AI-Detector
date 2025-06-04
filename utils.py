import random
import tensorflow as tf
import functools
from functools import lru_cache
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
            current_patch_features = patch_features[h, w, :]
            
            include_patch = tf.constant(False)
            
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
                include_patch = tf.cond(tf.logical_and(condition_met, tf.equal(action, 1)), 
                                        lambda: tf.constant(True), 
                                        lambda: include_patch) # Keep current include_patch status if not True
            
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

def evaluate_ga_individual_optimized(individual, precomputed_features, labels, patch_size,
                                   img_height, img_width, n_patches_h, n_patches_w,
                                   feature_weights, n_patches, max_possible_rules):
    """
    Optimized version of evaluate_ga_individual that uses precomputed patch features.
    This eliminates redundant feature extraction across all individuals.

    Args:
        individual: Rule set to evaluate
        precomputed_features: Pre-extracted patch features tensor [num_images, h_patches, w_patches, num_features]
        labels: Labels tensor [num_images]
        patch_size: Size of patches
        img_height, img_width: Image dimensions
        n_patches_h, n_patches_w: Number of patches in each dimension
        feature_weights: Weights for different features
        n_patches: Total number of patches
        max_possible_rules: Maximum number of rules allowed

    Returns:
        tuple: (fitness_score,)
    """
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from utils import generate_dynamic_mask, calculate_connectivity

        num_samples = tf.shape(precomputed_features)[0] # Keep as TensorFlow tensor
        
        # Track metrics across all evaluated images
        predictions = []
        # Initialize total_active_patches as a TensorFlow tensor to prevent overflow
        total_active_patches = tf.constant(0, dtype=tf.int64)
        connectivity_scores = []

        # Process each image using pre-computed features
        for img_idx in tf.range(num_samples): # Use tf.range for loop in TF graph
            # Use pre-computed features directly
            patch_features = precomputed_features[img_idx]

            # Generate mask using patch features
            patch_mask = generate_dynamic_mask(patch_features, n_patches_h, n_patches_w, individual)

            # Convert to TensorFlow tensor if it's not already
            if not isinstance(patch_mask, tf.Tensor):
                patch_mask = tf.convert_to_tensor(patch_mask, dtype=tf.float32)

            # Track connectivity score
            conn_score = calculate_connectivity(patch_mask)
            connectivity_scores.append(conn_score)

            # Track how many patches are active for efficiency calculation
            # Ensure active_patches is also a TensorFlow tensor before adding
            active_patches = tf.cast(tf.reduce_sum(patch_mask), dtype=tf.int64)
            total_active_patches += active_patches

            # Apply patch_mask directly to patch_features
            patch_mask_expanded = tf.expand_dims(tf.cast(patch_mask, tf.float32), axis=-1)
            masked_features = patch_features * patch_mask_expanded

            # Calculate weighted feature importance using configurable weights
            num_features = min(8, patch_features.shape[2])
            weights = feature_weights[:num_features]

            # Calculate per-feature scores
            feature_scores = tf.reduce_sum(masked_features[:, :, :num_features], axis=[0, 1])

            # Apply weights and sum
            total_score = tf.reduce_sum(feature_scores * weights)

            # Normalize by image size to make it scale-invariant
            normalized_score = total_score / (tf.cast(img_height * img_width, tf.float32))

            # Threshold for AI detection
            prediction = tf.cond(normalized_score > 0.01, lambda: tf.constant(1), lambda: tf.constant(0))
            predictions.append(prediction)

        # Calculate classification metrics
        predictions_tensor = tf.convert_to_tensor(predictions)
        labels_np = labels.numpy() if isinstance(labels, tf.Tensor) else labels
        predictions_np = predictions_tensor.numpy() if isinstance(predictions_tensor, tf.Tensor) else predictions_tensor

        accuracy = accuracy_score(labels_np, predictions_np)

        # Only calculate other metrics if we have both classes in predictions
        unique_predictions = tf.unique(predictions_tensor)[0]
        if tf.shape(unique_predictions)[0] > 1:
            precision = precision_score(labels_np, predictions_np, zero_division=0)
            recall = recall_score(labels_np, predictions_np, zero_division=0)
            f1 = f1_score(labels_np, predictions_np, zero_division=0)
        else:
            precision = 0.5
            recall = 0.5
            f1 = 0.5

        # Calculate average efficiency score
        # Ensure division is performed with float types
        avg_patch_selection_ratio = tf.cast(total_active_patches, tf.float32) / (tf.cast(num_samples * n_patches, tf.float32))

        # Convert to TensorFlow calculations
        avg_patch_ratio_tensor = tf.constant(avg_patch_selection_ratio, dtype=tf.float32)

        # Progressive penalty
        def efficiency_case_1(): return tf.constant(1.0, dtype=tf.float32)
        def efficiency_case_2(): return 1.0 - (avg_patch_ratio_tensor - 0.3) * 2
        def efficiency_case_3(): return 0.6 - (avg_patch_ratio_tensor - 0.5) * 1.2

        efficiency_score = tf.case([
            (avg_patch_ratio_tensor < 0.3, efficiency_case_1),
            (avg_patch_ratio_tensor < 0.5, efficiency_case_2)
        ], default=efficiency_case_3)

        # Calculate simplicity score based on number of rules
        simplicity_score = 1.0 - (len(individual) / max_possible_rules)

        # Use precomputed connectivity scores
        connectivity_score = tf.reduce_mean(tf.convert_to_tensor(connectivity_scores, dtype=tf.float32)).numpy() if connectivity_scores else 0.5

        # Combined fitness score - weighted average of metrics
        fitness = (
            accuracy * 0.4 +
            precision * 0.15 +
            recall * 0.15 +
            f1 * 0.1 +
            efficiency_score.numpy() * 0.1 +
            connectivity_score * 0.05 +
            simplicity_score * 0.05
        )

        return (fitness,)

    except Exception as e:
        tf.print("Error in optimized individual evaluation:", e)
        # Return a low fitness score instead of crashing
        return (0.0,)

def calculate_connectivity(patch_mask):
    """
    Calculate how connected the selected patches are in a vectorized manner.
    More connected patches are better as they likely represent coherent features.

    Args:
        patch_mask: Binary mask of shape (n_patches_h, n_patches_w)

    Returns:
        float: Connectivity score between 0 and 1
    """
    import tensorflow as tf # Need tf in the worker context
    # Ensure patch_mask is a TensorFlow tensor
    if not isinstance(patch_mask, tf.Tensor):
        patch_mask = tf.convert_to_tensor(patch_mask, dtype=tf.float32)

    # If no patches selected, return 0
    if tf.reduce_sum(patch_mask) == 0:
        return 0.0

    # Create shifted masks to check neighbors using TensorFlow operations
    # Pad the mask to handle edges properly
    padded_mask = tf.pad(patch_mask, [[1, 1], [1, 1]], constant_values=0)

    # Extract shifted versions (up, down, left, right)
    up = padded_mask[:-2, 1:-1]
    down = padded_mask[2:, 1:-1]
    left = padded_mask[1:-1, :-2]
    right = padded_mask[1:-1, 2:]

    # Count adjacent selected patches
    adjacent_count = tf.reduce_sum(
        tf.cast(patch_mask > 0, tf.float32) * tf.cast(up > 0, tf.float32) +
        tf.cast(patch_mask > 0, tf.float32) * tf.cast(down > 0, tf.float32) +
        tf.cast(patch_mask > 0, tf.float32) * tf.cast(left > 0, tf.float32) +
        tf.cast(patch_mask > 0, tf.float32) * tf.cast(right > 0, tf.float32)
    )

    # Total selected patches
    total_selected = tf.reduce_sum(patch_mask)

    # Normalize by maximum possible adjacencies
    max_possible_adj = tf.cast(total_selected * 4, tf.float32)
    if max_possible_adj <= 0:
        return 0.5 # Default for single or no patch selected

    connectivity = adjacent_count / max_possible_adj
    return connectivity.numpy()

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