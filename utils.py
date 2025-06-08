# utils.py
import tensorflow as tf
import global_config as config

@tf.function
def generate_dynamic_mask(patch_features, n_patches_h, n_patches_w, rule_tensor):
    """
    Generate a binary mask for patches based on dynamic rules.
    
    Args:
        patch_features: [n_patches_h, n_patches_w, n_features]
        n_patches_h: Height in patches
        n_patches_w: Width in patches  
        rule_tensor: [max_rules, 4] tensor where each row is [feature_idx, threshold, operator, action]
    
    Returns:
        Binary mask tensor of shape [n_patches_h, n_patches_w]
    """
    # Initialize mask with all zeros
    patch_mask = tf.zeros((n_patches_h, n_patches_w), dtype=tf.bool)
    
    # Filter valid rules (feature_idx >= 0 and action == 1)
    feature_indices = rule_tensor[:, 0]
    actions = rule_tensor[:, 3]
    
    valid_mask = tf.logical_and(
        feature_indices >= 0,
        tf.equal(actions, 1)  # action == 1 (include)
    )
    
    # Extract valid rules
    valid_rules = tf.boolean_mask(rule_tensor, valid_mask)
    n_valid_rules = tf.shape(valid_rules)[0]
    
    # Extract rule components
    valid_feature_indices = tf.cast(valid_rules[:, 0], tf.int32)
    valid_thresholds = valid_rules[:, 1]
    valid_operators = tf.cast(valid_rules[:, 2], tf.int32)
    
    # Process each valid rule and OR the results
    def process_rule(i, current_mask):
        feature_idx = valid_feature_indices[i]
        threshold = valid_thresholds[i]
        operator = valid_operators[i]
        
        # Bounds check for feature index
        feature_idx = tf.minimum(feature_idx, tf.shape(patch_features)[2] - 1)
        
        # Extract feature values for all patches
        feature_values = patch_features[:, :, feature_idx]
        
        # Apply condition based on operator
        condition = tf.cond(
            tf.equal(operator, 0),  # '>' operator
            lambda: feature_values > threshold,
            lambda: feature_values < threshold  # '<' operator
        )
        
        # OR with current mask
        return tf.logical_or(current_mask, condition)
    
    # Apply all valid rules using while_loop
    def rule_loop_condition(i, mask):
        return i < n_valid_rules
    
    def rule_loop_body(i, mask):
        new_mask = process_rule(i, mask)
        return i + 1, new_mask
    
    _, final_mask = tf.while_loop(
        rule_loop_condition,
        rule_loop_body,
        [tf.constant(0), patch_mask],
        parallel_iterations=1
    )
    
    return tf.cast(final_mask, tf.int8)


@tf.function
def calculate_connectivity(patch_mask):
    """
    Connectivity calculation using convolution.

    Args:
        patch_mask: Binary mask of shape (n_patches_h, n_patches_w)
    Returns:
        tf.Tensor: Connectivity score between 0 and 1
    """
    patch_mask = tf.cast(patch_mask, tf.float32)
    total_selected = tf.reduce_sum(patch_mask)
    
    # Early exit if no patches selected
    if tf.equal(total_selected, 0.0):
        return 0.0
    
    # Use convolution to count neighbors efficiently
    # Kernel for 4-connectivity (up, down, left, right)
    kernel = tf.constant([
        [0., 1., 0.],
        [1., 0., 1.],
        [0., 1., 0.]
    ], dtype=tf.float32)
    kernel = tf.reshape(kernel, [3, 3, 1, 1])
    
    # Add batch and channel dimensions for conv2d
    mask_4d = tf.expand_dims(tf.expand_dims(patch_mask, 0), -1)
    
    # Count neighbors for each patch
    neighbor_count = tf.nn.conv2d(mask_4d, kernel, strides=[1,1,1,1], padding='SAME')
    neighbor_count = tf.squeeze(neighbor_count)
    
    # Only count neighbors for selected patches
    valid_neighbor_count = patch_mask * neighbor_count
    total_connections = tf.reduce_sum(valid_neighbor_count)
    
    # Normalize by maximum possible connections (each patch can have up to 4 neighbors)
    max_connections = total_selected * 4.0
    connectivity = tf.cond(
        max_connections > 0,
        lambda: total_connections / max_connections,
        lambda: 0.0
    )
    
    return connectivity


@tf.function
def compute_image_scores(precomputed_features, feature_weights_sliced, 
                              individual_rule_tensor, n_patches_h, n_patches_w):
    """
    Batch process all images at once for better GPU utilization.
    
    Args:
        precomputed_features: [batch_size, n_patches_h, n_patches_w, n_features]
        feature_weights_sliced: Pre-sliced feature weights
        individual_rule_tensor: Rule tensor for mask generation
        n_patches_h, n_patches_w: Patch dimensions
    
    Returns:
        scores, active_patches_per_image, connectivity_scores
    """
    
    # Process all images in batch
    def process_single_image(img_features):
        # Generate mask
        patch_mask = generate_dynamic_mask(
            img_features, n_patches_h, n_patches_w, individual_rule_tensor
        )
        
        # Calculate connectivity
        conn_score = calculate_connectivity(patch_mask)
        
        # Count active patches
        active_patches = tf.reduce_sum(tf.cast(patch_mask, tf.int32))
        
        # Apply mask to features
        patch_mask_expanded = tf.expand_dims(tf.cast(patch_mask, tf.float32), axis=-1)
        masked_features = img_features * patch_mask_expanded
        
        # Calculate weighted feature importance
        feature_scores = tf.reduce_sum(masked_features, axis=[0, 1])
        total_score = tf.reduce_sum(feature_scores * feature_weights_sliced)
        
        return total_score, active_patches, conn_score
    
    # Use vectorized_map for better performance
    scores, active_patches_per_image, connectivity_scores = tf.vectorized_map(
        process_single_image, precomputed_features
    )
    
    return scores, active_patches_per_image, connectivity_scores


@tf.function
def calculate_adaptive_threshold(scores, labels):
    """    
    Calculate adaptive threshold based on scores and labels.
    """
    num_samples = tf.shape(scores)[0]
    score_mean = tf.reduce_mean(scores)
    score_std = tf.math.reduce_std(scores)
    
    # Calculate positive ratio
    positive_ratio = tf.reduce_mean(tf.cast(labels, tf.float32))
    
    # Percentile-based threshold
    sorted_scores = tf.sort(scores, direction='DESCENDING')
    threshold_idx = tf.cast(positive_ratio * tf.cast(num_samples, tf.float32), tf.int32)
    threshold_idx = tf.clip_by_value(threshold_idx, 0, num_samples - 1)
    percentile_threshold = sorted_scores[threshold_idx]
    
    # Statistical threshold
    statistical_threshold = score_mean + 0.5 * score_std
    
    # Conservative threshold
    adaptive_threshold = tf.minimum(percentile_threshold, statistical_threshold)
    adaptive_threshold = tf.maximum(adaptive_threshold, score_mean * 0.1)
    adaptive_threshold = tf.minimum(adaptive_threshold, score_mean + 2 * score_std)
    
    return adaptive_threshold


@tf.function(reduce_retracing=True)
def _compute_fitness_score(precomputed_features, labels,
                                n_patches_h, n_patches_w, feature_weights, n_patches, 
                                individual_len, max_possible_rules, individual_rule_tensor):
    """
    Core fitness computation with batch processing.
    """
    num_samples = tf.shape(precomputed_features)[0]
    
    # Pre-slice feature weights once
    num_features = tf.minimum(tf.shape(feature_weights)[0], tf.shape(precomputed_features)[3])
    feature_weights_sliced = feature_weights[:num_features]
    
    # Batch process all images
    scores, active_patches_per_image, connectivity_scores = compute_image_scores(
        precomputed_features, feature_weights_sliced, individual_rule_tensor, 
        n_patches_h, n_patches_w
    )
    
    # Normalize scores by image size
    normalized_scores = scores / (config.image_size * config.image_size)

    # Calculate adaptive threshold
    adaptive_threshold = calculate_adaptive_threshold(normalized_scores, labels)
    
    # Generate predictions
    predictions = tf.cast(normalized_scores > adaptive_threshold, tf.int64)
    
    # Fallback for extreme predictions
    num_predictions = tf.reduce_sum(tf.cast(predictions, tf.float32))
    score_mean = tf.reduce_mean(normalized_scores)
    score_std = tf.math.reduce_std(normalized_scores)
    
    def fallback_predictions():
        fallback_threshold = tf.cond(
            tf.equal(num_predictions, 0),
            lambda: score_mean * 0.5,
            lambda: tf.cond(
                tf.equal(num_predictions, tf.cast(num_samples, tf.float32)),
                lambda: score_mean + score_std,
                lambda: adaptive_threshold
            )
        )
        return tf.cast(normalized_scores > fallback_threshold, tf.int64)
    
    predictions = tf.cond(
        tf.logical_or(
            tf.equal(num_predictions, 0),
            tf.equal(num_predictions, tf.cast(num_samples, tf.float32))
        ),
        fallback_predictions,
        lambda: predictions
    )
    
    # Calculate metrics (optimized versions)
    precision, recall, f1 = calculate_precision_recall_f1(labels, predictions)
    balanced_accuracy = calculate_balanced_accuracy(labels, predictions)
    
    # Calculate efficiency and other scores
    total_active_patches = tf.reduce_sum(active_patches_per_image)
    avg_patch_selection_ratio = tf.cast(total_active_patches, tf.float32) / tf.cast(num_samples * n_patches, tf.float32)
    
    efficiency_score = tf.case([
        (avg_patch_selection_ratio < 0.3, lambda: 1.0),
        (avg_patch_selection_ratio < 0.5, lambda: 1.0 - (avg_patch_selection_ratio - 0.3) * 2)
    ], default=lambda: 0.6 - (avg_patch_selection_ratio - 0.5) * 1.2)
    
    simplicity_score = 1.0 - (tf.cast(individual_len, tf.float32) / tf.cast(max_possible_rules, tf.float32))
    connectivity_score = tf.reduce_mean(connectivity_scores)
    
    # Calculate final fitness
    fitness = (
        balanced_accuracy * config.fitness_weights['balanced_accuracy'] +
        f1 * config.fitness_weights['f1'] +
        precision * config.fitness_weights['precision'] +
        recall * config.fitness_weights['recall'] +
        efficiency_score * config.fitness_weights['efficiency_score'] +
        connectivity_score * config.fitness_weights['connectivity_score'] +
        simplicity_score * config.fitness_weights['simplicity_score']
    )
    
    # Debug information
    if config.verbose:
        tf.print("=== Fitness Debug ===")
        tf.print("Scores mean/std:", score_mean, score_std)
        tf.print("Threshold:", adaptive_threshold)
        tf.print("Predictions:", tf.reduce_sum(tf.cast(predictions, tf.float32)), "/", num_samples)
        tf.print("Precision:", precision)
        tf.print("Recall:", recall)
        tf.print("F1:", f1)
        tf.print("Balanced Acc:", balanced_accuracy)
        tf.print("Efficiency score:", efficiency_score)
        tf.print("Simplicity score:", simplicity_score)
        tf.print("Connectivity score:", connectivity_score)
        tf.print("Fitness:", fitness)
        tf.print("==================")
    
    return fitness


@tf.function
def calculate_precision_recall_f1(y_true, y_pred):
    """
    Optimized precision, recall, F1 calculation with fewer operations.
    """
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_f = tf.cast(y_pred, tf.float32)
    
    # Calculate components in one pass
    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fp = tf.reduce_sum((1 - y_true_f) * y_pred_f)
    fn = tf.reduce_sum(y_true_f * (1 - y_pred_f))
    
    # Calculate precision and recall with single condition checks
    precision = tf.cond(
        tf.equal(tp + fp, 0),
        lambda: tf.cond(tf.equal(tf.reduce_sum(y_true_f), 0), lambda: 1.0, lambda: 0.0),
        lambda: tp / (tp + fp)
    )
    
    recall = tf.cond(
        tf.equal(tp + fn, 0),
        lambda: 1.0,
        lambda: tp / (tp + fn)
    )
    
    f1 = tf.cond(
        tf.equal(precision + recall, 0),
        lambda: 0.0,
        lambda: 2 * (precision * recall) / (precision + recall)
    )
    
    return precision, recall, f1

@tf.function
def calculate_matthews_correlation_coefficient(y_true, y_pred):
    """
    Calculate Matthews Correlation Coefficient (MCC).
    MCC is a more balanced measure for imbalanced datasets.
    Returns a value between -1 and 1, where 1 is perfect prediction.
    """
    y_true = tf.cast(y_true, tf.int64)
    y_pred = tf.cast(y_pred, tf.int64)
    
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_f = tf.cast(y_pred, tf.float32)
    
    # Calculate confusion matrix components
    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fp = tf.reduce_sum((1 - y_true_f) * y_pred_f)
    fn = tf.reduce_sum(y_true_f * (1 - y_pred_f))
    tn = tf.reduce_sum((1 - y_true_f) * (1 - y_pred_f))
    
    # MCC formula: (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    numerator = tp * tn - fp * fn
    denominator = tf.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    mcc = tf.cond(
        tf.equal(denominator, 0),
        lambda: tf.constant(0.0, dtype=tf.float32),  # Return 0 if denominator is 0
        lambda: numerator / denominator
    )
    
    return mcc

@tf.function
def calculate_balanced_accuracy(y_true, y_pred):
    """
    Optimized balanced accuracy calculation.
    """
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_f = tf.cast(y_pred, tf.float32)
    
    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fp = tf.reduce_sum((1 - y_true_f) * y_pred_f)
    fn = tf.reduce_sum(y_true_f * (1 - y_pred_f))
    tn = tf.reduce_sum((1 - y_true_f) * (1 - y_pred_f))
    
    sensitivity = tf.cond(tf.equal(tp + fn, 0), lambda: 1.0, lambda: tp / (tp + fn))
    specificity = tf.cond(tf.equal(tn + fp, 0), lambda: 1.0, lambda: tn / (tn + fp))
    
    return (sensitivity + specificity) / 2.0


def evaluate_ga_individual(individual, precomputed_features, labels, 
                                n_patches_h, n_patches_w,
                                feature_weights, n_patches, max_possible_rules):
    """
    Evaluate a single individual in the genetic algorithm.
    """
    try:
        fitness = _compute_fitness_score(
            precomputed_features, labels, n_patches_h, n_patches_w, feature_weights, n_patches,
            individual.num_active_rules, max_possible_rules, individual.rules_tensor
        )
        
        return (fitness,)
        
    except Exception as e:
        if config.verbose:
            tf.print("Error in individual evaluation:", str(e))
        return (0.0,)


@tf.function
def convert_patch_mask_to_pixel_mask(patch_mask):
    """
    Convert a 2D patch mask to pixel-level mask using GPU-optimized TensorFlow operations.
    Designed for fixed patch sizes and image shapes with minimal function tracing.
    
    Args:
        patch_mask (tf.Tensor): 2D binary mask of shape (n_patches_h, n_patches_w)
    
    Returns:
        tf.Tensor: Pixel-level binary mask of shape (height, width)
    """

    patch_size = config.patch_size
    image_shape = (config.image_size, config.image_size)

    # Handle patch size input
    if isinstance(patch_size, int):
        patch_h = patch_w = patch_size
    else:
        patch_h, patch_w = patch_size
    
    # Convert to TensorFlow constants for graph optimization
    patch_h = tf.constant(patch_h, dtype=tf.int32)
    patch_w = tf.constant(patch_w, dtype=tf.int32)
    img_h = tf.constant(image_shape[0], dtype=tf.int32)
    img_w = tf.constant(image_shape[1], dtype=tf.int32)
    
    # Ensure patch_mask is float32 and add batch/channel dimensions for tf.image.resize
    patch_mask_float = tf.cast(patch_mask, tf.float32)
    expanded_mask = tf.expand_dims(tf.expand_dims(patch_mask_float, axis=0), axis=-1)
    
    # Use tf.image.resize with nearest neighbor - this is GPU-optimized
    pixel_mask = tf.image.resize(
        expanded_mask,
        [img_h, img_w],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    
    # Remove batch and channel dimensions
    pixel_mask = tf.squeeze(pixel_mask, axis=[0, -1])
    
    # Ensure binary output (handle any floating point precision issues)
    pixel_mask = tf.cast(pixel_mask >= 0.5, tf.float32)
    
    return pixel_mask


def get_edge_detection(image, method='canny', **kwargs):
    """Perform edge detection on an image using TensorFlow.    
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