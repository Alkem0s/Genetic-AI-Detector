# utils.py
import tensorflow as tf
import global_config

@tf.function
def generate_dynamic_mask(patch_features, n_patches_h, n_patches_w, rule_tensor):
    """
    Vectorized dynamic mask generation using TensorFlow operations.
    
    Args:
        patch_features: [n_patches_h, n_patches_w, n_features]
        n_patches_h: Height in patches
        n_patches_w: Width in patches  
        rule_tensor: [max_rules, 4] tensor where each row is [feature_idx, threshold, operator, action]
                    operator: 0 for '>', 1 for '<'
                    action: 0 for exclude, 1 for include
    
    Returns:
        Binary mask tensor of shape [n_patches_h, n_patches_w]
    """
    # Initialize mask with all zeros
    patch_mask = tf.zeros((n_patches_h, n_patches_w), dtype=tf.bool)
    
    # Get the number of actual rules (non-zero feature indices)
    valid_rules = tf.reduce_sum(tf.cast(rule_tensor[:, 0] >= 0, tf.int32))
    
    # Process each rule using tf.while_loop for dynamic number of rules
    def process_rule(i, current_mask):
        # Get rule parameters
        feature_idx = tf.cast(rule_tensor[i, 0], tf.int32)
        threshold = rule_tensor[i, 1]
        operator = tf.cast(rule_tensor[i, 2], tf.int32)
        action = tf.cast(rule_tensor[i, 3], tf.int32)
        
        # Skip if action is not 1 (include patch) or feature_idx is invalid
        def apply_rule():
            # Extract feature values for all patches
            feature_values = patch_features[:, :, feature_idx]
            
            # Apply condition based on operator
            condition = tf.cond(
                tf.equal(operator, 0),  # '>' operator
                lambda: feature_values > threshold,
                lambda: feature_values < threshold  # '<' operator
            )
            
            # Update mask (OR operation - if any rule matches, include patch)
            return tf.logical_or(current_mask, condition)
        
        # Only apply rule if action is 1 and feature_idx is valid
        should_apply = tf.logical_and(
            tf.equal(action, 1),
            tf.logical_and(
                feature_idx >= 0,
                feature_idx < tf.shape(patch_features)[2]
            )
        )
        
        return tf.cond(should_apply, apply_rule, lambda: current_mask)
    
    # Apply all valid rules
    def rule_loop_condition(i, mask):
        return i < valid_rules
    
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
def calculate_accuracy_tf(y_true, y_pred):
    """Calculate accuracy using TensorFlow operations."""
    y_true = tf.cast(y_true, tf.int64)
    y_pred = tf.cast(y_pred, tf.int64)
    correct_predictions = tf.cast(tf.equal(y_true, y_pred), tf.float32)
    return tf.reduce_mean(correct_predictions)


@tf.function
def calculate_precision_recall_f1_tf(y_true, y_pred):
    """
    Calculate precision, recall, and F1 score using TensorFlow operations.
    Handles edge cases where no positive predictions or no positive labels exist.
    """
    # Ensure both tensors have the same dtype
    y_true = tf.cast(y_true, tf.int64)
    y_pred = tf.cast(y_pred, tf.int64)
    
    # Convert to float32 for calculations
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_f = tf.cast(y_pred, tf.float32)
    
    # Calculate confusion matrix components
    tp = tf.reduce_sum(y_true_f * y_pred_f)  # True positives
    fp = tf.reduce_sum((1 - y_true_f) * y_pred_f)  # False positives
    fn = tf.reduce_sum(y_true_f * (1 - y_pred_f))  # False negatives
    tn = tf.reduce_sum((1 - y_true_f) * (1 - y_pred_f))  # True negatives
    
    # Calculate precision with better handling of edge cases
    precision = tf.cond(
        tf.equal(tp + fp, 0),
        # If no positive predictions, check if there are positive labels
        lambda: tf.cond(
            tf.equal(tf.reduce_sum(y_true_f), 0),
            lambda: tf.constant(1.0, dtype=tf.float32),  # Perfect precision if no positives to find
            lambda: tf.constant(0.0, dtype=tf.float32)   # Zero precision if missed all positives
        ),
        lambda: tp / (tp + fp)
    )
    
    # Calculate recall with better handling of edge cases
    recall = tf.cond(
        tf.equal(tp + fn, 0),
        # If no positive labels, perfect recall
        lambda: tf.constant(1.0, dtype=tf.float32),
        lambda: tp / (tp + fn)
    )
    
    # Calculate F1 score with improved handling
    f1 = tf.cond(
        tf.equal(precision + recall, 0),
        lambda: tf.constant(0.0, dtype=tf.float32),  # F1 is 0 if both precision and recall are 0
        lambda: 2 * (precision * recall) / (precision + recall)
    )
    
    if global_config.verbose:
        # Add debugging information
        tf.print("TP:", tp, "FP:", fp, "FN:", fn, "TN:", tn)
        tf.print("Precision:", precision, "Recall:", recall, "F1:", f1)
    
    return precision, recall, f1


@tf.function
def calculate_balanced_accuracy_tf(y_true, y_pred):
    """
    Calculate balanced accuracy (average of sensitivity and specificity).
    This is more robust for imbalanced datasets.
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
    
    # Sensitivity (recall) = TP / (TP + FN)
    sensitivity = tf.cond(
        tf.equal(tp + fn, 0),
        lambda: tf.constant(1.0, dtype=tf.float32),
        lambda: tp / (tp + fn)
    )
    
    # Specificity = TN / (TN + FP)
    specificity = tf.cond(
        tf.equal(tn + fp, 0),
        lambda: tf.constant(1.0, dtype=tf.float32),
        lambda: tn / (tn + fp)
    )
    
    # Balanced accuracy = (Sensitivity + Specificity) / 2
    balanced_acc = (sensitivity + specificity) / 2.0
    
    return balanced_acc


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


@tf.function(reduce_retracing=True)
def _compute_fitness_core(precomputed_features, labels, img_height, img_width, 
                         n_patches_h, n_patches_w, feature_weights, n_patches, 
                         individual_len, max_possible_rules, individual_tensor):
    """
    Updated core TensorFlow computation function with improved metrics.
    """
    
    def process_single_image(args):
        """Process a single image's features and return prediction and metrics."""
        img_features, img_idx = args
        
        # Generate mask using patch features and individual tensor
        patch_mask = generate_dynamic_mask(img_features, n_patches_h, n_patches_w, individual_tensor)
        
        # Convert to TensorFlow tensor if it's not already
        if not isinstance(patch_mask, tf.Tensor):
            patch_mask = tf.convert_to_tensor(patch_mask, dtype=tf.float32)
        
        # Calculate connectivity score
        conn_score = calculate_connectivity(patch_mask)
        
        # Count active patches
        active_patches = tf.cast(tf.reduce_sum(patch_mask), dtype=tf.int64)
        
        # Apply patch_mask to features
        patch_mask_expanded = tf.expand_dims(tf.cast(patch_mask, tf.float32), axis=-1)
        masked_features = img_features * patch_mask_expanded
        
        # Calculate weighted feature importance
        num_features = tf.minimum(8, tf.shape(img_features)[2])
        weights = feature_weights[:num_features]
        
        # Calculate per-feature scores
        feature_scores = tf.reduce_sum(masked_features[:, :, :num_features], axis=[0, 1])
        
        # Apply weights and sum
        total_score = tf.reduce_sum(feature_scores * weights)
        
        # Normalize by image size
        normalized_score = total_score / tf.cast(img_height * img_width, tf.float32)
        
        # Store the normalized score for threshold calculation later
        prediction = tf.constant(0, dtype=tf.int64)  # Placeholder, will be calculated globally
        
        return prediction, active_patches, conn_score, normalized_score
    
    num_samples = tf.shape(precomputed_features)[0]
    
    # Create image indices for map_fn
    image_indices = tf.range(num_samples, dtype=tf.int32)
    
    # Use tf.map_fn to vectorize processing across all images
    predictions, active_patches_per_image, connectivity_scores, scores = tf.map_fn(
        process_single_image,
        (precomputed_features, image_indices),
        fn_output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.int64),  # prediction (placeholder)
            tf.TensorSpec(shape=(), dtype=tf.int64),  # active_patches
            tf.TensorSpec(shape=(), dtype=tf.float32), # connectivity_score
            tf.TensorSpec(shape=(), dtype=tf.float32)  # normalized_score
        ),
        parallel_iterations=10
    )
    
    # Calculate adaptive threshold based on actual score distribution
    score_mean = tf.reduce_mean(scores)
    score_std = tf.math.reduce_std(scores)
    
    # Calculate class distribution
    num_positive_labels = tf.reduce_sum(tf.cast(labels, tf.float32))
    positive_ratio = num_positive_labels / tf.cast(num_samples, tf.float32)
    
    # Method 1: Percentile-based threshold
    # Sort scores and pick threshold based on expected positive ratio
    sorted_scores = tf.sort(scores, direction='DESCENDING')
    threshold_idx = tf.cast(positive_ratio * tf.cast(num_samples, tf.float32), tf.int32)
    threshold_idx = tf.maximum(1, tf.minimum(threshold_idx, num_samples - 1))
    percentile_threshold = sorted_scores[threshold_idx]
    
    # Method 2: Statistical threshold
    statistical_threshold = score_mean + 0.5 * score_std
    
    # Method 3: Otsu-like threshold (simplified)
    # Find threshold that maximizes between-class variance
    min_score = tf.reduce_min(scores)
    max_score = tf.reduce_max(scores)
    score_range = max_score - min_score
    
    # Use the more conservative of percentile and statistical methods
    adaptive_threshold = tf.minimum(percentile_threshold, statistical_threshold)
    
    # Ensure threshold is within reasonable bounds
    adaptive_threshold = tf.maximum(adaptive_threshold, score_mean * 0.1)  # At least 10% of mean
    adaptive_threshold = tf.minimum(adaptive_threshold, score_mean + 2 * score_std)  # Not more than 2 std above mean
    
    # Apply threshold to get final predictions
    predictions = tf.cast(scores > adaptive_threshold, tf.int64)
    
    # Fallback: If we get no predictions or all predictions, use a more aggressive approach
    num_predictions = tf.reduce_sum(tf.cast(predictions, tf.float32))
    
    def fallback_predictions():
        # If no predictions, lower threshold to get at least some predictions
        fallback_threshold = tf.cond(
            tf.equal(num_predictions, 0),
            lambda: score_mean * 0.5,  # Much lower threshold
            lambda: tf.cond(
                tf.equal(num_predictions, tf.cast(num_samples, tf.float32)),
                lambda: score_mean + score_std,  # Higher threshold if all predicted positive
                lambda: adaptive_threshold  # Keep original if reasonable
            )
        )
        return tf.cast(scores > fallback_threshold, tf.int64)
    
    # Use fallback if predictions are too extreme (all 0s or all 1s)
    predictions = tf.cond(
        tf.logical_or(
            tf.equal(num_predictions, 0),
            tf.equal(num_predictions, tf.cast(num_samples, tf.float32))
        ),
        fallback_predictions,
        lambda: predictions
    )
    
    # Calculate total active patches
    total_active_patches = tf.reduce_sum(active_patches_per_image)
    
    if global_config.verbose:
        # Print debugging information
        tf.print("=== Threshold Selection ===")
        tf.print("Score mean:", score_mean, "Score std:", score_std)
        tf.print("Positive ratio:", positive_ratio)
        tf.print("Percentile threshold:", percentile_threshold)
        tf.print("Statistical threshold:", statistical_threshold)
        tf.print("Selected threshold:", adaptive_threshold)
        tf.print("Scores > threshold:", tf.reduce_sum(tf.cast(scores > adaptive_threshold, tf.float32)))
        tf.print("===========================")
        
        tf.print("Prediction distribution:", tf.reduce_sum(tf.cast(predictions, tf.float32)), "/", num_samples)
        tf.print("Label distribution:", tf.reduce_sum(tf.cast(labels, tf.float32)), "/", num_samples)
    
    # Calculate classification metrics using improved TensorFlow operations
    accuracy = calculate_accuracy_tf(labels, predictions)
    precision, recall, f1 = calculate_precision_recall_f1_tf(labels, predictions)
    balanced_accuracy = calculate_balanced_accuracy_tf(labels, predictions)
    mcc = calculate_matthews_correlation_coefficient(labels, predictions)
    
    # Calculate average efficiency score
    avg_patch_selection_ratio = tf.cast(total_active_patches, tf.float32) / tf.cast(num_samples * n_patches, tf.float32)
    
    # Progressive penalty for efficiency
    efficiency_score = tf.case([
        (avg_patch_selection_ratio < 0.3, lambda: tf.constant(1.0, dtype=tf.float32)),
        (avg_patch_selection_ratio < 0.5, lambda: 1.0 - (avg_patch_selection_ratio - 0.3) * 2)
    ], default=lambda: 0.6 - (avg_patch_selection_ratio - 0.5) * 1.2)
    
    # Calculate simplicity score
    simplicity_score = 1.0 - (tf.cast(individual_len, tf.float32) / tf.cast(max_possible_rules, tf.float32))
    
    # Calculate average connectivity score
    connectivity_score = tf.reduce_mean(connectivity_scores)
    
    # Updated fitness calculation with better weights and inclusion of new metrics
    fitness_dict = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'efficiency_score': efficiency_score,
        'connectivity_score': connectivity_score,
        'simplicity_score': simplicity_score,
        'avg_score': tf.reduce_mean(scores),
        'score_std': tf.math.reduce_std(scores)
    }
    
    # Enhanced fitness calculation - prioritize balanced metrics
    fitness = (
        balanced_accuracy * 0.25 +
        f1 * 0.2 +
        mcc * 0.1 +
        precision * 0.1 +
        recall * 0.1 +
        efficiency_score * 0.1 +
        connectivity_score * 0.05 +
        simplicity_score * 0.05
    )
    
    fitness_dict['fitness'] = fitness

    if global_config.verbose:
        # Print detailed fitness metrics
        tf.print("=== Fitness Metrics ===")
        for key, value in fitness_dict.items():
            tf.print(f"{key}:", value)
        tf.print("=====================")
    else:
        # Print only the final fitness value
        tf.print("Final fitness:", fitness)
    
    return fitness


def evaluate_ga_individual(individual, precomputed_features, labels,
                                   img_height, img_width, n_patches_h, n_patches_w,
                                   feature_weights, n_patches, max_possible_rules):
    """
    Optimized and vectorized version of evaluate_ga_individual that uses precomputed patch features
    with TensorFlow graph compilation and vectorized operations.

    Args:
        individual: Rule set to evaluate (Python list/object)
        precomputed_features: Pre-extracted patch features tensor [num_images, h_patches, w_patches, num_features]
        labels: Labels tensor [num_images]
        img_height, img_width: Image dimensions
        n_patches_h, n_patches_w: Number of patches in each dimension
        feature_weights: Weights for different features
        n_patches: Total number of patches
        max_possible_rules: Maximum number of rules allowed

    Returns:
        tuple: (fitness_score,)
    """
    
    try:
        # Convert individual to tensor-compatible format
        individual_len = len(individual) if hasattr(individual, '__len__') else 1
        
        # Convert individual to tensor representation for TensorFlow operations
        # This assumes you have a function to convert individual rules to tensor format
        individual_tensor = convert_individual_to_tensor(individual, max_possible_rules)
        
        # Call the compiled TensorFlow function
        fitness = _compute_fitness_core(
            precomputed_features, labels, img_height, img_width,
            n_patches_h, n_patches_w, feature_weights, n_patches,
            individual_len, max_possible_rules, individual_tensor
        )
        
        return (fitness,)
        
    except Exception as e:
        tf.print("Error in individual evaluation:", e)
        return (0.0,)


def convert_individual_to_tensor(individual, max_possible_rules):
    """
    Convert individual rules to tensor format for TensorFlow operations.
    
    Args:
        individual: List of rule dictionaries with keys: 'feature', 'threshold', 'operator', 'action'
        max_possible_rules: Maximum number of rules to pad to
        
    Returns:
        tf.Tensor: Tensor of shape [max_possible_rules, 4] where each row contains:
                  [feature_idx, threshold, operator_code, action]
                  - feature_idx: Index of feature in feature_weights keys
                  - threshold: Threshold value
                  - operator_code: 0 for '>', 1 for '<'
                  - action: 0 for exclude, 1 for include
    """
    # Feature name to index mapping
    feature_names = list(global_config.feature_weights.keys())
    feature_name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    
    # Initialize tensor with invalid values (-1 for feature_idx indicates unused rule)
    rule_tensor = tf.constant(-1.0, shape=(max_possible_rules, 4), dtype=tf.float32)
    
    if not individual or len(individual) == 0:
        return rule_tensor
    
    # Convert individual rules to tensor format
    rule_data = []
    for i, rule in enumerate(individual):
        if i >= max_possible_rules:
            break
            
        # Get feature index
        feature_name = rule.get('feature', '')
        feature_idx = feature_name_to_idx.get(feature_name, -1)
        
        # Get threshold
        threshold = float(rule.get('threshold', 0.0))
        
        # Convert operator to code
        operator = rule.get('operator', '>')
        operator_code = 0.0 if operator == '>' else 1.0
        
        # Get action
        action = float(rule.get('action', 1))
        
        rule_data.append([float(feature_idx), threshold, operator_code, action])
    
    # Convert to tensor and update the initialized tensor
    if rule_data:
        rule_data_tensor = tf.constant(rule_data, dtype=tf.float32)
        num_rules = len(rule_data)
        
        # Create indices for scatter_nd_update
        indices = tf.reshape(tf.range(num_rules), (-1, 1))
        
        # Use tensor scatter update to fill in the rules
        rule_tensor = tf.tensor_scatter_nd_update(
            rule_tensor, 
            indices, 
            rule_data_tensor
        )
    
    return rule_tensor


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
def calculate_connectivity(patch_mask):
    """
    TensorFlow-native connectivity calculation that stays in the graph.
    
    Args:
        patch_mask: Binary mask of shape (n_patches_h, n_patches_w)

    Returns:
        tf.Tensor: Connectivity score between 0 and 1
    """
    # Ensure patch_mask is float32
    patch_mask = tf.cast(patch_mask, tf.float32)

    # If no patches selected, return 0
    total_selected = tf.reduce_sum(patch_mask)
    if_no_patches = tf.equal(total_selected, 0.0)

    def calculate_connectivity_core():
        # Pad the mask to handle edges
        padded_mask = tf.pad(patch_mask, [[1, 1], [1, 1]], constant_values=0.0)
        
        # Extract shifted versions (up, down, left, right)
        up = padded_mask[:-2, 1:-1]
        down = padded_mask[2:, 1:-1]
        left = padded_mask[1:-1, :-2]
        right = padded_mask[1:-1, 2:]
        
        # Count adjacent selected patches
        mask_binary = tf.cast(patch_mask > 0, tf.float32)
        adjacent_count = tf.reduce_sum(
            mask_binary * tf.cast(up > 0, tf.float32) +
            mask_binary * tf.cast(down > 0, tf.float32) +
            mask_binary * tf.cast(left > 0, tf.float32) +
            mask_binary * tf.cast(right > 0, tf.float32)
        )
        
        # Normalize by maximum possible adjacencies
        max_possible_adj = total_selected * 4.0
        connectivity = tf.cond(
            max_possible_adj > 0,
            lambda: adjacent_count / max_possible_adj,
            lambda: 0.5
        )
        
        return connectivity
    
    return tf.cond(if_no_patches, lambda: 0.0, calculate_connectivity_core)

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