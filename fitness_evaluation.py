# fitness_evaluation.py
import tensorflow as tf
from utils import generate_dynamic_mask

@tf.function
def calculate_connectivity(patch_mask):
    """
    Connectivity calculation using convolution.
    """
    patch_mask = tf.cast(patch_mask, tf.float32)
    total_selected = tf.reduce_sum(patch_mask)
    
    if tf.equal(total_selected, 0.0):
        return 0.0
    
    # Use convolution to count neighbors
    kernel = tf.constant([
        [0., 1., 0.],
        [1., 0., 1.],
        [0., 1., 0.]
    ], dtype=tf.float32)
    kernel = tf.reshape(kernel, [3, 3, 1, 1])
    
    mask_4d = tf.expand_dims(tf.expand_dims(patch_mask, 0), -1)
    neighbor_count = tf.nn.conv2d(mask_4d, kernel, strides=[1,1,1,1], padding='SAME')
    neighbor_count = tf.squeeze(neighbor_count)
    
    valid_neighbor_count = patch_mask * neighbor_count
    total_connections = tf.reduce_sum(valid_neighbor_count)
    
    # Get dimensions
    shape = tf.shape(patch_mask)
    n_patches_h = tf.cast(shape[0], tf.float32)
    n_patches_w = tf.cast(shape[1], tf.float32)
    
    # Calculate maximum possible connections with boundary adjustment
    max_connections = total_selected * 4.0 - 2.0 * (n_patches_h + n_patches_w - 2.0)
    
    # Ensure non-negative before division
    max_connections = tf.maximum(max_connections, total_selected * 0.1)
    
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
def compute_fitness_score(config, precomputed_features, labels,
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


def evaluate_ga_individual(individual, config, precomputed_features, labels, 
                                n_patches_h, n_patches_w,
                                feature_weights, n_patches, max_possible_rules):
    """
    Evaluate a single individual in the genetic algorithm.
    """
    try:
        fitness = compute_fitness_score(
            config, precomputed_features, labels, n_patches_h, n_patches_w, feature_weights, n_patches,
            individual.num_active_rules, max_possible_rules, individual.rules_tensor
        )
        
        return (fitness,)
        
    except Exception as e:
        if config.verbose:
            tf.print("Error in individual evaluation:", str(e))
        return (0.0,)