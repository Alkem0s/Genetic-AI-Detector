# fitness_evaluation.py
import tensorflow as tf
import metrics

@tf.function(reduce_retracing=True)
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
    
    valid_mask = tf.logical_or(
        tf.logical_and(feature_indices >= 0, actions == 1),
        tf.logical_and(feature_indices >= 0, actions == 0)
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
    """
    patch_mask = tf.cast(patch_mask, tf.float32)
    total_selected = tf.reduce_sum(patch_mask)
    
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
    
    # Each patch has max 4 neighbors; total_connections counts both directions.
    # connectivity = total_connections / (total_selected * 4) is strictly in [0, 1].
    # Use tf.cond (NOT Python if) so the guard executes at graph runtime.
    max_connections = total_selected * 4.0
    
    connectivity = tf.cond(
        total_selected > 0,
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
def calculate_honest_threshold(scores):
    """
    Compute an honest decision threshold without peeking at labels.

    Uses the MEDIAN of the score distribution.  For a balanced dataset this
    is equivalent to minimising balanced-accuracy error without needing labels:
    exactly half the samples fall above and half below, so the GA is forced to
    actually RANK AI above Real (or vice-versa) rather than trivially predicting
    a constant class.

    A fixed 0.5 threshold was previously used, but it fails whenever the score
    distribution is far from 0.5 (e.g. when mask sparsity is low and scores
    cluster near 0).
    """
    sorted_scores = tf.sort(scores)           # ascending
    n = tf.shape(sorted_scores)[0]
    median = sorted_scores[n // 2]            # 50th-percentile
    return median


@tf.function(reduce_retracing=True)
def compute_fitness_score(image_size, 
                          weight_balanced_acc, weight_f1, 
                          weight_efficiency, weight_connectivity, weight_simplicity,
                          verbose,
                          precomputed_features, labels,
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
    
    # Normalize by ACTIVE PATCHES per image (not image_size^2).
    # Each feature value is already in [0, 1] and weights sum to 1, so
    # score / active_patches gives the mean weighted feature value per
    # selected patch — a true [0, 1] signal regardless of mask sparsity.
    # When no patches are selected the score is 0 (handled by epsilon).
    active_f = tf.cast(active_patches_per_image, tf.float32)
    normalized_scores = scores / (active_f + 1e-8)
    # Clamp to [0, 1] as a safety net (weights are normalised but numerical
    # drift can occasionally push the value slightly above 1).
    normalized_scores = tf.clip_by_value(normalized_scores, 0.0, 1.0)
    
    # Calculate stats for debug (honestly, from the scores themselves)
    score_mean = tf.reduce_mean(normalized_scores)
    score_std = tf.math.reduce_std(normalized_scores)

    # Calculate honest threshold
    threshold = calculate_honest_threshold(normalized_scores)
    
    # Generate predictions
    predictions = tf.cast(normalized_scores > threshold, tf.int64)
    
    # Calculate metrics (optimized versions)
    precision, recall, f1 = metrics.calculate_precision_recall_f1(labels, predictions)
    balanced_accuracy = metrics.calculate_balanced_accuracy(labels, predictions)
    
    # Calculate efficiency and other scores
    total_active_patches = tf.reduce_sum(active_patches_per_image)
    avg_active_ratio = tf.cast(total_active_patches, tf.float32) / tf.cast(num_samples * n_patches, tf.float32)
    
    # --- QUALITY METRICS (Efficiency, Simplicity, Connectivity) ---
    
    # Efficiency score (Targeted Coverage)
    # We want around 10-30% of the image selected. 
    # CRITICAL: We must penalise 0% coverage (Lazy GA) and 100% coverage (Greedy GA).
    target_coverage = 0.2
    # Use a Gaussian-like penalty centered at target_coverage
    # Peak is 1.0 at 0.2, drops toward 0 at the extremes.
    efficiency_score = tf.exp(-tf.square(avg_active_ratio - target_coverage) / 0.05)
    
    # Add a hard floor: if coverage is extremely low (<1%), kill the efficiency score.
    efficiency_score = tf.where(avg_active_ratio < 0.01, 0.0, efficiency_score)
    
    # Connectivity score (measures how grouped the patches are)
    connectivity_score = tf.reduce_mean(connectivity_scores)
    
    # Simplicity score (penalty for having too many rules)
    # We want to encourage simpler, more generalisable rule sets.
    simplicity_score = 1.0 - (tf.cast(individual_len, tf.float32) / tf.cast(max_possible_rules, tf.float32))
    
    # --- TOTAL FITNESS CALCULATION ---
    # Combine everything using config weights
    fitness = (balanced_accuracy * weight_balanced_acc + 
               f1 * weight_f1 + 
               efficiency_score * weight_efficiency + 
               connectivity_score * weight_connectivity + 
               simplicity_score * weight_simplicity)
    
    # --- EMPTY MASK PENALTY ---
    # If the GA fails to select ANY patches for an image, it's effectively "guessing".
    # We penalise the fitness based on the % of images with empty masks.
    empty_masks = tf.reduce_sum(tf.cast(tf.equal(active_patches_per_image, 0), tf.float32))
    num_samples_f = tf.cast(num_samples, tf.float32)
    empty_mask_ratio = empty_masks / num_samples_f
    # Subtract up to 0.5 from fitness if all masks are empty
    fitness = fitness - (empty_mask_ratio * 0.5)
    
    # Clip fitness to [0, 1] range as a final safety measure
    fitness = tf.clip_by_value(fitness, 0.0, 1.0)
    
    # Debug information
    def print_debug():
        tf.print("=== Fitness Debug ===")
        tf.print("Scores mean/std:", score_mean, score_std)
        tf.print("Threshold:", threshold)
        tf.print("Predictions (AI/Total):", tf.reduce_sum(tf.cast(predictions, tf.float32)), "/", num_samples)
        tf.print("Balanced Acc:", balanced_accuracy)
        tf.print("F1 Score:    ", f1)
        tf.print("Efficiency score:", efficiency_score)
        tf.print("Simplicity score:", simplicity_score)
        tf.print("Connectivity score:", connectivity_score)
        tf.print("TOTAL FITNESS:     ", fitness)
        tf.print("==================")
        return tf.constant(0)


    tf.cond(verbose, print_debug, lambda: tf.constant(0))
    
    return fitness


@tf.function(reduce_retracing=True)
def evaluate_ga_population(rules_tensors, num_active_rules_tensors,
                            image_size_tf, weight_bal_acc, weight_f1, weight_eff, weight_conn, weight_simp,
                            verbose_tf, precomputed_features, labels, n_patches_h, n_patches_w,
                            feature_weights, n_patches, max_possible_rules):
    """
    Evaluate an entire population of individuals in parallel on the GPU.
    Uses nested vectorization to score all individuals across all images in one shot.
    """
    def eval_individual(args):
        rules, num_active = args
        return compute_fitness_score(
            image_size_tf, weight_bal_acc, weight_f1, weight_eff, weight_conn, weight_simp, verbose_tf,
            precomputed_features, labels, n_patches_h, n_patches_w, feature_weights, n_patches,
            num_active, max_possible_rules, rules
        )
    
    # Use map_fn to keep the GPU saturated while avoiding OOM.
    fitnesses = tf.map_fn(
        eval_individual, 
        (rules_tensors, num_active_rules_tensors),
        fn_output_signature=tf.float32,
        parallel_iterations=10
    )
    return fitnesses


def evaluate_ga_individual(individual, config, precomputed_features, labels, 
                                n_patches_h, n_patches_w,
                                feature_weights, n_patches, max_possible_rules):
    """
    Evaluate a single individual in the genetic algorithm.
    """
    try:
        image_size_tf = tf.constant(config.image_size, dtype=tf.float32)
        weight_bal_acc = tf.constant(config.fitness_weights['balanced_accuracy'], dtype=tf.float32)
        weight_f1 = tf.constant(config.fitness_weights['f1'], dtype=tf.float32)
        weight_eff = tf.constant(config.fitness_weights['efficiency_score'], dtype=tf.float32)
        weight_conn = tf.constant(config.fitness_weights['connectivity_score'], dtype=tf.float32)
        weight_simp = tf.constant(config.fitness_weights['simplicity_score'], dtype=tf.float32)
        verbose_tf = tf.constant(bool(config.verbose), dtype=tf.bool)
        
        fitness = compute_fitness_score(
            image_size_tf, weight_bal_acc, weight_f1, weight_eff, weight_conn, weight_simp, verbose_tf,
            precomputed_features, labels, n_patches_h, n_patches_w, feature_weights, n_patches,
            individual.num_active_rules, max_possible_rules, individual.rules_tensor
        )
        
        return (fitness,)
        
    except Exception as e:
        if config.verbose:
            tf.print("Error in individual evaluation:", str(e))
        return (0.0,)