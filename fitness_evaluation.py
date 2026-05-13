import tensorflow as tf
import metrics

@tf.function(reduce_retracing=True)
def generate_dynamic_mask(patch_features, rule_tensor):
    """
    Fully vectorized mask generation for a batch of images.
    patch_features: [Batch, H, W, F]
    rule_tensor: [Rules, 4] (feature_idx, threshold, operator, action)
    """
    # Extract rule components from tensor
    feature_indices = tf.cast(rule_tensor[:, 0], tf.int32)
    thresholds = rule_tensor[:, 1]
    operators = tf.cast(rule_tensor[:, 2], tf.int32)
    actions = tf.cast(rule_tensor[:, 3], tf.int32)

    # Filter rules by action: 1 = Include, 0 = Exclude
    include_mask = tf.logical_and(feature_indices >= 0, actions == 1)
    exclude_mask = tf.logical_and(feature_indices >= 0, actions == 0)
    
    def evaluate_rules(v_mask):
        if not tf.reduce_any(v_mask):
            return tf.zeros(tf.shape(patch_features)[:-1], dtype=tf.bool)
        
        v_indices = tf.boolean_mask(feature_indices, v_mask)
        v_thresholds = tf.boolean_mask(thresholds, v_mask)
        v_operators = tf.boolean_mask(operators, v_mask)
        
        max_f = tf.shape(patch_features)[-1] - 1
        v_indices = tf.clip_by_value(v_indices, 0, max_f)
        selected_features = tf.gather(patch_features, v_indices, axis=-1)
        
        # operator == 0 -> '>', operator == 1 -> '<'
        is_greater = tf.logical_and(tf.equal(v_operators, 0), selected_features > v_thresholds)
        is_less    = tf.logical_and(tf.equal(v_operators, 1), selected_features < v_thresholds)
        
        rule_results = tf.logical_or(is_greater, is_less)
        return tf.reduce_any(rule_results, axis=-1)

    # Final mask = (Any Include Rule matches) AND NOT (Any Exclude Rule matches)
    final_include = evaluate_rules(include_mask)
    final_exclude = evaluate_rules(exclude_mask)
    
    final_mask = tf.logical_and(final_include, tf.logical_not(final_exclude))
    
    return tf.cast(final_mask, tf.int8)


@tf.function
def calculate_connectivity(patch_mask):
    """
    Fully vectorized connectivity calculation for a batch of images.
    patch_mask: [Batch, H, W]
    """
    patch_mask_f = tf.cast(patch_mask, tf.float32)
    total_selected = tf.reduce_sum(patch_mask_f, axis=[-2, -1])
    
    kernel = tf.constant([
        [0., 1., 0.],
        [1., 0., 1.],
        [0., 1., 0.]
    ], dtype=tf.float32)
    kernel = tf.reshape(kernel, [3, 3, 1, 1])
    
    # mask_4d: [Batch, H, W, 1]
    mask_4d = tf.expand_dims(patch_mask_f, -1)
    neighbor_count = tf.nn.conv2d(mask_4d, kernel, strides=[1,1,1,1], padding='SAME')
    neighbor_count = tf.squeeze(neighbor_count, axis=-1)
    
    valid_neighbor_count = patch_mask_f * neighbor_count
    total_connections = tf.reduce_sum(valid_neighbor_count, axis=[-2, -1])
    
    # Each selected patch has at most 4 neighbors
    max_connections = total_selected * 4.0
    
    # Safe division: 0.0 connectivity if no patches selected
    connectivity = total_connections / (max_connections + 1e-8)
    
    return connectivity


@tf.function
def compute_batch_divergence_score(masked_feature_vectors, labels, feature_weights):
    """
    Compute a divergence score measuring how well the selected patches separate
    AI-generated images from Human-generated images across the batch.
    """
    eps = tf.constant(1e-8, dtype=tf.float32)

    ai_mask    = tf.cast(labels > 0.5, tf.bool)
    human_mask = tf.logical_not(ai_mask)

    ai_features    = tf.boolean_mask(masked_feature_vectors, ai_mask)
    human_features = tf.boolean_mask(masked_feature_vectors, human_mask)

    n_ai    = tf.shape(ai_features)[0]
    n_human = tf.shape(human_features)[0]

    def _compute():
        mu_ai    = tf.reduce_mean(ai_features,    axis=0)
        mu_human = tf.reduce_mean(human_features, axis=0)
        var_ai   = tf.math.reduce_variance(ai_features,    axis=0)
        var_human = tf.math.reduce_variance(human_features, axis=0)

        # 1. Fisher Discriminant Ratio (FDR)
        fdr = tf.square(mu_ai - mu_human) / (var_ai + var_human + eps)
        
        # 2. Bhattacharyya Distance (Gaussian approximation)
        var_avg  = (var_ai + var_human) / 2.0
        # Use log(det) approximation for univariate channels: log(std_avg / sqrt(std1*std2))
        # Simplified for per-channel: 0.5 * log(var_avg / sqrt(var1*var2))
        bc_shape     = 0.5 * tf.math.log((var_avg + eps) / (tf.sqrt(var_ai * var_human + eps) + eps))
        bc_mahal     = 0.25 * tf.square(mu_ai - mu_human) / (var_ai + var_human + eps)
        bhattacharyya = tf.maximum(bc_shape + bc_mahal, 0.0)

        fdr_norm = 1.0 - tf.exp(-fdr)
        bc_norm  = 1.0 - tf.exp(-bhattacharyya)

        composite = 0.6 * fdr_norm + 0.4 * bc_norm

        num_f = tf.minimum(tf.shape(feature_weights)[0], tf.shape(composite)[0])
        fw    = feature_weights[:num_f]
        comp  = composite[:num_f]

        weight_sum = tf.reduce_sum(fw) + eps
        divergence = tf.reduce_sum(comp * fw) / weight_sum

        return tf.clip_by_value(divergence, 0.0, 1.0)

    return tf.cond(
        tf.logical_or(tf.equal(n_ai, 0), tf.equal(n_human, 0)),
        lambda: tf.constant(0.0, dtype=tf.float32),
        _compute,
    )


@tf.function
def compute_image_scores(precomputed_features, feature_weights, individual_rule_tensor):
    """
    Process all images in a single vectorized pass.
    """
    # 1. Generate masks for the entire batch at once
    # [Batch, H, W]
    patch_mask = generate_dynamic_mask(precomputed_features, individual_rule_tensor)
    
    # 2. Calculate connectivity for the entire batch
    # [Batch]
    connectivity_scores = calculate_connectivity(patch_mask)
    
    # 3. Calculate active patch counts
    # [Batch]
    active_patches_per_image = tf.reduce_sum(tf.cast(patch_mask, tf.int32), axis=[-2, -1])
    
    # 4. Calculate mean feature vectors for the batch
    # [Batch, H, W, 1]
    patch_mask_f = tf.expand_dims(tf.cast(patch_mask, tf.float32), axis=-1)
    # [Batch, H, W, F]
    masked_features = precomputed_features * patch_mask_f
    
    active_f = tf.cast(active_patches_per_image, tf.float32)
    # [Batch, F]
    sum_features = tf.reduce_sum(masked_features, axis=[1, 2])
    masked_feature_vectors = sum_features / (tf.expand_dims(active_f, -1) + 1e-8)
    masked_feature_vectors = tf.clip_by_value(masked_feature_vectors, 0.0, 1.0)

    return masked_feature_vectors, active_patches_per_image, connectivity_scores


@tf.function(reduce_retracing=True)
def evaluate_ga_individual(precomputed_features, labels, feature_weights, 
                           individual_rule_tensor, n_patches_h, n_patches_w,
                           fitness_weights, max_possible_rules):
    """
    Vectorized evaluation of a single GA individual.
    """
    # --- Image Scoring ---
    masked_feature_vectors, active_patches, connectivity_scores = compute_image_scores(
        precomputed_features, feature_weights, individual_rule_tensor
    )

    # --- Component Scores ---
    # 1. Divergence
    divergence = compute_batch_divergence_score(masked_feature_vectors, labels, feature_weights)

    # 2. Efficiency (Penalty for 0 or all patches)
    n_total_patches = tf.cast(n_patches_h * n_patches_w, tf.float32)
    mean_active = tf.reduce_mean(tf.cast(active_patches, tf.float32))
    # Penalty if mean_active is near 0 or near n_total
    efficiency = 1.0 - tf.abs((mean_active / (n_total_patches + 1e-8)) - 0.5) * 2.0
    efficiency = tf.maximum(efficiency, 0.0)

    # 3. Connectivity
    connectivity = tf.reduce_mean(connectivity_scores)

    # 4. Simplicity
    active_rules = tf.reduce_sum(tf.cast(individual_rule_tensor[:, 0] >= 0, tf.float32))
    simplicity = 1.0 - (active_rules / tf.cast(max_possible_rules, tf.float32))

    # --- Weighted Fitness ---
    total_fitness = (
        divergence * fitness_weights['divergence_score'] +
        efficiency * fitness_weights['efficiency_score'] +
        connectivity * fitness_weights['connectivity_score'] +
        simplicity * fitness_weights['simplicity_score']
    )

    return total_fitness, divergence, efficiency, connectivity, simplicity


@tf.function(reduce_retracing=True)
def evaluate_ga_population(precomputed_features, labels, feature_weights,
                           rules_tensors, num_active_rules_tensors,
                           n_patches_h, n_patches_w, fitness_weights,
                           max_possible_rules):
    """
    Map over the population to evaluate each individual.
    Since compute_image_scores is now fully vectorized, we can use a simple map_fn.
    """
    def eval_individual(args):
        rules, num_active = args
        return evaluate_ga_individual(
            precomputed_features, labels, feature_weights,
            rules, n_patches_h, n_patches_w,
            fitness_weights, max_possible_rules
        )

    # We use a small parallel_iterations here because each individual evaluation
    # is now a massive vectorized op on 5000 images. 
    # Too much parallelism here would hit VRAM limits.
    fitness_tuples = tf.map_fn(
        eval_individual,
        (rules_tensors, num_active_rules_tensors),
        fn_output_signature=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
        parallel_iterations=4,  
    )

    return fitness_tuples