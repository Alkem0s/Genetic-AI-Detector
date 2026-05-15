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
    
    # 1. Compute for ALL rules to keep shapes static
    max_f = tf.shape(patch_features)[-1] - 1
    safe_indices = tf.clip_by_value(feature_indices, 0, max_f)
    
    # [Batch, H, W, Rules]
    selected_features = tf.gather(patch_features, safe_indices, axis=-1)
    
    # Broadcast thresholds and operators
    is_greater = tf.logical_and(tf.equal(operators, 0), selected_features > thresholds)
    is_less    = tf.logical_and(tf.equal(operators, 1), selected_features < thresholds)
    
    # rule_results: [Batch, H, W, Rules]
    rule_results = tf.logical_or(is_greater, is_less)
    
    def evaluate_rules(v_mask):
        # v_mask: [Rules] -> reshape to [1, 1, 1, Rules]
        v_mask_reshaped = tf.reshape(v_mask, [1, 1, 1, -1])
        valid_rule_results = tf.logical_and(rule_results, v_mask_reshaped)
        # Reduce over the Rules dimension
        return tf.reduce_any(valid_rule_results, axis=-1)

    # Final mask = (Any Include Rule matches) AND NOT (Any Exclude Rule matches)
    final_include = evaluate_rules(include_mask)
    final_exclude = evaluate_rules(exclude_mask)
    
    final_mask = tf.logical_and(final_include, tf.logical_not(final_exclude))
    
    # --- Anti-Dummy Rule Check ---
    # A rule is considered 'active' if it actually discriminates with significant coverage.
    # It must have between 1% and 99% mean activation across all patches in the batch.
    # rule_results shape: [Batch, H, W, Rules]
    rule_means = tf.reduce_mean(tf.cast(rule_results, tf.float32), axis=[0, 1, 2])
    rule_activity_mask = tf.logical_and(rule_means > 0.01, rule_means < 0.99)
    
    return tf.cast(final_mask, tf.int8), tf.cast(rule_activity_mask, tf.int8)


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

    ai_mask_f    = tf.cast(labels > 0.5, tf.float32)
    human_mask_f = 1.0 - ai_mask_f

    n_ai_f    = tf.reduce_sum(ai_mask_f)
    n_human_f = tf.reduce_sum(human_mask_f)

    # Expand masks for broadcasting: [Batch, 1]
    ai_mask_exp = tf.expand_dims(ai_mask_f, -1)
    human_mask_exp = tf.expand_dims(human_mask_f, -1)

    def _compute():
        # Weighted mean
        mu_ai    = tf.reduce_sum(masked_feature_vectors * ai_mask_exp, axis=0) / (n_ai_f + eps)
        mu_human = tf.reduce_sum(masked_feature_vectors * human_mask_exp, axis=0) / (n_human_f + eps)
        
        # Weighted variance
        diff_ai = (masked_feature_vectors - mu_ai) * ai_mask_exp
        var_ai  = tf.reduce_sum(tf.square(diff_ai), axis=0) / (n_ai_f + eps)
        
        diff_human = (masked_feature_vectors - mu_human) * human_mask_exp
        var_human  = tf.reduce_sum(tf.square(diff_human), axis=0) / (n_human_f + eps)

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
        tf.logical_or(n_ai_f < 0.5, n_human_f < 0.5),
        lambda: tf.constant(0.0, dtype=tf.float32),
        _compute,
    )


@tf.function
def compute_image_scores(precomputed_features, feature_weights, individual_rule_tensor):
    """
    Process all images in a single vectorized pass.
    """
    # 1. Generate masks for the entire batch at once
    # patch_mask: [Batch, H, W], rule_activity: [Rules]
    patch_mask, rule_activity = generate_dynamic_mask(precomputed_features, individual_rule_tensor)
    
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

    return masked_feature_vectors, active_patches_per_image, connectivity_scores, rule_activity


@tf.function(reduce_retracing=True)
def evaluate_ga_individual(precomputed_features, labels, feature_weights, 
                           individual_rule_tensor, n_patches_h, n_patches_w,
                           fitness_weights, max_possible_rules, inactive_penalty,
                           target_sparsity, sparsity_radius):
    """
    Vectorized evaluation of a single GA individual.
    """
    # --- Image Scoring ---
    masked_feature_vectors, active_patches, connectivity_scores, rule_activity = compute_image_scores(
        precomputed_features, feature_weights, individual_rule_tensor
    )

    # --- Component Scores ---
    n_total_patches = tf.cast(n_patches_h * n_patches_w, tf.float32)
    mean_active = tf.reduce_mean(tf.cast(active_patches, tf.float32))
    sparsity = mean_active / (n_total_patches + 1e-8)

    # 1. Divergence with Sparsity Gate (Prevents overfitting to tiny patch samples)
    divergence = compute_batch_divergence_score(masked_feature_vectors, labels, feature_weights)
    
    # If sparsity is below 10%, we scale down the divergence linearly to zero
    sparsity_gate = tf.minimum(sparsity / 0.10, 1.0)
    divergence = divergence * sparsity_gate

    # 2. Efficiency (Non-linear penalty for sparse or dense masks)
    # We want a 'sweet spot' defined by target and radius.
    # We implement a plateau: any sparsity in [target - radius, target + radius] gets a perfect 1.0 efficiency.
    # Outside this range, it drops off quadratically.
    dist = tf.maximum(tf.abs(sparsity - target_sparsity) - sparsity_radius, 0.0)
    efficiency = 1.0 - tf.pow(dist / 0.4, 2.0)
    
    # Hard floor: if we have fewer than 5% patches, we crash the efficiency to near-zero
    # to prevent overfitting to local noise.
    efficiency = tf.where(sparsity < 0.05, efficiency * 0.1, efficiency)
    efficiency = tf.maximum(efficiency, 0.0)

    # 3. Connectivity
    connectivity = tf.reduce_mean(connectivity_scores)

    # 4. Simplicity
    active_rules = tf.reduce_sum(tf.cast(individual_rule_tensor[:, 0] >= 0, tf.float32))
    simplicity = 1.0 - (active_rules / tf.cast(max_possible_rules, tf.float32))

    # --- Weighted Fitness ---
    base_fitness = (
        divergence * fitness_weights['divergence_score'] +
        efficiency * fitness_weights['efficiency_score'] +
        connectivity * fitness_weights['connectivity_score'] +
        simplicity * fitness_weights['simplicity_score']
    )

    # --- Inactive Weight Penalty ---
    # We penalize individuals that ignore features which were given high weights in Phase 1.
    # CRITICAL: We only count a feature as 'utilized' if it was used in an ACTIVE rule.
    # Dummy rules (always True or always False) are ignored.
    
    # Get feature indices for all rules
    rule_feature_indices = tf.cast(individual_rule_tensor[:, 0], tf.int32)
    
    # Filter for rules that are both valid (index >= 0) AND active (have variance)
    valid_rule_mask = rule_feature_indices >= 0
    # rule_activity is now int8, so we check > 0
    effective_rule_mask = tf.logical_and(valid_rule_mask, rule_activity > 0)
    
    # Map those back to the features
    num_features = tf.shape(feature_weights)[0]
    # features_utilized: [num_features]
    features_utilized = tf.reduce_any(
        tf.logical_and(
            tf.equal(
                tf.expand_dims(tf.range(num_features), 1), 
                tf.expand_dims(rule_feature_indices, 0)
            ),
            tf.expand_dims(effective_rule_mask, 0)
        ),
        axis=1
    )
    
    # Sum weights of features NOT utilized by any effective rule
    inactive_weights_sum = tf.reduce_sum(tf.boolean_mask(feature_weights, tf.logical_not(features_utilized)))
    penalty = inactive_weights_sum * inactive_penalty
    
    total_fitness = base_fitness - penalty

    return total_fitness, divergence, efficiency, connectivity, simplicity, base_fitness


@tf.function(jit_compile=True, reduce_retracing=True)
def evaluate_ga_population(precomputed_features, labels, feature_weights,
                           rules_tensors, num_active_rules_tensors,
                           n_patches_h, n_patches_w, fitness_weights,
                           max_possible_rules, inactive_penalty,
                           target_sparsity, sparsity_radius):
    """
    Map over the population to evaluate each individual.
    """
    def eval_individual(args):
        rules, num_active = args
        return evaluate_ga_individual(
            precomputed_features, labels, feature_weights,
            rules, n_patches_h, n_patches_w,
            fitness_weights, max_possible_rules, 
            inactive_penalty, target_sparsity, sparsity_radius
        )

    fitness_tuples = tf.map_fn(
        eval_individual,
        (rules_tensors, num_active_rules_tensors),
        fn_output_signature=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
        parallel_iterations=4,  
    )

    return fitness_tuples