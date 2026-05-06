import sys
import os
import numpy as np
import random
import time
from deap import base, creator, tools
from feature_extractor import FeatureExtractor
from fitness_evaluation import generate_dynamic_mask, evaluate_ga_individual, compute_fitness_score
import logging
import json
import tensorflow as tf

logger = logging.getLogger(__name__)

class GeneticFeatureOptimizer:
    """
    Class for optimizing feature selection in AI image detection using genetic algorithms.
    Uses DEAP library to evolve optimal feature-based conditional rules that generate
    dynamic masks per image based on extracted features.
    Rules are stored as tensors from the start to avoid conversion overhead.
    
    Features are precomputed immediately upon initialization and reused across multiple runs.
    """

    def __init__(self, images, labels, config):
        """
        Initialize the genetic algorithm optimizer.
        Features are precomputed immediately after initialization.

        Args:
            images: Numpy array of preprocessed images
            labels: Numpy array of labels (0 for human, 1 for AI)
            config: A dictionary or object containing configuration parameters.
        """
        logger.info("Initializing GeneticFeatureOptimizer...")
        
        # Keep raw images on CPU (Numpy) to save VRAM
        # We only upload batches to GPU during _precompute_features
        self.images = images 
        self.labels = tf.convert_to_tensor(labels, dtype=tf.float32) if not isinstance(labels, tf.Tensor) else labels

        self.config = config
        self.patch_size = config.patch_size
        self.batch_size = config.extraction_batch_size
        self.population_size = config.population_size
        self.n_generations = config.n_generations
        self.crossover_prob = config.crossover_prob
        self.mutation_prob = config.mutation_prob
        self.tournament_size = config.tournament_size
        self.num_elites = config.num_elites

        # Use numpy shape to avoid triggering a full GPU copy
        image_count = images.shape[0] if hasattr(images, 'shape') else len(images)
        self.eval_sample_size = min(config.sample_size, image_count)
        logger.info(f"Using evaluation sample size: {self.eval_sample_size}")

        self.rules_per_individual = config.rules_per_individual
        self.max_possible_rules = config.max_possible_rules
        self.random_seed = config.random_seed
        self.verbose = config.verbose

        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
            tf.random.set_seed(self.random_seed)
            logger.info(f"Random seed set to {self.random_seed}")

        # Pre-calculate constants for evaluation
        self.image_size_tf = tf.constant(config.image_size, dtype=tf.float32)
        self.weight_f1 = tf.constant(config.fitness_weights.get('f1', 0.35), dtype=tf.float32)
        self.weight_bal_acc = tf.constant(config.fitness_weights.get('balanced_accuracy', 0.35), dtype=tf.float32)
        self.weight_eff = tf.constant(config.fitness_weights.get('efficiency_score', 0.1), dtype=tf.float32)
        self.weight_conn = tf.constant(config.fitness_weights.get('connectivity_score', 0.1), dtype=tf.float32)
        self.weight_simp = tf.constant(config.fitness_weights.get('simplicity_score', 0.1), dtype=tf.float32)
        self.verbose_tf = tf.constant(bool(config.verbose), dtype=tf.bool)
        
        self.feature_weights = tf.convert_to_tensor(list(config.feature_weights.values()), dtype=tf.float32)
        self.feature_names = list(config.feature_weights.keys())
        self.num_features = len(self.feature_names)

        # Create feature name to index mapping for tensor operations
        self.feature_name_to_idx = {name: idx for idx, name in enumerate(self.feature_names)}

        self.n_patches_h = tf.constant(config.image_size // self.patch_size, dtype=tf.int32)
        self.n_patches_w = tf.constant(config.image_size // self.patch_size, dtype=tf.int32)
        self.n_patches = self.n_patches_h * self.n_patches_w
        
        self.n_patches_tf = tf.constant(self.n_patches, dtype=tf.int32)
        self.max_rules_tf = tf.constant(self.max_possible_rules, dtype=tf.int32)

        logger.info(f"Initialized with {self.n_patches} patches ({self.n_patches_h}×{self.n_patches_w})")

        self.feature_extractor = FeatureExtractor(config)

        # Initialize precomputed features storage
        self.precomputed_features = None
        self.eval_labels = None
        self.features_computed = False
        self.feature_scales = None

        # Run history for multiple runs
        self.all_run_histories = []
        self.current_run_history = []

        self._setup_deap()
        
        # Precompute features immediately after initialization
        self._precompute_features()
        
        logger.info("GeneticFeatureOptimizer initialization complete with precomputed features.")

    def _precompute_features(self):
        """
        Precompute patch features for the evaluation dataset once.
        This eliminates redundant feature extraction across all individuals and runs.
        Can be called multiple times safely - will skip if already computed.
        """
        if self.features_computed:
            logger.info("Features already precomputed, skipping...")
            return
            
        logger.info("Precomputing patch features for evaluation dataset...")
        start_time = time.time()
        
        num_images_in_input = self.images.shape[0] if hasattr(self.images, 'shape') else len(self.images)
        
        # Determine the actual sample to use for evaluation
        if self.eval_sample_size < num_images_in_input:
            # Use consistent sampling across runs by using the same seed for sampling
            if self.random_seed is not None:
                np.random.seed(self.random_seed)
            indices = np.random.choice(num_images_in_input, self.eval_sample_size, replace=False)
            indices_tensor = tf.convert_to_tensor(indices, dtype=tf.int32)
            if hasattr(self.images, 'shape'):
                sample_images = self.images[indices]
            else:
                sample_images = [self.images[i] for i in indices]
            self.eval_labels = tf.gather(self.labels, indices_tensor)
        else:
            sample_images = self.images
            self.eval_labels = self.labels

        num_samples = sample_images.shape[0] if hasattr(sample_images, 'shape') else len(sample_images)
        logger.info(f"Extracting features for {num_samples} evaluation images...")

        # Extract features for all images in batches
        started_profiler = False
        if self.config.profile:
            profile_dir = os.path.join(self.config.profile_log_dir, 'feature_extraction')
            tf.profiler.experimental.start(profile_dir)
            started_profiler = True
            logger.info(f"Feature extraction profiling started. Logging to {profile_dir}")

        all_batch_features = []
        for i in range(0, num_samples, self.batch_size):
            batch_end = min(i + self.batch_size, num_samples)
            batch_images = sample_images[i:batch_end]
            
            # Extract features once per batch
            with tf.profiler.experimental.Trace('batch_extraction', step_num=i):
                batch_patch_features = self.feature_extractor.extract_batch_patch_features(batch_images)
            all_batch_features.append(batch_patch_features)
            
            logger.info(f"Processed batch {i//self.batch_size + 1}/{(num_samples + self.batch_size - 1)//self.batch_size}")
            
        if started_profiler:
            tf.profiler.experimental.stop()
            logger.info("Feature extraction profiling stopped.")

        # Concatenate all batch features into a single tensor
        raw_features = tf.concat(all_batch_features, axis=0)
        
        # --- Automated Feature Scaling (Dynamic Normalization) ---
        # We calculate the 99th percentile for each feature to use as a scaling divisor.
        # This maps the vast majority of 'AI signals' into the [0, 1] range.
        logger.info("Auto-calibrating feature scales (Dynamic Normalization)...")
        
        # Flatten across batch and spatial dims to get [total_patches, num_features]
        total_patches = tf.shape(raw_features)[0] * tf.shape(raw_features)[1] * tf.shape(raw_features)[2]
        flat_raw = tf.reshape(raw_features, [total_patches, -1])
        
        # Only calculate scales if they don't exist (i.e., during initial training setup)
        if self.feature_scales is None:
            logger.info("Auto-calibrating feature scales (Dynamic Normalization)...")
            # Move to numpy for percentile calculation
            flat_raw_np = flat_raw.numpy()
            scales = np.percentile(np.abs(flat_raw_np), 99, axis=0)
            
            # Avoid division by zero
            scales = np.maximum(scales, 1e-6)
            self.feature_scales = tf.convert_to_tensor(scales, dtype=tf.float32)
            logger.debug(f"Calculated feature scales: {dict(zip(self.feature_names, scales.tolist()))}")
        else:
            logger.info("Using existing feature scales for normalization.")
        
        # Apply scaling: [batch, h, w, features] / [features]
        self.precomputed_features = raw_features / self.feature_scales
        self.features_computed = True
        
        end_time = time.time()
        logger.info(f"Feature precomputation and auto-calibration complete in {end_time - start_time:.2f} seconds")
        logger.info(f"Final normalized features shape: {self.precomputed_features.shape}")
        logger.debug(f"Calculated feature scales: {dict(zip(self.feature_names, scales))}")

    def recompute_features(self, force=False):
        """
        Force recomputation of features. Useful if images or configuration has changed.
        
        Args:
            force (bool): If True, forces recomputation even if features are already computed.
        """
        if force or not self.features_computed:
            logger.info("Forcing feature recomputation...")
            self.features_computed = False
            self.precomputed_features = None
            self.eval_labels = None
            self._precompute_features()
        else:
            logger.info("Features already computed. Use force=True to recompute.")

    def _tensor_rules_similar(self, ind1, ind2):
        """
        Custom similarity function for comparing tensor-based individuals.
        Returns True if individuals are considered similar (same rules).
        """
        try:
            # Compare the rules tensors
            rules1 = ind1.rules_tensor
            rules2 = ind2.rules_tensor
            
            # Check if tensors have the same shape
            if not tf.reduce_all(tf.equal(tf.shape(rules1), tf.shape(rules2))):
                return False
                
            # Check if active rule counts are the same
            if ind1.num_active_rules != ind2.num_active_rules:
                return False
            
            # Compare only the active rules (non -1 entries)
            # Get active rules from both individuals
            active_mask1 = rules1[:, 0] >= 0
            active_mask2 = rules2[:, 0] >= 0
            
            # If different number of active rules, they're different
            if not tf.reduce_all(tf.equal(active_mask1, active_mask2)):
                return False
            
            active_rules1 = tf.boolean_mask(rules1, active_mask1)
            active_rules2 = tf.boolean_mask(rules2, active_mask2)
            
            # Compare active rules with small tolerance for floating point comparison
            tolerance = 1e-6
            rules_close = tf.reduce_all(tf.abs(active_rules1 - active_rules2) < tolerance)
            
            return bool(rules_close.numpy())
            
        except Exception as e:
            logger.warning(f"Error in similarity comparison: {e}")
            # Fallback to False if comparison fails
            return False

    def _setup_deap(self):
        """Set up the DEAP genetic algorithm framework"""
        logger.debug("Setting up DEAP components...")
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            logger.debug("Created 'FitnessMax' in DEAP creator.")

        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
            logger.debug("Created 'Individual' in DEAP creator.")

        self.toolbox = base.Toolbox()

        self.toolbox.register("attr_rule", self._random_rule_tensor)
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        logger.debug("DEAP toolbox registered for rule, individual, and population creation.")

        self.toolbox.register("evaluate", self._evaluate_individual_wrapper)
        self.toolbox.register("mate", self._crossover_tensor_rules)
        self.toolbox.register("mutate", self._mutate_tensor_rules)
        self.toolbox.register("select", self._tensor_tournament_selection)
        logger.debug("DEAP toolbox registered for evaluation, mating, mutation, and selection.")

        self._setup_sequential_processing()
        logger.debug("DEAP setup complete.")

    def _setup_sequential_processing(self):
        """Setup for sequential processing as per user's request to use single GPU."""
        self.use_joblib = False
        self.n_workers = 1
        logger.debug("Configured for sequential evaluation on the main process to leverage GPU.")

    def _create_individual(self):
        """Create an individual with tensor-based rules"""
        # Create a tensor to hold all rules for this individual
        # Rule tensor shape: [max_rules, 4] where 4 = [feature_idx, threshold, operator, action]
        # We'll use -1 to indicate unused rule slots
        rules_tensor = tf.fill([self.max_possible_rules, 4], -1.0)
        
        # Fill with actual rules up to rules_per_individual
        rule_data = []
        for _ in range(self.rules_per_individual):
            feature_idx = tf.random.uniform([], 0, self.num_features, dtype=tf.int32)
            threshold = tf.random.uniform([], 0.0, 1.0, dtype=tf.float32)
            operator = tf.random.uniform([], 0, 2, dtype=tf.int32)  # 0 for '>', 1 for '<'
            action = tf.random.uniform([], 0, 2, dtype=tf.int32)    # 0 or 1
            
            rule_data.append([tf.cast(feature_idx, tf.float32), threshold, 
                            tf.cast(operator, tf.float32), tf.cast(action, tf.float32)])
        
        if rule_data:
            rules_tensor = tf.tensor_scatter_nd_update(
                rules_tensor,
                tf.expand_dims(tf.range(len(rule_data)), 1),
                tf.stack(rule_data)
            )
        
        # Store the number of active rules
        num_active_rules = len(rule_data)
        
        # Create Individual instance that holds the tensor and metadata
        individual = creator.Individual([rules_tensor])
        individual.num_active_rules = num_active_rules
        individual.rules_tensor = rules_tensor
        
        return individual

    def _random_rule_tensor(self):
        """Generate a single random rule as tensor data"""
        feature_idx = tf.random.uniform([], 0, self.num_features, dtype=tf.int32)
        threshold = tf.random.uniform([], 0.0, 1.0, dtype=tf.float32)
        operator = tf.random.uniform([], 0, 2, dtype=tf.int32)  # 0 for '>', 1 for '<'
        action = tf.random.uniform([], 0, 2, dtype=tf.int32)    # 0 or 1
        
        return [tf.cast(feature_idx, tf.float32), threshold, 
                tf.cast(operator, tf.float32), tf.cast(action, tf.float32)]

    def _evaluate_individual_wrapper(self, individual):
        """Wrapper for individual evaluation that uses precomputed features"""
        if self.verbose:
            start_time = time.time()
        
        # Features should already be precomputed at this point
        if not self.features_computed or self.precomputed_features is None:
            logger.error("Features not precomputed!")
            raise RuntimeError("Features must be precomputed before evaluation")
        
        # Pass the tensor directly to the evaluation function
        fitness = evaluate_ga_individual(
            individual, self.config, self.precomputed_features, self.eval_labels, 
            self.n_patches_h, self.n_patches_w,
            self.feature_weights, self.n_patches, self.max_possible_rules,
        )
        
        if self.verbose:
            end_time = time.time()
            logger.debug(f"Individual evaluation took {end_time - start_time:.4f} seconds.")
        return fitness

    def _evaluate_population(self, population):
        """
        Evaluate fitness for the entire population in a single batch on the GPU.
        Only evaluates individuals with invalid fitness.
        """
        # Find individuals that need evaluation
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        
        if not invalid_ind:
            return [ind.fitness.values for ind in population]
            
        # Collect rules and num_active_rules into tensors
        rules_tensors = tf.stack([ind.rules_tensor for ind in invalid_ind])
        num_active_rules_tensors = tf.stack([tf.cast(ind.num_active_rules, tf.int32) for ind in invalid_ind])
        
        from fitness_evaluation import evaluate_ga_population
        
        # Call the batch evaluation function
        batch_fitnesses = evaluate_ga_population(
            rules_tensors, num_active_rules_tensors,
            self.image_size_tf, self.weight_bal_acc, self.weight_f1, self.weight_eff, self.weight_conn, self.weight_simp,
            self.verbose_tf, self.precomputed_features, self.eval_labels,
            self.n_patches_h, self.n_patches_w, self.feature_weights,
            self.n_patches_tf, self.max_rules_tf
        )
        
        # Convert to list of tuples for DEAP
        batch_fitness_list = [(float(f),) for f in batch_fitnesses.numpy()]
        
        # Map back to the population and assign fitness values
        invalid_idx = 0
        final_fitnesses = []
        for ind in population:
            if not ind.fitness.valid:
                ind.fitness.values = batch_fitness_list[invalid_idx]
                final_fitnesses.append(ind.fitness.values)
                invalid_idx += 1
            else:
                final_fitnesses.append(ind.fitness.values)
                
        return final_fitnesses

    def _crossover_tensor_rules(self, ind1, ind2):
        """Crossover operation for tensor-based rules"""
        # Get the rules tensors
        rules1 = ind1.rules_tensor
        rules2 = ind2.rules_tensor
        
        # Simple point crossover - swap some rules between individuals
        crossover_point = tf.random.uniform([], 1, self.max_possible_rules, dtype=tf.int32)
        
        # Create masks for swapping
        mask = tf.cast(tf.range(self.max_possible_rules) < crossover_point, tf.float32)
        mask = tf.expand_dims(mask, 1)
        
        # Perform crossover
        new_rules1 = rules1 * mask + rules2 * (1 - mask)
        new_rules2 = rules2 * mask + rules1 * (1 - mask)
        
        # Update individuals
        ind1.rules_tensor = new_rules1
        ind1[0] = new_rules1
        
        ind2.rules_tensor = new_rules2
        ind2[0] = new_rules2
        
        # Update active rule counts (approximate)
        ind1.num_active_rules = self._count_active_rules(new_rules1)
        ind2.num_active_rules = self._count_active_rules(new_rules2)
        
        return ind1, ind2

    def _mutate_tensor_rules(self, individual):
        """Mutate tensor-based rules"""
        rules = individual.rules_tensor
        
        # Create mutation mask
        mutation_rate = 0.1
        mutation_mask = tf.random.uniform([self.max_possible_rules]) < mutation_rate
        
        # Mutate thresholds
        threshold_noise = tf.random.uniform([self.max_possible_rules], -0.1, 0.1)
        new_thresholds = tf.clip_by_value(rules[:, 1] + threshold_noise, 0.0, 1.0)
        
        # Mutate operators (flip some)
        operator_flip_mask = tf.random.uniform([self.max_possible_rules]) < 0.05
        new_operators = tf.where(operator_flip_mask, 1.0 - rules[:, 2], rules[:, 2])
        
        # Mutate actions (flip some)
        action_flip_mask = tf.random.uniform([self.max_possible_rules]) < 0.05
        new_actions = tf.where(action_flip_mask, 1.0 - rules[:, 3], rules[:, 3])
        
        # Mutate features
        feature_change_mask = tf.random.uniform([self.max_possible_rules]) < 0.05
        new_features = tf.where(
            feature_change_mask,
            tf.cast(tf.random.uniform([self.max_possible_rules], 0, self.num_features, dtype=tf.int32), tf.float32),
            rules[:, 0]
        )
        
        # Apply mutations only where mutation_mask is True
        mutation_mask_expanded = tf.expand_dims(tf.cast(mutation_mask, tf.float32), 1)
        
        new_rules = tf.stack([
            tf.where(mutation_mask, new_features, rules[:, 0]),
            tf.where(mutation_mask, new_thresholds, rules[:, 1]),
            tf.where(mutation_mask, new_operators, rules[:, 2]),
            tf.where(mutation_mask, new_actions, rules[:, 3])
        ], axis=1)
        
        # Update individual
        individual.rules_tensor = new_rules
        individual[0] = new_rules
        
        # Possibly add or remove rules
        if tf.random.uniform([]) < 0.05 and individual.num_active_rules < self.max_possible_rules:
            # Add a new rule
            next_idx = individual.num_active_rules
            if next_idx < self.max_possible_rules:
                new_rule_data = self._random_rule_tensor()
                individual.rules_tensor = tf.tensor_scatter_nd_update(
                    individual.rules_tensor, 
                    [[next_idx]], 
                    [new_rule_data]
                )
                individual.num_active_rules += 1
        
        if tf.random.uniform([]) < 0.05 and individual.num_active_rules > 1:
            # Remove a rule by setting it to -1
            remove_idx = tf.random.uniform([], 0, individual.num_active_rules, dtype=tf.int32)
            individual.rules_tensor = tf.tensor_scatter_nd_update(
                individual.rules_tensor,
                [[remove_idx]],
                [[-1.0, -1.0, -1.0, -1.0]]
            )
            # We don't decrement num_active_rules immediately to keep indexing simple,
            # but we update the rule itself. Better: compact rules.
            # For now just update the tensor.
        
        # Re-sync DEAP list with tensor
        individual[0] = individual.rules_tensor
        
        return individual,

    def _tensor_tournament_selection(self, individuals, k):
        """
        Perform tournament selection using vectorized tensor operations.
        """
        # Get all fitness values as a single tensor
        fitnesses = tf.constant([ind.fitness.values[0] for ind in individuals], dtype=tf.float32)
        pop_size = len(individuals)
        
        # Generate random tournament indices for all k selection slots at once
        tourn_indices = tf.random.uniform(
            [k, self.tournament_size], 
            minval=0, 
            maxval=pop_size, 
            dtype=tf.int32
        )
        
        # Gather fitnesses for each tournament bracket
        bracket_fitnesses = tf.gather(fitnesses, tourn_indices)
        
        # Find the index of the best individual in each bracket
        winners_in_bracket_idx = tf.argmax(bracket_fitnesses, axis=1, output_type=tf.int32)
        
        # Map back to the original population indices
        row_indices = tf.range(k, dtype=tf.int32)
        final_winner_indices = tf.gather_nd(
            tourn_indices, 
            tf.stack([row_indices, winners_in_bracket_idx], axis=1)
        )
        
        # Return the winning individuals
        return [individuals[i] for i in final_winner_indices.numpy()]

    def _count_active_rules(self, rules_tensor):
        """Count the number of active rules in a tensor (non -1 entries)"""
        return tf.reduce_sum(tf.cast(rules_tensor[:, 0] >= 0, tf.int32)).numpy()
    
    def get_feature_importance(self, individual):
        """
        Analyze which features contribute most based on the tensor rule set.
        Uses precomputed features for efficiency.
        """
        logger.debug("Calculating feature importance...")
        feature_scores = {feature: 0 for feature in self.feature_names}
        
        # Extract active rules from tensor
        rules_tensor = individual.rules_tensor
        active_mask = rules_tensor[:, 0] >= 0
        active_rules = tf.boolean_mask(rules_tensor, active_mask)
        
        # Count rule frequency per feature
        if tf.shape(active_rules)[0] > 0:
            feature_indices = tf.cast(active_rules[:, 0], tf.int32)
            for idx in feature_indices:
                feature_name = self.feature_names[idx.numpy()]
                feature_scores[feature_name] += 1
        
        logger.debug(f"Initial rule-based feature scores: {feature_scores}")

        # Use precomputed features
        num_samples = min(50, tf.shape(self.precomputed_features)[0].numpy())
        sample_features = self.precomputed_features[:num_samples]
        logger.debug(f"Analyzing feature triggers on {num_samples} precomputed feature sets.")

        feature_triggers = {feature: 0 for feature in self.feature_names}
        total_patches_evaluated = 0

        # Analyze trigger frequency using tensor operations
        if tf.shape(active_rules)[0] > 0:
            total_patches_evaluated = num_samples * self.n_patches_h.numpy() * self.n_patches_w.numpy()
            
            for rule_idx in range(tf.shape(active_rules)[0]):
                rule = active_rules[rule_idx]
                feature_idx = tf.cast(rule[0], tf.int32)
                threshold = rule[1] 
                operator = tf.cast(rule[2], tf.int32)
                
                if feature_idx < tf.shape(sample_features)[3]:
                    feature_values = sample_features[:, :, :, feature_idx]
                    
                    # Check condition across the whole volume
                    condition_met = tf.cond(
                        tf.equal(operator, 0),
                        lambda: feature_values > threshold,
                        lambda: feature_values < threshold
                    )
                    
                    triggers = tf.reduce_sum(tf.cast(condition_met, tf.int32)).numpy()
                    
                    feature_name = self.feature_names[feature_idx.numpy()]
                    feature_triggers[feature_name] += int(triggers)

        logger.debug(f"Feature triggers: {feature_triggers}. Total patches evaluated: {total_patches_evaluated}")

        # Combine rule frequency and trigger frequency
        for feature in feature_scores:
            rule_freq = feature_scores[feature] / max(individual.num_active_rules, 1)
            trigger_freq = feature_triggers[feature] / max(total_patches_evaluated, 1) if total_patches_evaluated > 0 else 0
            feature_scores[feature] = 0.7 * rule_freq + 0.3 * trigger_freq

        total = sum(feature_scores.values())
        if total > 0:
            for key in feature_scores:
                feature_scores[key] = feature_scores[key] / total
        logger.debug(f"Feature importance calculation complete. Scores: {feature_scores}")

        return feature_scores

    def run(self, run_id=None):
        """
        Execute the genetic algorithm optimization.
        Can be called multiple times, reusing precomputed features.
        
        Args:
            run_id (str, optional): Identifier for this run. If None, uses run number.
        """
        run_number = len(self.all_run_histories) + 1
        if run_id is None:
            run_id = f"run_{run_number}"
            
        logger.debug(f"Starting genetic algorithm run '{run_id}' with population={self.population_size}, generations={self.n_generations}")
        logger.debug(f"Using precomputed features (computed={self.features_computed})")

        # Ensure features are precomputed (should already be done in __init__)
        if not self.features_computed:
            logger.warning("Features not precomputed, this should not happen in refactored version")
            self._precompute_features()

        # Reset current run history
        self.current_run_history = []

        pop = self.toolbox.population(n=self.population_size)
        logger.debug(f"Initial population of {len(pop)} individuals created.")

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        stats.register("std", np.std)
        logger.debug("Statistics registered for DEAP.")

        # Create HallOfFame with custom similarity function
        hof = tools.HallOfFame(self.num_elites, similar=self._tensor_rules_similar)
        logger.debug(f"Hall of Fame initialized with capacity {self.num_elites} and custom similarity function.")

        run_start_time = time.time()

        for gen in range(self.n_generations):
            gen_start_time = time.time()
            try:
                fitnesses = self._evaluate_population(pop)
            except Exception as e:
                logger.error(f"Error in sequential evaluation for generation {gen + 1}: {e}")
                raise
            gen_end_time = time.time()
            logger.debug(f"Generation {gen + 1} evaluation phase took {gen_end_time - gen_start_time:.4f} seconds.")
            
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit
            
            hof.update(pop)
            record = stats.compile(pop)
            
            # Only log generation stats at DEBUG level to keep the console clean during HPO
            logger.debug(f"  [{run_id}] Gen {gen + 1}/{self.n_generations}: Max={record['max']:.4f}, Avg={record['avg']:.4f}")
            
            self.current_run_history.append({
                'generation': gen + 1,
                'max_fitness': record['max'],
                'avg_fitness': record['avg'],
                'std_fitness': record['std'],
                'generation_time': gen_end_time - gen_start_time
            })
            
            if gen < self.n_generations - 1:
                # Select the next generation individuals
                elites = [self.toolbox.clone(ind) for ind in hof]
                
                offspring_size = self.population_size - len(elites)
                offspring = self.toolbox.select(pop, offspring_size)
                offspring = list(map(self.toolbox.clone, offspring))
                
                crossover_start_time = time.time()
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < self.crossover_prob:
                        self.toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values
                crossover_end_time = time.time()
                logger.debug(f"Crossover phase took {crossover_end_time - crossover_start_time:.4f} seconds.")

                mutation_start_time = time.time()
                for mutant in offspring:
                    if random.random() < self.mutation_prob:
                        self.toolbox.mutate(mutant)
                        del mutant.fitness.values
                mutation_end_time = time.time()
                logger.debug(f"Mutation phase took {mutation_end_time - mutation_start_time:.4f} seconds.")

                pop[:] = elites + offspring

        run_end_time = time.time()
        total_run_time = run_end_time - run_start_time

        logger.debug(f"Genetic algorithm run '{run_id}' complete in {total_run_time:.2f} seconds.")

        best_ind = hof[0]

        logger.debug("Best rule set found:")
        active_rules = tf.boolean_mask(best_ind.rules_tensor, best_ind.rules_tensor[:, 0] >= 0)
        for i in range(tf.shape(active_rules)[0]):
            rule = active_rules[i]
            feature_idx = tf.cast(rule[0], tf.int32)
            threshold = rule[1]
            operator = ">" if tf.cast(rule[2], tf.int32) == 0 else "<"
            action = "include" if tf.cast(rule[3], tf.int32) == 1 else "exclude"
            feature_name = self.feature_names[feature_idx.numpy()]
            logger.debug(f"Rule {i+1}: if {feature_name} {operator} {threshold:.2f} → {action} patch")

        # Calculate final statistics using precomputed features
        sample_size = min(20, tf.shape(self.precomputed_features)[0].numpy())
        sample_features = self.precomputed_features[:sample_size]
        logger.debug(f"Calculating average mask statistics on {sample_size} sample images.")

        total_active = tf.constant(0, dtype=tf.float32)

        mask_gen_start_time = time.time()
        for j in range(sample_size):
            patch_features = sample_features[j]
            mask = generate_dynamic_mask(patch_features, self.n_patches_h, self.n_patches_w, best_ind.rules_tensor)
            if not isinstance(mask, tf.Tensor):
                mask = tf.convert_to_tensor(mask, dtype=tf.float32)
            else:
                mask = tf.cast(mask, tf.float32)
            total_active += tf.reduce_sum(mask)
        mask_gen_end_time = time.time()
        logger.debug(f"Mask generation for final statistics took {mask_gen_end_time - mask_gen_start_time:.4f} seconds.")

        avg_active = total_active / tf.cast(sample_size, tf.float32)
        avg_percentage = avg_active / (tf.cast(self.n_patches_h, tf.float32) * tf.cast(self.n_patches_w, tf.float32)) * 100

        # --- Mask sparsity statistics over the *full* evaluation sample ---
        # (patch selection ratio = fraction of patches selected per image)
        total_patches_per_image = tf.cast(self.n_patches_h, tf.float32) * tf.cast(self.n_patches_w, tf.float32)
        full_sample_size = min(
            tf.shape(self.precomputed_features)[0].numpy(),
            self.eval_sample_size,
        )
        full_sample_features = self.precomputed_features[:full_sample_size]
        sparsity_ratios = []
        for j in range(full_sample_size):
            patch_features_j = full_sample_features[j]
            mask_j = generate_dynamic_mask(
                patch_features_j, self.n_patches_h, self.n_patches_w, best_ind.rules_tensor
            )
            ratio = float(
                tf.reduce_sum(tf.cast(mask_j, tf.float32)).numpy() / total_patches_per_image.numpy()
            )
            sparsity_ratios.append(ratio)

        sparsity_ratios_np = np.array(sparsity_ratios, dtype=np.float32)
        mask_sparsity_mean = float(np.mean(sparsity_ratios_np))
        mask_sparsity_std  = float(np.std(sparsity_ratios_np))

        logger.info(
            f"[{run_id}] Mask sparsity: mean={mask_sparsity_mean:.4f}, "
            f"std={mask_sparsity_std:.4f} (over {full_sample_size} images)"
        )

        logger.debug(f"Best fitness: {best_ind.fitness.values[0]:.4f}")
        logger.debug(f"Rules: {best_ind.num_active_rules}")
        logger.debug(f"Average active patches: {avg_active:.1f} ({avg_percentage:.1f}%) out of {self.n_patches}")

        self.history = self.current_run_history

        # Store run results
        run_results = {
            'run_id': run_id,
            'run_number': run_number,
            'best_individual': best_ind,
            'best_fitness': best_ind.fitness.values[0],
            'best_rules_tensor': best_ind.rules_tensor.numpy().tolist(),
            'best_feature_importance': self.get_feature_importance(best_ind),
            'rule_count': best_ind.num_active_rules,
            'avg_active_patches': avg_active.numpy(),
            'avg_active_percentage': avg_percentage.numpy(),
            # Sparsity fields used by RandomMaskGenerator
            'mask_sparsity_mean': mask_sparsity_mean,
            'mask_sparsity_std':  mask_sparsity_std,
            'history': self.history,
            'total_run_time': total_run_time
        }
        
        self.all_run_histories.append(run_results)
        return run_results

    def validate(self, images, labels, individual):
        """
        Evaluate a single individual on a new set of images (Validation).
        This is used for 'Honest' evaluation on unseen data.
        
        Args:
            images: New set of images
            labels: Labels for the new images
            individual: The individual (rule set) to evaluate
            
        Returns:
            Validation fitness score
        """
        logger.info(f"Starting 'Honest' validation on {len(images)} unseen images...")
        
        # 1. Precompute features for the validation set (one-time)
        # We temporarily swap out images/labels to use the existing _precompute_features
        old_images = self.images
        old_labels = self.labels
        old_features = self.precomputed_features
        
        self.images = images
        self.labels = tf.convert_to_tensor(labels, dtype=tf.float32)
        self.features_computed = False  # Force recomputation for validation set
        self._precompute_features()
        
        # 2. Evaluate the individual on these features
        fitness = compute_fitness_score(
            self.image_size_tf, self.weight_bal_acc, self.weight_f1, self.weight_eff, self.weight_conn, self.weight_simp,
            tf.constant(True), # Verbose for validation
            self.precomputed_features, self.labels,
            self.n_patches_h, self.n_patches_w, self.feature_weights, self.n_patches_tf,
            individual.num_active_rules, self.max_rules_tf, individual.rules_tensor
        )
        
        # 3. Restore training data
        self.images = old_images
        self.labels = old_labels
        self.precomputed_features = old_features
        self.features_computed = True
        
        logger.info(f"Honest Validation Fitness: {fitness:.4f}")
        return float(fitness)

    def get_run_history(self):
        """
        Returns the history of fitness statistics and timing recorded during the last GA run.
        """
        return self.history

    def save_feature_scales(self, path):
        """Save feature scales to a JSON file"""
        if self.feature_scales is None:
            logger.warning("No feature scales to save.")
            return
            
        scales_dict = dict(zip(self.feature_names, self.feature_scales.numpy().tolist()))
        with open(path, 'w') as f:
            json.dump(scales_dict, f, indent=4)
        logger.info(f"Feature scales saved to {path}")

    def load_feature_scales(self, path):
        """Load feature scales from a JSON file"""
        if not os.path.exists(path):
            logger.warning(f"Feature scales file {path} not found.")
            return
            
        with open(path, 'r') as f:
            scales_dict = json.load(f)
            
        # Reconstruct the tensor in the correct order
        scales_list = [scales_dict[name] for name in self.feature_names]
        self.feature_scales = tf.convert_to_tensor(scales_list, dtype=tf.float32)
        logger.info(f"Feature scales loaded from {path}")