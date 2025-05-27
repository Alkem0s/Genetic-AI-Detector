import numpy as np
import random
import multiprocessing
import tensorflow as tf
from deap import base, creator, tools, algorithms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import concurrent.futures
import threading

from feature_extractor import FeatureExtractor
import global_config
import utils

# Set up logging
import logging
logger = logging.getLogger(__name__)

# Global feature extractor - will be initialized per process
_global_feature_extractor = None

def _init_worker():
    """Initialize worker process with its own feature extractor and TensorFlow settings"""
    global _global_feature_extractor
    
    # Configure TensorFlow for multiprocessing
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Disable GPU for worker processes to avoid CUDA context issues.
            # Memory growth setting is not relevant for disabled GPUs in workers.
            tf.config.set_visible_devices([], 'GPU')
            logger.debug(f"Worker {multiprocessing.current_process().pid}: GPU disabled.")
        except RuntimeError as e:
            logger.warning(f"Worker {multiprocessing.current_process().pid}: Could not disable GPU: {e}")
    
    # Initialize feature extractor for this worker
    _global_feature_extractor = FeatureExtractor()
    logger.debug(f"Worker {multiprocessing.current_process().pid} initialized with FeatureExtractor")

def _worker_evaluate_individual(args):
    """
    Worker function for multiprocessing evaluation.
    Receives pre-sampled NumPy arrays to avoid OOM due to full data copying.
    """
    global _global_feature_extractor
    
    individual, images_data_sampled, labels_data_sampled, patch_size, img_height, img_width, \
    n_patches_h, n_patches_w, feature_weights, eval_sample_size_actual, batch_size, \
    n_patches, max_possible_rules = args
    
    # Convert sampled numpy arrays back to tensors for TensorFlow operations within the worker
    images = tf.convert_to_tensor(images_data_sampled)
    labels = tf.convert_to_tensor(labels_data_sampled)
    
    # eval_sample_size is now the actual size of the images/labels passed to this worker
    return GeneticFeatureOptimizer._evaluate_individual_static(
        individual, images, labels, patch_size, img_height, img_width,
        n_patches_h, n_patches_w, feature_weights, eval_sample_size_actual, # Pass the actual size of the sampled data
        batch_size, n_patches, max_possible_rules, _global_feature_extractor
    )

class GeneticFeatureOptimizer:
    """
    Class for optimizing feature selection in AI image detection using genetic algorithms.
    Uses DEAP library to evolve optimal feature-based conditional rules that generate
    dynamic masks per image based on extracted features.
    """

    def __init__(self, images, labels, detector_config=None, ga_config=None):
        """
        Initialize the genetic algorithm optimizer.

        Args:
            images: Numpy array of preprocessed images
            labels: Numpy array of labels (0 for human, 1 for AI)
            detector_config: AIDetectionConfig instance
            ga_config: GAConfig instance
        """
        logger.info("Initializing GeneticFeatureOptimizer...")
        self.detector_config = detector_config
        self.ga_config = ga_config

        # Convert inputs to TensorFlow tensors if they aren't already
        # ELIMINATE REDUNDANT IMAGE DATA STORAGE: Only store TensorFlow tensors.
        self.images = tf.convert_to_tensor(images) if not isinstance(images, tf.Tensor) else images
        self.labels = tf.convert_to_tensor(labels) if not isinstance(labels, tf.Tensor) else labels

        # Removed: self.images_np and self.labels_np to prevent duplicate storage.

        # Load all necessary values from the config
        self.patch_size = self.detector_config.patch_size
        self.batch_size = self.detector_config.batch_size
        self.population_size = self.ga_config.population_size
        self.n_generations = self.ga_config.n_generations
        self.crossover_prob = self.ga_config.crossover_prob
        self.mutation_prob = self.ga_config.mutation_prob
        self.tournament_size = self.ga_config.tournament_size
        self.use_multiprocessing = self.ga_config.use_multiprocessing
        
        # Ensure eval_sample_size doesn't exceed available images
        self.eval_sample_size = min(self.ga_config.sample_size_for_ga, tf.shape(images)[0].numpy())
        logger.info(f"Using evaluation sample size: {self.eval_sample_size}")

        self.rules_per_individual = self.ga_config.rules_per_individual
        self.max_possible_rules = self.ga_config.max_possible_rules
        self.random_seed = self.detector_config.random_seed

        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
            tf.random.set_seed(self.random_seed)
            logger.info(f"Random seed set to {self.random_seed}")

        # Feature weights (optional)
        self.feature_weights = tf.convert_to_tensor(list(global_config.feature_weights.values()),
                                                 dtype=tf.float32)

        # Define feature names for rule generation
        self.feature_names = list(global_config.feature_weights.keys())

        # Extract image dimensions
        if len(self.images.shape) >= 3:
            img_shape = tf.shape(images[0])
            self.img_height, self.img_width = img_shape[0].numpy(), img_shape[1].numpy()
            logger.info(f"Image dimensions: {self.img_height}x{self.img_width}")
        else:
            logger.error("Images must have at least 3 dimensions [batch, height, width, ...]")
            raise ValueError("Images must have at least 3 dimensions [batch, height, width, ...]")

        self.n_patches_h = tf.constant(self.img_height // self.patch_size, dtype=tf.int32)
        self.n_patches_w = tf.constant(self.img_width // self.patch_size, dtype=tf.int32)
        self.n_patches = self.n_patches_h * self.n_patches_w

        logger.info(f"Initialized with {self.n_patches} patches ({self.n_patches_h}×{self.n_patches_w})")

        # Initialize feature extractor for main process
        self.feature_extractor = FeatureExtractor()

        # Setup DEAP components
        self._setup_deap()
        logger.info("GeneticFeatureOptimizer initialization complete.")

    def _setup_deap(self):
        """Set up the DEAP genetic algorithm framework"""
        logger.info("Setting up DEAP components...")
        # Check if creator already has these attributes to avoid redefinition errors
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            logger.debug("Created 'FitnessMax' in DEAP creator.")

        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
            logger.debug("Created 'Individual' in DEAP creator.")

        self.toolbox = base.Toolbox()

        # Each individual is a list of rules
        self.toolbox.register("attr_rule", self._random_rule)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                             self.toolbox.attr_rule, n=self.rules_per_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        logger.debug("DEAP toolbox registered for rule, individual, and population creation.")

        # Register genetic operations
        self.toolbox.register("evaluate", self._evaluate_individual_wrapper)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self._mutate_rule_set)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        logger.debug("DEAP toolbox registered for evaluation, mating, mutation, and selection.")

        # Set up parallel processing strategy
        self._setup_parallel_processing()
        logger.info("DEAP setup complete.")

    def _setup_parallel_processing(self):
        """Setup parallel processing based on available resources and CUDA availability"""
        has_gpu = len(tf.config.list_physical_devices('GPU')) > 0
        cpu_count = multiprocessing.cpu_count()
        
        if self.use_multiprocessing and cpu_count > 1:
            if has_gpu:
                # Use ThreadPoolExecutor for GPU systems to avoid CUDA context issues
                logger.info(f"GPU detected. Using ThreadPoolExecutor with {min(cpu_count, 4)} threads")
                self.use_threads = True
                self.n_workers = min(cpu_count, 4)  # Limit threads to avoid overwhelming GPU
            else:
                # Use ProcessPoolExecutor for CPU-only systems for better parallelism
                logger.info(f"CPU-only system. Using ProcessPoolExecutor with {cpu_count} processes")
                self.use_threads = False
                self.n_workers = cpu_count
        else:
            logger.info("Parallel processing disabled. Using sequential evaluation.")
            self.use_threads = None
            self.n_workers = 1

    def _evaluate_individual_wrapper(self, individual):
        """Wrapper for individual evaluation that works with both single and multiprocessing"""
        # When using ThreadPoolExecutor or sequential, the original self.images and self.labels are accessible.
        # _evaluate_individual_static will then sample from these if eval_sample_size is smaller.
        return self._evaluate_individual_static(
            individual, self.images, self.labels, self.patch_size,
            self.img_height, self.img_width, self.n_patches_h, self.n_patches_w,
            self.feature_weights, self.eval_sample_size, self.batch_size,
            self.n_patches, self.max_possible_rules, self.feature_extractor
        )

    def _parallel_evaluate_population(self, population):
        """Evaluate population in parallel"""
        if self.use_threads is None:
            # Sequential evaluation
            return [self._evaluate_individual_wrapper(ind) for ind in population]
        
        if self.use_threads:
            # Use ThreadPoolExecutor for GPU systems
            # ThreadPoolExecutor shares memory, so self.images and self.labels are directly accessible.
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = [executor.submit(self._evaluate_individual_wrapper, ind) for ind in population]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            return results
        else:
            # Use ProcessPoolExecutor for CPU systems
            # OPTIMIZE DATA TRANSFER FOR MULTIPROCESSING:
            # Sample data once in the main process and pass this smaller subset to workers.
            
            num_images = tf.shape(self.images)[0].numpy()
            
            # Ensure eval_sample_size does not exceed available images
            current_eval_sample_size = min(self.eval_sample_size, num_images)
            
            # Get random indices for the sample
            indices = random.sample(range(num_images), current_eval_sample_size)
            indices_tensor = tf.convert_to_tensor(indices, dtype=tf.int32)

            # Sample the images and labels once from the main TensorFlow tensors and convert to NumPy.
            # This sampled data (a significantly smaller portion of the original images)
            # will be passed to each worker. This avoids copying the *entire* dataset.
            sampled_images_np = tf.gather(self.images, indices_tensor).numpy()
            sampled_labels_np = tf.gather(self.labels, indices_tensor).numpy()

            eval_args = []
            for individual in population:
                args = (individual, sampled_images_np, sampled_labels_np, self.patch_size,
                       self.img_height, self.img_width, self.n_patches_h.numpy(), 
                       self.n_patches_w.numpy(), self.feature_weights.numpy(),
                       current_eval_sample_size, # This is the actual size of the sampled data being passed
                       self.batch_size, self.n_patches.numpy(),
                       self.max_possible_rules)
                eval_args.append(args)
            
            with multiprocessing.Pool(processes=self.n_workers, initializer=_init_worker) as pool:
                results = pool.map(_worker_evaluate_individual, eval_args)
            return results

    def _random_rule(self):
        """Generate a random rule for feature-based conditional mask generation."""
        rule = {
            "feature": random.choice(self.feature_names),
            "threshold": tf.random.uniform([], 0.0, 1.0).numpy(),
            "operator": random.choice([">", "<"]),
            "action": tf.random.uniform([], 0, 2, dtype=tf.int32).numpy()
        }
        logger.debug(f"Generated random rule: {rule}")
        return rule

    def _mutate_rule_set(self, individual):
        """
        Mutate a rule set by modifying thresholds, operators, features, or actions.
        May also add or remove rules if appropriate.
        """
        logger.debug(f"Mutating individual with {len(individual)} rules.")
        for i, rule in enumerate(individual):
            # Modify threshold with small delta
            if random.random() < 0.2:
                delta = tf.random.uniform([], -0.1, 0.1).numpy()
                rule["threshold"] += delta
                rule["threshold"] = tf.clip_by_value(tf.constant(rule["threshold"]), 0.0, 1.0).numpy()
                logger.debug(f"Rule {i}: Mutated threshold to {rule['threshold']:.2f}")

            # Swap comparison operator
            if random.random() < 0.1:
                rule["operator"] = ">" if rule["operator"] == "<" else "<"
                logger.debug(f"Rule {i}: Mutated operator to {rule['operator']}")

            # Replace feature name
            if random.random() < 0.1:
                rule["feature"] = random.choice(self.feature_names)
                logger.debug(f"Rule {i}: Mutated feature to {rule['feature']}")

            # Flip action
            if random.random() < 0.05:
                rule["action"] = 1 - rule["action"]
                logger.debug(f"Rule {i}: Mutated action to {rule['action']}")

        # Possibly add a new rule
        if random.random() < 0.05 and len(individual) < self.max_possible_rules:
            individual.append(self._random_rule())
            logger.debug(f"Added a new rule. Total rules: {len(individual)}")

        # Possibly remove a rule
        if random.random() < 0.05 and len(individual) > 1:
            removed_rule = individual.pop(random.randrange(len(individual)))
            logger.debug(f"Removed a rule. Total rules: {len(individual)}. Removed: {removed_rule['feature']}")

        return individual,

    @staticmethod
    def _calculate_connectivity(patch_mask):
        """
        Calculate how connected the selected patches are in a vectorized manner.
        More connected patches are better as they likely represent coherent features.

        Args:
            patch_mask: Binary mask of shape (n_patches_h, n_patches_w)

        Returns:
            float: Connectivity score between 0 and 1
        """
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

    @staticmethod
    def _evaluate_individual_static(individual, images, labels, patch_size,
                                   img_height, img_width, n_patches_h, n_patches_w,
                                   feature_weights, eval_sample_size, batch_size,
                                   n_patches, max_possible_rules, feature_extractor):
        """
        Static method for evaluating individuals - can be called from worker processes.
        It samples from the 'images' and 'labels' it receives IF `eval_sample_size` is
        less than the `num_images_in_input`. Otherwise, it processes all received images.
        """
        num_images_in_input = tf.shape(images)[0].numpy()
        
        # Determine the actual sample size to use.
        # If the input `images` is already a pre-sampled subset (e.g., from ProcessPoolExecutor),
        # then `eval_sample_size` should be `num_images_in_input`, and we should not re-sample.
        # If `images` is the full dataset (e.g., from sequential or ThreadPoolExecutor),
        # then we sample `eval_sample_size` from it.
        if eval_sample_size < num_images_in_input:
            indices = random.sample(range(num_images_in_input), eval_sample_size)
            indices_tensor = tf.convert_to_tensor(indices, dtype=tf.int32)
            sample_images = tf.gather(images, indices_tensor)
            sample_labels = tf.gather(labels, indices_tensor)
        else:
            # No need to sample, use all provided images
            sample_images = images
            sample_labels = labels

        # Track metrics across all evaluated images
        predictions = []
        total_active_patches = 0
        connectivity_scores = []

        # Process images in batches to improve efficiency
        num_samples = tf.shape(sample_images)[0].numpy()

        for i in range(0, num_samples, batch_size):
            # Get current batch
            batch_end = min(i + batch_size, num_samples)
            batch_images = sample_images[i:batch_end]
            batch_labels = sample_labels[i:batch_end]

            # Extract features for all images in batch at once
            batch_patch_features = feature_extractor.extract_batch_patch_features(batch_images)

            # Process each image in the batch
            for j in range(tf.shape(batch_images)[0].numpy()):
                img = batch_images[j]
                label = batch_labels[j]

                # Get pre-computed patch features for this image
                patch_features = batch_patch_features[j]

                # Generate mask using patch features
                patch_mask = utils.generate_dynamic_mask(patch_features, n_patches_h, n_patches_w, individual)

                # Convert to TensorFlow tensor if it's not already
                if not isinstance(patch_mask, tf.Tensor):
                    patch_mask = tf.convert_to_tensor(patch_mask, dtype=tf.float32)

                # Track connectivity score
                conn_score = GeneticFeatureOptimizer._calculate_connectivity(patch_mask)
                connectivity_scores.append(conn_score)

                # Track how many patches are active for efficiency calculation
                active_patches = tf.reduce_sum(patch_mask).numpy()
                total_active_patches += active_patches

                # Convert patch mask to pixel mask
                pixel_mask = utils.convert_patch_mask_to_pixel_mask(patch_mask,
                                                                   image_shape=(img_height, img_width),
                                                                   patch_size=patch_size)

                # Extract full feature stack for this image
                feature_stack = feature_extractor.extract_all_features(img)

                # Ensure feature_stack is a tensor
                if not isinstance(feature_stack, tf.Tensor):
                    feature_stack = tf.convert_to_tensor(feature_stack)

                if not isinstance(pixel_mask, tf.Tensor):
                    pixel_mask = tf.convert_to_tensor(pixel_mask)

                # Apply mask to feature stack using TensorFlow operations
                # Expand dimensions for broadcasting
                pixel_mask_expanded = tf.expand_dims(pixel_mask, axis=-1)
                masked_features = feature_stack * pixel_mask_expanded

                # Calculate weighted feature importance using configurable weights
                # Use only the first 8 features or as many as available
                num_features = min(8, feature_stack.shape[2])

                # Extract relevant weights
                weights = feature_weights[:num_features]

                # Calculate per-feature scores
                feature_scores = tf.reduce_sum(masked_features[:, :, :num_features], axis=[0, 1])

                # Apply weights and sum
                total_score = tf.reduce_sum(feature_scores * weights)

                # Normalize by image size to make it scale-invariant
                normalized_score = total_score / (img_height * img_width)

                # Threshold for AI detection
                prediction = 1 if normalized_score > 0.01 else 0
                predictions.append(prediction)

        # Calculate classification metrics
        predictions_tensor = tf.convert_to_tensor(predictions)
        sample_labels_np = sample_labels.numpy() if isinstance(sample_labels, tf.Tensor) else sample_labels
        predictions_np = predictions_tensor.numpy() if isinstance(predictions_tensor, tf.Tensor) else predictions_tensor

        accuracy = accuracy_score(sample_labels_np, predictions_np)

        # Only calculate other metrics if we have both classes in predictions
        unique_predictions = tf.unique(predictions_tensor)[0]
        if tf.shape(unique_predictions)[0] > 1:
            precision = precision_score(sample_labels_np, predictions_np, zero_division=0)
            recall = recall_score(sample_labels_np, predictions_np, zero_division=0)
            f1 = f1_score(sample_labels_np, predictions_np, zero_division=0)
        else:
            # If all predictions are the same class
            precision = 0.5  # Neutral value
            recall = 0.5
            f1 = 0.5

        # Calculate average efficiency score
        avg_patch_selection_ratio = total_active_patches / (num_samples * n_patches)

        # Convert to TensorFlow calculations
        avg_patch_ratio_tensor = tf.constant(avg_patch_selection_ratio, dtype=tf.float32)

        # Progressive penalty: small for <30%, moderate for 30-50%, high for >50%
        def efficiency_case_1(): return tf.constant(1.0, dtype=tf.float32)
        def efficiency_case_2(): return 1.0 - (avg_patch_ratio_tensor - 0.3) * 2  # Linear penalty
        def efficiency_case_3(): return 0.6 - (avg_patch_ratio_tensor - 0.5) * 1.2  # Steeper penalty

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
            accuracy * 0.4 +           # Accuracy is most important
            precision * 0.15 +         # Precision helps reduce false positives
            recall * 0.15 +            # Recall helps catch all AI images
            f1 * 0.1 +                 # F1 score for balance
            efficiency_score.numpy() * 0.1 +   # Efficiency for compact solutions
            connectivity_score * 0.05 + # Connectivity for more coherent patch selections
            simplicity_score * 0.05    # Simplicity for more interpretable rules
        )

        return (fitness,)

    def get_feature_importance(self, rule_set):
        """
        Analyze which features contribute most based on the rule set.

        Args:
            rule_set: The optimized rule set

        Returns:
            dict: Feature importance scores
        """
        logger.info("Calculating feature importance...")
        # Initialize feature importance counters
        feature_scores = {feature: 0 for feature in self.feature_names}

        # Count feature usage in rules
        for rule in rule_set:
            feature = rule['feature']
            feature_scores[feature] += 1
        logger.debug(f"Initial rule-based feature scores: {feature_scores}")

        # Sample images to see which features actually contribute to masks
        num_images = tf.shape(self.images)[0].numpy()
        indices = random.sample(range(num_images), min(50, num_images))

        indices_tensor = tf.convert_to_tensor(indices, dtype=tf.int32)
        sample_images = tf.gather(self.images, indices_tensor)
        logger.debug(f"Analyzing feature triggers on {len(indices)} sample images.")

        # Track how often each feature's rules are triggered
        feature_triggers = {feature: 0 for feature in self.feature_names}
        total_patches_evaluated = 0

        # Process images in batches
        num_samples = tf.shape(sample_images)[0].numpy()

        for i in range(0, num_samples, self.batch_size):
            # Get current batch
            batch_end = min(i + self.batch_size, num_samples)
            batch_images = sample_images[i:batch_end]

            # Extract features for all images in batch at once
            batch_patch_features = self.feature_extractor.extract_batch_patch_features(batch_images)

            # Process each image in the batch
            for j in range(tf.shape(batch_images)[0].numpy()):
                # Get pre-computed patch features for this image
                patch_features = batch_patch_features[j]

                # Define a function to check rules for a single patch
                def process_patch(h, w):
                    # Extract patch feature values from precomputed patch features
                    patch_feature_dict = {}
                    for idx, feature_name in enumerate(self.feature_names):
                        if idx < patch_features.shape[2]:
                            patch_feature_dict[feature_name] = patch_features[h, w, idx].numpy()

                    # Check which rules are triggered
                    for rule in rule_set:
                        feature = rule["feature"]
                        if feature not in patch_feature_dict:
                            continue

                        value = patch_feature_dict[feature]
                        threshold = rule["threshold"]
                        operator = rule["operator"]

                        # Evaluate if rule is triggered
                        if (operator == ">" and value > threshold) or (operator == "<" and value < threshold):
                            feature_triggers[feature] += 1

                # Process all patches for this image
                for h in range(self.n_patches_h):
                    for w in range(self.n_patches_w):
                        process_patch(h, w)
                        total_patches_evaluated += 1

        logger.debug(f"Feature triggers: {feature_triggers}. Total patches evaluated: {total_patches_evaluated}")

        # Combine rule frequency and trigger frequency
        for feature in feature_scores:
            # Normalize rule frequency
            rule_freq = feature_scores[feature] / max(len(rule_set), 1) if rule_set else 0

            # Normalize trigger frequency
            trigger_freq = feature_triggers[feature] / max(total_patches_evaluated, 1) if total_patches_evaluated > 0 else 0

            # Combined score (70% rule frequency, 30% trigger frequency)
            feature_scores[feature] = 0.7 * rule_freq + 0.3 * trigger_freq

        # Normalize to sum to 1
        total = sum(feature_scores.values())
        if total > 0:
            for key in feature_scores:
                feature_scores[key] = feature_scores[key] / total
        logger.info(f"Feature importance calculation complete. Scores: {feature_scores}")

        return feature_scores

    def run(self):
        """
        Execute the genetic algorithm optimization.

        Returns:
            list: The optimized rule set
            dict: Statistics from the optimization process
        """
        logger.info(f"Starting genetic algorithm with population={self.population_size}, generations={self.n_generations}")

        # Create initial population
        pop = self.toolbox.population(n=self.population_size)
        logger.info(f"Initial population of {len(pop)} individuals created.")

        # Track statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        stats.register("std", np.std)
        logger.debug("Statistics registered for DEAP.")

        # Hall of Fame to track best individual
        hof = tools.HallOfFame(1)
        logger.debug("Hall of Fame initialized.")

        # Track evolution history
        history = []

        # Custom evolution loop with parallel evaluation
        for gen in range(self.n_generations):
            logger.info(f"Generation {gen + 1}/{self.n_generations}")
            
            # Evaluate population in parallel
            fitnesses = self._parallel_evaluate_population(pop)
            
            # Assign fitness values
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit
            
            # Update hall of fame and statistics
            hof.update(pop)
            record = stats.compile(pop)
            
            # Log progress
            logger.info(f"Gen {gen + 1}: Max={record['max']:.4f}, Avg={record['avg']:.4f}, Std={record['std']:.4f}")
            
            history.append({
                'generation': gen + 1,
                'max_fitness': record['max'],
                'avg_fitness': record['avg'],
                'std_fitness': record['std']
            })
            
            # Selection and breeding for next generation (except last generation)
            if gen < self.n_generations - 1:
                # Select next generation
                offspring = self.toolbox.select(pop, len(pop))
                offspring = list(map(self.toolbox.clone, offspring))
                
                # Apply crossover
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < self.crossover_prob:
                        self.toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values
                
                # Apply mutation
                for mutant in offspring:
                    if random.random() < self.mutation_prob:
                        self.toolbox.mutate(mutant)
                        del mutant.fitness.values
                
                pop[:] = offspring

        logger.info("Genetic algorithm evolution complete.")

        # Return the best individual
        best_ind = hof[0]

        # Log the best rule set in human-readable format
        logger.info("Best rule set found:")
        for i, rule in enumerate(best_ind):
            logger.info(f"Rule {i+1}: if {rule['feature']} {rule['operator']} {rule['threshold']:.2f} → {'include' if rule['action'] == 1 else 'exclude'} patch")

        # Calculate average mask statistics on a sample of images
        num_images = tf.shape(self.images)[0].numpy()
        sample_indices = random.sample(range(num_images), min(20, num_images))
        indices_tensor = tf.convert_to_tensor(sample_indices, dtype=tf.int32)
        sample_images = tf.gather(self.images, indices_tensor)
        logger.info(f"Calculating average mask statistics on {len(sample_indices)} sample images.")

        total_active = tf.constant(0, dtype=tf.float32)

        # Process evaluation images in batches
        num_samples = tf.shape(sample_images)[0].numpy()

        for i in range(0, num_samples, self.batch_size):
            batch_end = min(i + self.batch_size, num_samples)
            batch_images = sample_images[i:batch_end]
            batch_patch_features = self.feature_extractor.extract_batch_patch_features(batch_images)

            for j in range(tf.shape(batch_images)[0].numpy()):
                # Get pre-computed patch features
                patch_features = batch_patch_features[j]
                # Generate mask using precomputed features
                mask = utils.generate_dynamic_mask(patch_features, self.n_patches_h, self.n_patches_w, best_ind)
                # Convert to tensor if needed
                if not isinstance(mask, tf.Tensor):
                    mask = tf.convert_to_tensor(mask, dtype=tf.float32)
                total_active += tf.reduce_sum(mask)

        avg_active = total_active / tf.cast(tf.shape(sample_images)[0], tf.float32)
        avg_percentage = avg_active / (self.n_patches_h * self.n_patches_w) * 100

        logger.info(f"Best fitness: {best_ind.fitness.values[0]:.4f}")
        logger.info(f"Rules: {len(best_ind)}")
        logger.info(f"Average active patches: {avg_active:.1f} ({avg_percentage:.1f}%) out of {self.n_patches}")

        return best_ind, {
            'best_fitness': best_ind.fitness.values[0],
            'rule_count': len(best_ind),
            'avg_active_patches': avg_active.numpy(),
            'avg_active_percentage': avg_percentage.numpy(),
            'history': history
        }

    def optimize_hyperparameters(self, param_grid, metric='fitness'):
        """
        Find optimal hyperparameters for the genetic algorithm.

        Args:
            param_grid: Dictionary with parameter names as keys and lists of values to try
            metric: Metric to optimize ('fitness', 'accuracy', etc.)

        Returns:
            dict: Best hyperparameters
            float: Best score
        """
        logger.info("Starting hyperparameter optimization")

        # Keep track of best configuration
        best_score = -float('inf')
        best_params = {}

        # Simple grid search
        param_combinations = self._generate_combinations(param_grid)
        logger.info(f"Generated {len(param_combinations)} hyperparameter combinations.")

        for i, params in enumerate(param_combinations):
            logger.info(f"[{i+1}/{len(param_combinations)}] Trying parameters: {params}")

            # Update parameters
            for key, value in params.items():
                setattr(self, key, value)

            # Re-setup DEAP with new parameters if necessary (e.g., if rules_per_individual changes)
            # This ensures the toolbox uses the updated `self` attributes.
            self._setup_deap() # This will re-register `evaluate` with the updated parameters

            # Run a quick optimization (fewer generations)
            quick_gens = min(5, self.n_generations)
            old_gens = self.n_generations
            self.n_generations = quick_gens
            logger.debug(f"Running quick optimization for {quick_gens} generations.")

            try:
                _, stats = self.run()
                score = stats['best_fitness']

                logger.info(f"Parameters {params} yielded fitness: {score:.4f}")
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    logger.info(f"New best: {best_score:.4f} with {best_params}")
            except Exception as e:
                logger.error(f"Error with parameters {params}: {e}")

            # Restore original generations
            self.n_generations = old_gens

        # Final run with best parameters
        logger.info(f"Hyperparameter optimization complete. Best hyperparameters: {best_params}, Best score: {best_score:.4f}")
        for key, value in best_params.items():
            setattr(self, key, value)

        self._setup_deap() # This will re-register `evaluate` with the best parameters for final use
        return best_params, best_score

    def _generate_combinations(self, param_grid):
        """Generate all combinations of parameters in param_grid"""
        logger.debug("Generating parameter combinations...")
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = []

        def _recursive_combine(idx, current_combo):
            if idx == len(keys):
                combinations.append(current_combo.copy())
                return

            for val in values[idx]:
                current_combo[keys[idx]] = val
                _recursive_combine(idx + 1, current_combo)

        _recursive_combine(0, {})
        logger.debug(f"Generated {len(combinations)} combinations.")
        return combinations