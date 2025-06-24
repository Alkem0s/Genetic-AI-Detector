import sys
import numpy as np
import random
import time
from deap import base, creator, tools
from feature_extractor import FeatureExtractor
from utils import generate_dynamic_mask
from fitness_evaluation import evaluate_ga_individual
import logging
import tensorflow as tf

logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)

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
        
        self.images = tf.convert_to_tensor(images) if not isinstance(images, tf.Tensor) else images
        self.labels = tf.convert_to_tensor(labels) if not isinstance(labels, tf.Tensor) else labels

        self.config = config
        self.patch_size = config.patch_size
        self.batch_size = config.extraction_batch_size
        self.population_size = config.population_size
        self.n_generations = config.n_generations
        self.crossover_prob = config.crossover_prob
        self.mutation_prob = config.mutation_prob
        self.tournament_size = config.tournament_size
        self.num_elites = config.num_elites

        self.eval_sample_size = min(config.sample_size, tf.shape(images)[0].numpy())
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

        self.feature_weights = tf.convert_to_tensor(list(config.feature_weights.values()), dtype=tf.float32)
        self.feature_names = list(config.feature_weights.keys())

        # Create feature name to index mapping for tensor operations
        self.feature_name_to_idx = {name: idx for idx, name in enumerate(self.feature_names)}
        self.num_features = len(self.feature_names)

        self.n_patches_h = tf.constant(config.image_size // self.patch_size, dtype=tf.int32)
        self.n_patches_w = tf.constant(config.image_size // self.patch_size, dtype=tf.int32)
        self.n_patches = self.n_patches_h * self.n_patches_w

        logger.info(f"Initialized with {self.n_patches} patches ({self.n_patches_h}×{self.n_patches_w})")

        self.feature_extractor = FeatureExtractor(config)

        # Initialize precomputed features storage
        self.precomputed_features = None
        self.eval_labels = None
        self.features_computed = False

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
        
        num_images_in_input = tf.shape(self.images)[0].numpy()
        
        # Determine the actual sample to use for evaluation
        if self.eval_sample_size < num_images_in_input:
            # Use consistent sampling across runs by using the same seed for sampling
            if self.random_seed is not None:
                np.random.seed(self.random_seed)
            indices = np.random.choice(num_images_in_input, self.eval_sample_size, replace=False)
            indices_tensor = tf.convert_to_tensor(indices, dtype=tf.int32)
            sample_images = tf.gather(self.images, indices_tensor)
            self.eval_labels = tf.gather(self.labels, indices_tensor)
        else:
            sample_images = self.images
            self.eval_labels = self.labels

        num_samples = tf.shape(sample_images)[0].numpy()
        logger.info(f"Extracting features for {num_samples} evaluation images...")

        # Extract features for all images in batches
        all_batch_features = []
        for i in range(0, num_samples, self.batch_size):
            batch_end = min(i + self.batch_size, num_samples)
            batch_images = sample_images[i:batch_end]
            
            # Extract features once per batch
            batch_patch_features = self.feature_extractor.extract_batch_patch_features(batch_images)
            all_batch_features.append(batch_patch_features)
            
            logger.debug(f"Processed batch {i//self.batch_size + 1}/{(num_samples + self.batch_size - 1)//self.batch_size}")

        # Concatenate all batch features into a single tensor
        self.precomputed_features = tf.concat(all_batch_features, axis=0)
        self.features_computed = True
        
        end_time = time.time()
        logger.info(f"Feature precomputation complete in {end_time - start_time:.2f} seconds")
        logger.info(f"Precomputed features shape: {self.precomputed_features.shape}")

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
        logger.info("Setting up DEAP components...")
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
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        logger.debug("DEAP toolbox registered for evaluation, mating, mutation, and selection.")

        self._setup_sequential_processing()
        logger.info("DEAP setup complete.")

    def _setup_sequential_processing(self):
        """Setup for sequential processing as per user's request to use single GPU."""
        self.use_joblib = False
        self.n_workers = 1
        logger.info("Configured for sequential evaluation on the main process to leverage GPU.")

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
            logger.error("Features not precomputed! This should not happen with the refactored code.")
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

    def _sequential_evaluate_population(self, population):
        """Evaluate population sequentially on the main process"""
        logger.debug(f"Evaluating population of {len(population)} individuals sequentially on GPU/CPU.")

        results = []
        for i, ind in enumerate(population):
            result = self._evaluate_individual_wrapper(ind)
            results.append(result)
            logger.debug(f"Individual {i+1} fitness: {result[0]:.4f}")
        return results

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
            tf.where(tf.squeeze(mutation_mask_expanded) > 0, new_features, rules[:, 0]),
            tf.where(tf.squeeze(mutation_mask_expanded) > 0, new_thresholds, rules[:, 1]),
            tf.where(tf.squeeze(mutation_mask_expanded) > 0, new_operators, rules[:, 2]),
            tf.where(tf.squeeze(mutation_mask_expanded) > 0, new_actions, rules[:, 3])
        ], axis=1)
        
        # Possibly add or remove rules
        if tf.random.uniform([]) < 0.05 and individual.num_active_rules < self.max_possible_rules:
            # Add a new rule
            next_idx = individual.num_active_rules
            if next_idx < self.max_possible_rules:
                new_rule_data = self._random_rule_tensor()
                new_rules = tf.tensor_scatter_nd_update(
                    new_rules, 
                    [[next_idx]], 
                    [new_rule_data]
                )
                individual.num_active_rules += 1
        
        if tf.random.uniform([]) < 0.05 and individual.num_active_rules > 1:
            # Remove a rule by setting it to -1
            remove_idx = tf.random.uniform([], 0, individual.num_active_rules, dtype=tf.int32)
            new_rules = tf.tensor_scatter_nd_update(
                new_rules,
                [[remove_idx]],
                [[-1.0, -1.0, -1.0, -1.0]]
            )
            individual.num_active_rules -= 1
        
        # Update individual
        individual.rules_tensor = new_rules
        individual[0] = new_rules
        
        return individual,

    def _count_active_rules(self, rules_tensor):
        """Count the number of active rules in a tensor (non -1 entries)"""
        return tf.reduce_sum(tf.cast(rules_tensor[:, 0] >= 0, tf.int32)).numpy()
    
    def get_feature_importance(self, individual):
        """
        Analyze which features contribute most based on the tensor rule set.
        Uses precomputed features for efficiency.
        """
        logger.info("Calculating feature importance...")
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

        # Use precomputed features (should always be available now)
        num_samples = min(50, tf.shape(self.precomputed_features)[0].numpy())
        sample_features = self.precomputed_features[:num_samples]
        logger.debug(f"Analyzing feature triggers on {num_samples} precomputed feature sets.")

        feature_triggers = {feature: 0 for feature in self.feature_names}
        total_patches_evaluated = 0

        # Analyze trigger frequency using tensor operations
        if tf.shape(active_rules)[0] > 0:
            for j in range(tf.shape(sample_features)[0]):
                patch_features = sample_features[j]
                
                for h in range(self.n_patches_h):
                    for w in range(self.n_patches_w):
                        patch_feature_values = patch_features[h, w, :]
                        
                        # Check each active rule
                        for rule_idx in range(tf.shape(active_rules)[0]):
                            rule = active_rules[rule_idx]
                            feature_idx = tf.cast(rule[0], tf.int32)
                            threshold = rule[1] 
                            operator = tf.cast(rule[2], tf.int32)
                            
                            if feature_idx < tf.shape(patch_feature_values)[0]:
                                value = patch_feature_values[feature_idx]
                                
                                # Check condition
                                condition_met = tf.cond(
                                    tf.equal(operator, 0),
                                    lambda: value > threshold,
                                    lambda: value < threshold
                                )
                                
                                if condition_met:
                                    feature_name = self.feature_names[feature_idx.numpy()]
                                    feature_triggers[feature_name] += 1
                        
                        total_patches_evaluated += 1

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
        logger.info(f"Feature importance calculation complete. Scores: {feature_scores}")

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
            
        logger.info(f"Starting genetic algorithm run '{run_id}' with population={self.population_size}, generations={self.n_generations}")
        logger.info(f"Using precomputed features (computed={self.features_computed})")

        # Ensure features are precomputed (should already be done in __init__)
        if not self.features_computed:
            logger.warning("Features not precomputed, this should not happen in refactored version")
            self._precompute_features()

        # Reset current run history
        self.current_run_history = []

        pop = self.toolbox.population(n=self.population_size)
        logger.info(f"Initial population of {len(pop)} individuals created.")

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
            logger.info(f"Generation {gen + 1}/{self.n_generations}")
            
            gen_start_time = time.time()
            try:
                fitnesses = self._sequential_evaluate_population(pop)
                logger.debug(f"Generation {gen + 1}: Received {len(fitnesses)} fitness values")
            except Exception as e:
                logger.error(f"Error in sequential evaluation for generation {gen + 1}: {e}")
                raise
            gen_end_time = time.time()
            logger.info(f"Generation {gen + 1} evaluation phase took {gen_end_time - gen_start_time:.4f} seconds.")
            
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit
            
            hof.update(pop)
            record = stats.compile(pop)
            
            logger.info(f"Gen {gen + 1}: Max={record['max']:.4f}, Avg={record['avg']:.4f}, Std={record['std']:.4f}")
            
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

        logger.info(f"Genetic algorithm run '{run_id}' complete in {total_run_time:.2f} seconds.")

        best_ind = hof[0]

        logger.info("Best rule set found:")
        active_rules = tf.boolean_mask(best_ind.rules_tensor, best_ind.rules_tensor[:, 0] >= 0)
        for i in range(tf.shape(active_rules)[0]):
            rule = active_rules[i]
            feature_idx = tf.cast(rule[0], tf.int32)
            threshold = rule[1]
            operator = ">" if tf.cast(rule[2], tf.int32) == 0 else "<"
            action = "include" if tf.cast(rule[3], tf.int32) == 1 else "exclude"
            feature_name = self.feature_names[feature_idx.numpy()]
            logger.info(f"Rule {i+1}: if {feature_name} {operator} {threshold:.2f} → {action} patch")

        # Calculate final statistics using precomputed features
        sample_size = min(20, tf.shape(self.precomputed_features)[0].numpy())
        sample_features = self.precomputed_features[:sample_size]
        logger.info(f"Calculating average mask statistics on {sample_size} sample images.")

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
        logger.info(f"Mask generation for final statistics took {mask_gen_end_time - mask_gen_start_time:.4f} seconds.")

        avg_active = total_active / tf.cast(sample_size, tf.float32)
        avg_percentage = avg_active / (tf.cast(self.n_patches_h, tf.float32) * tf.cast(self.n_patches_w, tf.float32)) * 100

        logger.info(f"Best fitness: {best_ind.fitness.values[0]:.4f}")
        logger.info(f"Rules: {best_ind.num_active_rules}")
        logger.info(f"Average active patches: {avg_active:.1f} ({avg_percentage:.1f}%) out of {self.n_patches}")

        # Store run results
        run_results = {
            'run_id': run_id,
            'run_number': run_number,
            'best_fitness': best_ind.fitness.values[0],
            'rule_count': best_ind.num_active_rules,
            'avg_active_patches': avg_active.numpy(),
            'avg_active_percentage': avg_percentage.numpy(),
            'history': self.history
        }

    def get_run_history(self):
        """
        Returns the history of fitness statistics and timing recorded during the last GA run.
        """
        return self.history