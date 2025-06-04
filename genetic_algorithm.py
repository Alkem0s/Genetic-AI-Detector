import sys
import numpy as np
import random
import time
from deap import base, creator, tools
from feature_extractor import FeatureExtractor
import global_config
from utils import evaluate_ga_individual_optimized, generate_dynamic_mask
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

        self.images = tf.convert_to_tensor(images) if not isinstance(images, tf.Tensor) else images
        self.labels = tf.convert_to_tensor(labels) if not isinstance(labels, tf.Tensor) else labels

        self.patch_size = self.detector_config.patch_size
        self.batch_size = self.detector_config.batch_size
        self.population_size = self.ga_config.population_size
        self.n_generations = self.ga_config.n_generations
        self.crossover_prob = self.ga_config.crossover_prob
        self.mutation_prob = self.ga_config.mutation_prob
        self.tournament_size = self.ga_config.tournament_size

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

        self.feature_weights = tf.convert_to_tensor(list(global_config.feature_weights.values()),
                                                 dtype=tf.float32)
        self.feature_names = list(global_config.feature_weights.keys())

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

        self.feature_extractor = FeatureExtractor()

        # Initialize precomputed features storage
        self.precomputed_features = None
        self.eval_labels = None

        self.history = []
        self._setup_deap()
        logger.info("GeneticFeatureOptimizer initialization complete.")

    def _precompute_features(self):
        """
        Precompute patch features for the evaluation dataset once before GA starts.
        This eliminates redundant feature extraction across all individuals.
        """
        logger.info("Precomputing patch features for evaluation dataset...")
        start_time = time.time()
        
        num_images_in_input = tf.shape(self.images)[0].numpy()
        
        # Determine the actual sample to use for evaluation
        if self.eval_sample_size < num_images_in_input:
            indices = random.sample(range(num_images_in_input), self.eval_sample_size)
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
        
        end_time = time.time()
        logger.info(f"Feature precomputation complete in {end_time - start_time:.2f} seconds")
        logger.info(f"Precomputed features shape: {self.precomputed_features.shape}")

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

        self.toolbox.register("attr_rule", self._random_rule)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                             self.toolbox.attr_rule, n=self.rules_per_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        logger.debug("DEAP toolbox registered for rule, individual, and population creation.")

        self.toolbox.register("evaluate", self._evaluate_individual_wrapper)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self._mutate_rule_set)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        logger.debug("DEAP toolbox registered for evaluation, mating, mutation, and selection.")

        self._setup_sequential_processing()
        logger.info("DEAP setup complete.")

    def _setup_sequential_processing(self):
        """Setup for sequential processing as per user's request to use single GPU."""
        self.use_joblib = False
        self.n_workers = 1
        logger.info("Configured for sequential evaluation on the main process to leverage GPU.")

    def _evaluate_individual_wrapper(self, individual):
        """Wrapper for individual evaluation that uses precomputed features"""
        start_time = time.time()
        
        # Ensure features are precomputed
        if self.precomputed_features is None:
            logger.warning("Features not precomputed, computing on-the-fly (less efficient)")
            self._precompute_features()
        
        fitness = evaluate_ga_individual_optimized(
            individual, self.precomputed_features, self.eval_labels, self.patch_size,
            self.img_height, self.img_width, self.n_patches_h, self.n_patches_w,
            self.feature_weights, self.n_patches, self.max_possible_rules
        )
        
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

    def _random_rule(self):
        """Generate a random rule for feature-based conditional mask generation."""
        rule = {
            "feature": random.choice(self.feature_names),
            "threshold": tf.random.uniform([], 0.0, 1.0).numpy(),
            "operator": random.choice([">", "<"]),
            "action": tf.random.uniform([], 0, 2, dtype=tf.int32).numpy()
        }
        return rule

    def _mutate_rule_set(self, individual):
        """
        Mutate a rule set by modifying thresholds, operators, features, or actions.
        May also add or remove rules if appropriate.
        """
        logger.debug(f"Mutating individual with {len(individual)} rules.")
        for i, rule in enumerate(individual):
            if random.random() < 0.2:
                delta = tf.random.uniform([], -0.1, 0.1).numpy()
                rule["threshold"] += delta
                rule["threshold"] = tf.clip_by_value(tf.constant(rule["threshold"]), 0.0, 1.0).numpy()
                logger.debug(f"Rule {i}: Mutated threshold to {rule['threshold']:.2f}")

            if random.random() < 0.1:
                rule["operator"] = ">" if rule["operator"] == "<" else "<"
                logger.debug(f"Rule {i}: Mutated operator to {rule['operator']}")

            if random.random() < 0.1:
                rule["feature"] = random.choice(self.feature_names)
                logger.debug(f"Rule {i}: Mutated feature to {rule['feature']}")

            if random.random() < 0.05:
                rule["action"] = 1 - rule["action"]
                logger.debug(f"Rule {i}: Mutated action to {rule['action']}")

        if random.random() < 0.05 and len(individual) < self.max_possible_rules:
            individual.append(self._random_rule())
            logger.debug(f"Added a new rule. Total rules: {len(individual)}")

        if random.random() < 0.05 and len(individual) > 1:
            removed_rule = individual.pop(random.randrange(len(individual)))
            logger.debug(f"Removed a rule. Total rules: {len(individual)}. Removed: {removed_rule['feature']}")

        return individual,
    
    def get_feature_importance(self, rule_set):
        """
        Analyze which features contribute most based on the rule set.
        Uses precomputed features for efficiency.
        """
        logger.info("Calculating feature importance...")
        feature_scores = {feature: 0 for feature in self.feature_names}

        for rule in rule_set:
            feature = rule['feature']
            feature_scores[feature] += 1
        logger.debug(f"Initial rule-based feature scores: {feature_scores}")

        # Use precomputed features if available, otherwise sample from images
        if self.precomputed_features is not None:
            # Use a subset of precomputed features for analysis
            num_samples = min(50, tf.shape(self.precomputed_features)[0].numpy())
            sample_features = self.precomputed_features[:num_samples]
            logger.debug(f"Analyzing feature triggers on {num_samples} precomputed feature sets.")
        else:
            # Fallback to original sampling method
            num_images = tf.shape(self.images)[0].numpy()
            indices = random.sample(range(num_images), min(50, num_images))
            indices_tensor = tf.convert_to_tensor(indices, dtype=tf.int32)
            sample_images = tf.gather(self.images, indices_tensor)
            logger.debug(f"Analyzing feature triggers on {len(indices)} sample images.")
            
            # Extract features for this subset
            sample_features_list = []
            num_samples = tf.shape(sample_images)[0].numpy()
            for i in range(0, num_samples, self.batch_size):
                batch_end = min(i + self.batch_size, num_samples)
                batch_images = sample_images[i:batch_end]
                batch_patch_features = self.feature_extractor.extract_batch_patch_features(batch_images)
                sample_features_list.append(batch_patch_features)
            sample_features = tf.concat(sample_features_list, axis=0)

        feature_triggers = {feature: 0 for feature in self.feature_names}
        total_patches_evaluated = 0

        num_samples = tf.shape(sample_features)[0].numpy()

        for j in range(num_samples):
            patch_features = sample_features[j]

            def process_patch(h, w):
                patch_feature_dict = {}
                for idx, feature_name in enumerate(self.feature_names):
                    if idx < patch_features.shape[2]:
                        patch_feature_dict[feature_name] = patch_features[h, w, idx].numpy()

                for rule in rule_set:
                    feature = rule["feature"]
                    if feature not in patch_feature_dict:
                        continue

                    value = patch_feature_dict[feature]
                    threshold = rule["threshold"]
                    operator = rule["operator"]

                    if (operator == ">" and value > threshold) or (operator == "<" and value < threshold):
                        feature_triggers[feature] += 1

            for h in range(self.n_patches_h):
                for w in range(self.n_patches_w):
                    process_patch(h, w)
                    total_patches_evaluated += 1

        logger.debug(f"Feature triggers: {feature_triggers}. Total patches evaluated: {total_patches_evaluated}")

        for feature in feature_scores:
            rule_freq = feature_scores[feature] / max(len(rule_set), 1) if rule_set else 0
            trigger_freq = feature_triggers[feature] / max(total_patches_evaluated, 1) if total_patches_evaluated > 0 else 0
            feature_scores[feature] = 0.7 * rule_freq + 0.3 * trigger_freq

        total = sum(feature_scores.values())
        if total > 0:
            for key in feature_scores:
                feature_scores[key] = feature_scores[key] / total
        logger.info(f"Feature importance calculation complete. Scores: {feature_scores}")

        return feature_scores

    def run(self):
        """
        Execute the genetic algorithm optimization.
        """
        logger.info(f"Starting genetic algorithm with population={self.population_size}, generations={self.n_generations}")

        # Precompute features once before starting GA
        self._precompute_features()

        pop = self.toolbox.population(n=self.population_size)
        logger.info(f"Initial population of {len(pop)} individuals created.")

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        stats.register("std", np.std)
        logger.debug("Statistics registered for DEAP.")

        hof = tools.HallOfFame(1)
        logger.debug("Hall of Fame initialized.")

        self.history = [] # Reset history for each run call

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
            
            self.history.append({
                'generation': gen + 1,
                'max_fitness': record['max'],
                'avg_fitness': record['avg'],
                'std_fitness': record['std'],
                'generation_time': gen_end_time - gen_start_time
            })
            
            if gen < self.n_generations - 1:
                offspring = self.toolbox.select(pop, len(pop))
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
                
                pop[:] = offspring

        logger.info("Genetic algorithm evolution complete.")

        best_ind = hof[0]

        logger.info("Best rule set found:")
        for i, rule in enumerate(best_ind):
            logger.info(f"Rule {i+1}: if {rule['feature']} {rule['operator']} {rule['threshold']:.2f} → {'include' if rule['action'] == 1 else 'exclude'} patch")

        # Calculate final statistics using precomputed features
        sample_size = min(20, tf.shape(self.precomputed_features)[0].numpy())
        sample_features = self.precomputed_features[:sample_size]
        logger.info(f"Calculating average mask statistics on {sample_size} sample images.")

        total_active = tf.constant(0, dtype=tf.float32)

        mask_gen_start_time = time.time()
        for j in range(sample_size):
            patch_features = sample_features[j]
            mask = generate_dynamic_mask(patch_features, self.n_patches_h, self.n_patches_w, best_ind)
            if not isinstance(mask, tf.Tensor):
                mask = tf.convert_to_tensor(mask, dtype=tf.float32)
            total_active += tf.reduce_sum(mask)
        mask_gen_end_time = time.time()
        logger.info(f"Mask generation for final statistics took {mask_gen_end_time - mask_gen_start_time:.4f} seconds.")

        avg_active = total_active / tf.cast(sample_size, tf.float32)
        avg_percentage = avg_active / (self.n_patches_h * self.n_patches_w) * 100

        logger.info(f"Best fitness: {best_ind.fitness.values[0]:.4f}")
        logger.info(f"Rules: {len(best_ind)}")
        logger.info(f"Average active patches: {avg_active:.1f} ({avg_percentage:.1f}%) out of {self.n_patches}")

        return best_ind, {
            'best_fitness': best_ind.fitness.values[0],
            'rule_count': len(best_ind),
            'avg_active_patches': avg_active.numpy(),
            'avg_active_percentage': avg_percentage.numpy(),
            'history': self.history
        }

    def get_run_history(self):
        """
        Returns the history of fitness statistics and timing recorded during the last GA run.
        """
        return self.history

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

        best_score = -float('inf')
        best_params = {}

        param_combinations = self._generate_combinations(param_grid)
        logger.info(f"Generated {len(param_combinations)} hyperparameter combinations.")

        for i, params in enumerate(param_combinations):
            logger.info(f"[{i+1}/{len(param_combinations)}] Trying parameters: {params}")

            for key, value in params.items():
                setattr(self, key, value)

            self._setup_deap() # Re-setup DEAP with new parameters

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

            self.n_generations = old_gens # Restore original generations for next combination

        logger.info(f"Hyperparameter optimization complete. Best hyperparameters: {best_params}, Best score: {best_score:.4f}")
        for key, value in best_params.items():
            setattr(self, key, value)

        self._setup_deap()
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