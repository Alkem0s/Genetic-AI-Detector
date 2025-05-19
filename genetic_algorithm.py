import numpy as np
import random
import multiprocessing
from deap import base, creator, tools, algorithms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

from feature_extractor import FeatureExtractor
import global_config
from utils import utils

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('GeneticFeatureOptimizer')

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
        self.detector_config = detector_config
        self.ga_config = ga_config
        self.feature_extractor = FeatureExtractor(config=self.detector_config)
        self.images = images
        self.labels = labels

        # Load all necessary values from the config
        self.patch_size = self.detector_config.patch_size
        self.batch_size = self.detector_config.batch_size
        self.population_size = self.ga_config.population_size
        self.n_generations = self.ga_config.n_generations
        self.crossover_prob = self.ga_config.crossover_prob
        self.mutation_prob = self.ga_config.mutation_prob
        self.tournament_size = self.ga_config.tournament_size
        self.use_multiprocessing = self.ga_config.use_multiprocessing
        self.eval_sample_size = min(self.ga_config.sample_size_for_ga, len(images))
        self.rules_per_individual = self.ga_config.rules_per_individual
        self.max_possible_rules = self.ga_config.max_possible_rules
        self.random_seed = self.detector_config.random_seed

        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

        # Feature weights (optional)
        self.feature_weights = list(global_config.feature_weights.values())

        # Define feature names for rule generation
        self.feature_names = list(global_config.feature_weights.keys())

        # Extract image dimensions
        self.img_height, self.img_width = images[0].shape[:2]
        self.n_patches_h = self.img_height // self.patch_size
        self.n_patches_w = self.img_width // self.patch_size
        self.n_patches = self.n_patches_h * self.n_patches_w

        logger.info(f"Initialized with {self.n_patches} patches ({self.n_patches_h}×{self.n_patches_w})")

        # Setup DEAP components
        self._setup_deap()

    
    def _setup_deap(self):
        """Set up the DEAP genetic algorithm framework"""
        # Check if creator already has these attributes to avoid redefinition errors
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        
        # Each individual is a list of rules
        self.toolbox.register("attr_rule", self._random_rule)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, 
                             self.toolbox.attr_rule, n=self.rules_per_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Register genetic operations
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self._mutate_rule_set)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        
        # Parallel processing
        if self.use_multiprocessing and multiprocessing.cpu_count() > 1:
            pool = multiprocessing.Pool()
            self.toolbox.register("map", pool.map)
            logger.info(f"Using multiprocessing with {multiprocessing.cpu_count()} cores")
    
    def _random_rule(self):
        """Generate a random rule for feature-based conditional mask generation."""
        return {
            "feature": random.choice(self.feature_names),
            "threshold": random.uniform(0.0, 1.0),
            "operator": random.choice([">", "<"]),
            "action": random.choice([0, 1])
        }
    
    def _mutate_rule_set(self, individual):
        """
        Mutate a rule set by modifying thresholds, operators, features, or actions.
        May also add or remove rules if appropriate.
        """
        for rule in individual:
            # Modify threshold with small delta
            if random.random() < 0.2:
                rule["threshold"] += random.uniform(-0.1, 0.1)
                rule["threshold"] = min(max(rule["threshold"], 0.0), 1.0)
            
            # Swap comparison operator
            if random.random() < 0.1:
                rule["operator"] = ">" if rule["operator"] == "<" else "<"
            
            # Replace feature name
            if random.random() < 0.1:
                rule["feature"] = random.choice(self.feature_names)
            
            # Flip action
            if random.random() < 0.05:
                rule["action"] = 1 - rule["action"]
        
        # Possibly add a new rule
        if random.random() < 0.05 and len(individual) < self.max_possible_rules:
            individual.append(self._random_rule())
        
        # Possibly remove a rule
        if random.random() < 0.05 and len(individual) > 1:
            individual.pop(random.randrange(len(individual)))
        
        return individual,
    
    def _extract_patch_features_from_image(self, image):
        """
        Extract patch-level features from an image.
        
        Args:
            image: Input image
            
        Returns:
            numpy.ndarray: Patch features of shape (n_patches_h, n_patches_w, n_features)
            numpy.ndarray: Full feature stack for the image
        """
        # Extract full feature stack
        _, feature_stack, _ = self.feature_extractor.extract_all_features(image)
        
        # Initialize patch features array
        patch_features = np.zeros((self.n_patches_h, self.n_patches_w, len(self.feature_names)), dtype=np.float32)
        
        # Extract features for each patch
        for h in range(self.n_patches_h):
            for w in range(self.n_patches_w):
                # Calculate patch boundaries
                y_start = h * self.patch_size
                y_end = min((h + 1) * self.patch_size, self.img_height)
                x_start = w * self.patch_size
                x_end = min((w + 1) * self.patch_size, self.img_width)
                
                # Extract patch features
                for i, _ in enumerate(self.feature_names):
                    if i < feature_stack.shape[2]:  # Make sure the feature index is valid
                        patch_features[h, w, i] = np.mean(feature_stack[y_start:y_end, x_start:x_end, i])
        
        return patch_features, feature_stack
    
    def _evaluate_individual(self, individual):
        """
        Evaluate how well a rule set helps classify images.
        
        This fitness function considers multiple metrics:
        - Accuracy: How well the patches predict AI vs human
        - Precision and recall: Ensuring balanced detection
        - Efficiency: Penalizing selecting too many patches
        - Consistency: Rewarding consistent performance across different image types
        - Simplicity: Favoring simpler rule sets
        """
        # Select a random subset of images for evaluation
        indices = random.sample(range(len(self.images)), min(self.eval_sample_size, len(self.images)))
        sample_images = self.images[indices]
        sample_labels = self.labels[indices]
        
        # Track metrics across all evaluated images
        predictions = []
        total_active_patches = 0
        connectivity_scores = []
        
        # Process images in batches to improve efficiency
        for i in range(0, len(sample_images), self.batch_size):
            # Get current batch
            batch_images = sample_images[i:i+self.batch_size]
            batch_labels = sample_labels[i:i+self.batch_size]
            
            # Extract features for all images in batch at once
            batch_patch_features = self.feature_extractor.extract_batch_patch_features(batch_images, self.patch_size)
            
            # Process each image in the batch
            for j, (img, label) in enumerate(zip(batch_images, batch_labels)):
                # Get pre-computed patch features for this image
                patch_features = batch_patch_features[j]
                
                # Generate mask using patch features
                patch_mask = utils.generate_dynamic_mask(patch_features, individual)

                # Track connectivity score
                conn_score = self._calculate_connectivity(patch_mask)
                connectivity_scores.append(conn_score)
                
                # Track how many patches are active for efficiency calculation
                active_patches = np.sum(patch_mask)
                total_active_patches += active_patches
                
                # Convert patch mask to pixel mask
                pixel_mask = utils.convert_patch_mask_to_pixel_mask(patch_mask, image_shape=(self.img_height, self.img_width), patch_size=self.patch_size)
                
                # We need to get the full feature stack for scoring
                # Extract full feature stack for this image
                _, feature_stack, _ = self.feature_extractor.extract_all_features(img)
                
                # Apply mask to feature stack
                masked_features = np.zeros_like(feature_stack)
                for c in range(feature_stack.shape[2]):
                    masked_features[:,:,c] = feature_stack[:,:,c] * pixel_mask
                
                # Calculate weighted feature importance using configurable weights
                total_score = 0
                for k in range(min(8, feature_stack.shape[2])):  # Ensure we don't go out of bounds
                    total_score += np.sum(masked_features[:,:,k]) * self.feature_weights[k]
                
                # Normalize by image size to make it scale-invariant
                normalized_score = total_score / (self.img_height * self.img_width)
                
                # Threshold for AI detection
                prediction = 1 if normalized_score > 0.01 else 0
                predictions.append(prediction)
        
        # Calculate classification metrics
        accuracy = accuracy_score(sample_labels, predictions)
        
        # Only calculate other metrics if we have both classes in predictions
        if len(set(predictions)) > 1:
            precision = precision_score(sample_labels, predictions, zero_division=0)
            recall = recall_score(sample_labels, predictions, zero_division=0)
            f1 = f1_score(sample_labels, predictions, zero_division=0)
        else:
            # If all predictions are the same class
            precision = 0.5  # Neutral value
            recall = 0.5
            f1 = 0.5
        
        # Calculate average efficiency score
        avg_patch_selection_ratio = total_active_patches / (len(sample_images) * self.n_patches)
        efficiency_score = 1.0
        
        # Progressive penalty: small for <30%, moderate for 30-50%, high for >50%
        if avg_patch_selection_ratio < 0.3:
            efficiency_score = 1.0
        elif avg_patch_selection_ratio < 0.5:
            efficiency_score = 1.0 - (avg_patch_selection_ratio - 0.3) * 2  # Linear penalty
        else:
            efficiency_score = 0.6 - (avg_patch_selection_ratio - 0.5) * 1.2  # Steeper penalty
        
        # Calculate simplicity score based on number of rules
        simplicity_score = 1.0 - (len(individual) / self.max_possible_rules)
        
        # Use precomputed connectivity scores
        connectivity_score = np.mean(connectivity_scores) if connectivity_scores else 0.5
        
        # Combined fitness score - weighted average of metrics
        fitness = (
            accuracy * 0.4 +           # Accuracy is most important
            precision * 0.15 +         # Precision helps reduce false positives
            recall * 0.15 +            # Recall helps catch all AI images
            f1 * 0.1 +                 # F1 score for balance
            efficiency_score * 0.1 +   # Efficiency for compact solutions
            connectivity_score * 0.05 + # Connectivity for more coherent patch selections
            simplicity_score * 0.05    # Simplicity for more interpretable rules
        )
        
        return (fitness,)
    
    def _calculate_connectivity(self, patch_mask):
        """
        Calculate how connected the selected patches are in a vectorized manner.
        More connected patches are better as they likely represent coherent features.
        
        Args:
            patch_mask: Binary mask of shape (n_patches_h, n_patches_w)
        
        Returns:
            float: Connectivity score between 0 and 1
        """
        # If no patches selected, return 0
        if np.sum(patch_mask) == 0:
            return 0
        
        # Create shifted masks to check neighbors
        up = np.roll(patch_mask, shift=-1, axis=0)
        down = np.roll(patch_mask, shift=1, axis=0)
        left = np.roll(patch_mask, shift=-1, axis=1)
        right = np.roll(patch_mask, shift=1, axis=1)
        
        # Ensure edges do not wrap around
        up[-1, :] = 0
        down[0, :] = 0
        left[:, -1] = 0
        right[:, 0] = 0
        
        # Count adjacent selected patches
        adjacent_count = np.sum((patch_mask & up) + (patch_mask & down) + 
                                (patch_mask & left) + (patch_mask & right))
        
        # Total selected patches
        total_selected = np.sum(patch_mask)
        
        # Normalize by maximum possible adjacencies
        max_adjacencies = min(4 * total_selected - 2 * np.sqrt(total_selected) * 2, 
                              total_selected * 4)
        
        if max_adjacencies <= 0:
            return 0.5  # Single patch selected
        
        connectivity = adjacent_count / max_adjacencies
        return connectivity
    
    def get_feature_importance(self, rule_set):
        """
        Analyze which features contribute most based on the rule set.
        
        Args:
            rule_set: The optimized rule set
            
        Returns:
            dict: Feature importance scores
        """
        # Initialize feature importance counters
        feature_scores = {feature: 0 for feature in self.feature_names}
        
        # Count feature usage in rules
        for rule in rule_set:
            feature = rule['feature']
            feature_scores[feature] += 1
        
        # Sample images to see which features actually contribute to masks
        indices = random.sample(range(len(self.images)), min(50, len(self.images)))
        sample_images = self.images[indices]
        
        # Track how often each feature's rules are triggered
        feature_triggers = {feature: 0 for feature in self.feature_names}
        total_patches_evaluated = 0
        
        # Process images in batches
        for i in range(0, len(sample_images), self.batch_size):
            # Get current batch
            batch_images = sample_images[i:i+self.batch_size]
            
            # Extract features for all images in batch at once
            batch_patch_features = self.feature_extractor.extract_batch_patch_features(batch_images, self.patch_size)
            
            # Process each image in the batch
            for j, img in enumerate(batch_images):
                # Get pre-computed patch features for this image
                patch_features = batch_patch_features[j]
                
                # For each patch, check which feature's rules are triggered
                for h in range(self.n_patches_h):
                    for w in range(self.n_patches_w):
                        # Extract patch feature values from precomputed patch features
                        patch_feature_dict = {}
                        for i, feature_name in enumerate(self.feature_names):
                            if i < patch_features.shape[2]:
                                patch_feature_dict[feature_name] = patch_features[h, w, i]
                        
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
                        
                        total_patches_evaluated += 1
        
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
        
        # Track statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        stats.register("std", np.std)
        
        # Hall of Fame to track best individual
        hof = tools.HallOfFame(1)
        
        # Track evolution history
        history = []
        
        # Run the algorithm
        pop, logbook = algorithms.eaSimple(pop, self.toolbox, 
                                          cxpb=self.crossover_prob, 
                                          mutpb=self.mutation_prob,
                                          ngen=self.n_generations, 
                                          stats=stats,
                                          halloffame=hof,
                                          verbose=True)
        
        # Gather statistics
        gen_stats = logbook.select("gen")
        fit_stats = logbook.select("max")
        avg_stats = logbook.select("avg")
        
        for gen, max_fit, avg_fit in zip(gen_stats, fit_stats, avg_stats):
            history.append({
                'generation': gen,
                'max_fitness': max_fit,
                'avg_fitness': avg_fit
            })
            logger.info(f"Generation {gen}: Max fitness = {max_fit:.4f}, Avg fitness = {avg_fit:.4f}")
        
        # Return the best individual
        best_ind = hof[0]
        
        # Log the best rule set in human-readable format
        logger.info("Best rule set:")
        for i, rule in enumerate(best_ind):
            logger.info(f"Rule {i+1}: if {rule['feature']} {rule['operator']} {rule['threshold']:.2f} → {'include' if rule['action'] == 1 else 'exclude'} patch")
        
        # Calculate average mask statistics on a sample of images
        sample_indices = random.sample(range(len(self.images)), min(20, len(self.images)))
        sample_images = self.images[sample_indices]
        
        total_active = 0
        
        # Process evaluation images in batches
        for i in range(0, len(sample_images), self.batch_size):
            batch_images = sample_images[i:i+self.batch_size]
            batch_patch_features = self.feature_extractor.extract_batch_patch_features(batch_images, self.patch_size)
            
            for j, img in enumerate(batch_images):
                # Get pre-computed patch features
                patch_features = batch_patch_features[j]
                # Generate mask using precomputed features
                mask = utils.generate_dynamic_mask(patch_features, best_ind)
                total_active += np.sum(mask)
        
        avg_active = total_active / len(sample_images)
        avg_percentage = avg_active / (self.n_patches_h * self.n_patches_w) * 100
        
        logger.info(f"Best fitness: {best_ind.fitness.values[0]:.4f}")
        logger.info(f"Rules: {len(best_ind)}")
        logger.info(f"Average active patches: {avg_active:.1f} ({avg_percentage:.1f}%) out of {self.n_patches}")
        
        return best_ind, {
            'best_fitness': best_ind.fitness.values[0],
            'rule_count': len(best_ind),
            'avg_active_patches': avg_active,
            'avg_active_percentage': avg_percentage,
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
        
        for params in param_combinations:
            logger.info(f"Trying parameters: {params}")
            
            # Update parameters
            for key, value in params.items():
                setattr(self, key, value)
                
            # Re-setup DEAP with new parameters
            self._setup_deap()
            
            # Run a quick optimization (fewer generations)
            quick_gens = min(5, self.n_generations)
            old_gens = self.n_generations
            self.n_generations = quick_gens
            
            try:
                _, stats = self.run()
                score = stats['best_fitness']
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    logger.info(f"New best: {best_score:.4f} with {best_params}")
            except Exception as e:
                logger.error(f"Error with parameters {params}: {e}")
            
            # Restore original generations
            self.n_generations = old_gens
        
        # Final run with best parameters
        logger.info(f"Best hyperparameters: {best_params}")
        for key, value in best_params.items():
            setattr(self, key, value)
        
        self._setup_deap()
        return best_params, best_score
    
    def _generate_combinations(self, param_grid):
        """Generate all combinations of parameters in param_grid"""
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
        return combinations