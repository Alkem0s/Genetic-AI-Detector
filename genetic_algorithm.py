import numpy as np
import random
import multiprocessing
from deap import base, creator, tools, algorithms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

from ai_detection_config import AIDetectionConfig
import utils

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('GeneticFeatureOptimizer')

class GeneticFeatureOptimizer:
    """
    Class for optimizing feature selection in AI image detection using genetic algorithms.
    Uses DEAP library to evolve optimal patch selections that highlight AI-generated image artifacts.
    """
    
    def __init__(self, feature_extractor, images, labels, config=None,
                 population_size=50, n_generations=20,
                 crossover_prob=0.5, mutation_prob=0.2, tournament_size=3,
                 use_multiprocessing=True, eval_sample_size=100):
        """
        Initialize the genetic algorithm optimizer.

        Args:
            feature_extractor: Reference to the feature extractor class
            images: Numpy array of preprocessed images
            labels: Numpy array of labels (0 for human, 1 for AI)
            config: AIDetectionConfig instance or None to use defaults
            population_size: Size of the genetic algorithm population
            n_generations: Number of generations to evolve
            crossover_prob: Probability of crossover
            mutation_prob: Probability of mutation
            tournament_size: Tournament size for selection
            use_multiprocessing: Whether to use multiprocessing for evaluations
            eval_sample_size: Number of images to use for evaluation (for speed)
        """
        self.config = config or AIDetectionConfig()
        self.feature_extractor = feature_extractor
        self.images = images
        self.labels = labels
        self.patch_size = self.config.patch_size
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.use_multiprocessing = use_multiprocessing
        self.eval_sample_size = min(eval_sample_size, len(images))

        # Use config values
        self.feature_weights = list(self.config.feature_weights.values())
        self.feature_indices = self.config.feature_indices

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
        
        # Each gene represents whether a patch should be highlighted (1) or not (0)
        self.toolbox.register("attr_bool", random.randint, 0, 1)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, 
                             self.toolbox.attr_bool, n=self.n_patches)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Register genetic operations
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        
        # Parallel processing
        if self.use_multiprocessing and multiprocessing.cpu_count() > 1:
            pool = multiprocessing.Pool()
            self.toolbox.register("map", pool.map)
            logger.info(f"Using multiprocessing with {multiprocessing.cpu_count()} cores")
    
    def _evaluate_individual(self, individual):
        """
        Evaluate how well a patch selection mask helps classify images.
        
        This improved fitness function considers multiple metrics:
        - Accuracy: How well the patches predict AI vs human
        - Precision and recall: Ensuring balanced detection
        - Efficiency: Penalizing selecting too many patches
        - Consistency: Rewarding consistent performance across different image types
        """
        # Convert individual to patch mask
        patch_mask = np.array(individual).reshape(self.n_patches_h, self.n_patches_w)
        
        # Convert patch mask to pixel mask using the helper method
        pixel_mask = utils.convert_patch_mask_to_pixel_mask(patch_mask)
        
        # Select a random subset of images for evaluation
        indices = random.sample(range(len(self.images)), min(self.eval_sample_size, len(self.images)))
        sample_images = self.images[indices]
        sample_labels = self.labels[indices]
        
        # Extract features and apply mask
        feature_maps = []
        for img in sample_images:
            # Extract features
            _, feature_stack, _ = self.feature_extractor.extract_all_features(img)
            
            # Apply mask to feature stack
            masked_features = np.zeros_like(feature_stack)
            for c in range(feature_stack.shape[2]):
                masked_features[:,:,c] = feature_stack[:,:,c] * pixel_mask
                
            feature_maps.append(masked_features)
        
        # Make predictions using the masked features
        predictions = []
        for feature_map in feature_maps:
            # Calculate weighted feature importance using configurable weights
            total_score = 0
            for i in range(min(8, feature_map.shape[2])):  # Ensure we don't go out of bounds
                total_score += np.sum(feature_map[:,:,i]) * self.feature_weights[i]
            
            # Normalize by image size to make it scale-invariant
            normalized_score = total_score / (self.img_height * self.img_width)
            
            # Threshold for AI detection (can be optimized further)
            prediction = 1 if normalized_score > 0.01 else 0
            predictions.append(prediction)
        
        # Calculate metrics
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
        
        # Calculate efficiency score - penalize using too many patches
        patch_selection_ratio = sum(individual) / len(individual)
        efficiency_score = 1.0
        
        # Progressive penalty: small for <30%, moderate for 30-50%, high for >50%
        if patch_selection_ratio < 0.3:
            efficiency_score = 1.0
        elif patch_selection_ratio < 0.5:
            efficiency_score = 1.0 - (patch_selection_ratio - 0.3) * 2  # Linear penalty
        else:
            efficiency_score = 0.6 - (patch_selection_ratio - 0.5) * 1.2  # Steeper penalty
        
        # Calculate connectivity score - reward patches that are connected
        connectivity_score = self._calculate_connectivity(patch_mask)
        
        # Combined fitness score - weighted average of metrics
        fitness = (
            accuracy * 0.4 +           # Accuracy is most important
            precision * 0.15 +         # Precision helps reduce false positives
            recall * 0.15 +            # Recall helps catch all AI images
            f1 * 0.1 +                 # F1 score for balance
            efficiency_score * 0.1 +   # Efficiency for compact solutions
            connectivity_score * 0.1   # Connectivity for more coherent patch selections
        )
        
        return (fitness,)
    
    def _calculate_connectivity(self, patch_mask):
        """
        Calculate how connected the selected patches are.
        More connected patches are better as they likely represent coherent features.
        
        Returns:
            float: Connectivity score between 0 and 1
        """
        # If no patches selected, return 0
        if np.sum(patch_mask) == 0:
            return 0
            
        # Count adjacent selected patches
        adjacent_count = 0
        total_selected = np.sum(patch_mask)
        
        for i in range(self.n_patches_h):
            for j in range(self.n_patches_w):
                if patch_mask[i, j] == 1:
                    # Check neighbors (up, down, left, right)
                    for ni, nj in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
                        if 0 <= ni < self.n_patches_h and 0 <= nj < self.n_patches_w:
                            if patch_mask[ni, nj] == 1:
                                adjacent_count += 1
        
        # Normalize by maximum possible adjacencies (4 * selected - perimeter)
        # This is an approximation; actual max depends on configuration
        max_adjacencies = min(4 * total_selected - 2 * np.sqrt(total_selected) * 2, 
                             total_selected * 4)
        
        if max_adjacencies <= 0:
            return 0.5  # Single patch selected
            
        connectivity = adjacent_count / max_adjacencies
        return connectivity
    
    def run(self):
        """
        Execute the genetic algorithm optimization.
        
        Returns:
            numpy.ndarray: The optimized patch mask
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
        best_mask = np.array(best_ind).reshape(self.n_patches_h, self.n_patches_w)
        
        # Calculate additional metrics on the best individual
        active_patches = np.sum(best_mask)
        active_percentage = active_patches / (self.n_patches_h * self.n_patches_w) * 100
        
        logger.info(f"Best fitness: {best_ind.fitness.values[0]:.4f}")
        logger.info(f"Selected {active_patches} patches ({active_percentage:.1f}%) out of {len(best_ind)}")
        
        return best_mask, {
            'best_fitness': best_ind.fitness.values[0],
            'active_patches': active_patches,
            'active_percentage': active_percentage,
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
        # For a more advanced approach, consider using a library like optuna
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
    
    def get_feature_importance(self, best_mask):
        """
        Analyze which features contribute most to the fitness with the given mask.
        
        Args:
            best_mask: The optimized patch mask
            
        Returns:
            dict: Feature importance scores
        """
        # Create pixel mask from patch mask
        pixel_mask = utils.convert_patch_mask_to_pixel_mask(
            best_mask,
            patch_size=self.patch_size
        )
        
        # Sample images
        indices = random.sample(range(len(self.images)), min(100, len(self.images)))
        sample_images = self.images[indices]
        sample_labels = self.labels[indices]
        
        # Track importance by feature - update to match all 8 features in feature_extractor.py
        feature_scores = {
            'gradient': 0,
            'pattern': 0, 
            'noise': 0,
            'edge': 0,
            'symmetry': 0,
            'texture': 0,
            'color': 0,
            'hash': 0
        }
        
        feature_names = list(feature_scores.keys())
        
        # For each image, measure feature contribution
        for img, label in zip(sample_images, sample_labels):
            _, feature_stack, _ = self.feature_extractor.extract_all_features(img)
            
            # For each feature type, measure its contribution
            for feature_idx, name in enumerate(feature_names):
                if feature_idx >= feature_stack.shape[2]:
                    continue  # Skip if feature index out of bounds
                    
                # Create mask for just this feature
                feature_only = np.zeros_like(feature_stack)
                feature_only[:,:,feature_idx] = feature_stack[:,:,feature_idx] * pixel_mask
                
                # Calculate feature importance
                importance = np.sum(feature_only)
                feature_scores[name] += importance
        
        # Normalize scores
        total = sum(feature_scores.values())
        if total > 0:
            for key in feature_scores:
                feature_scores[key] = feature_scores[key] / total
        
        return feature_scores