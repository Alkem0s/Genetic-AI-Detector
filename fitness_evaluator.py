# fitness_evaluator.py
"""
Wrapper class for fitness evaluation functions to maintain compatibility
with the existing genetic algorithm code structure.
"""

from fitness_evaluation import compute_fitness_score, evaluate_ga_individual

class FitnessEvaluator:
    """
    Wrapper class for fitness evaluation functions.
    This maintains compatibility with the existing genetic algorithm structure.
    """
    
    def __init__(self, config):
        """
        Initialize the fitness evaluator.
        
        Args:
            config: Configuration object containing fitness parameters
        """
        self.config = config
    
    def evaluate_ga_individual(self, individual, config, precomputed_features, labels, 
                             n_patches_h, n_patches_w, feature_weights, n_patches, 
                             max_possible_rules):
        """
        Evaluate a single individual in the genetic algorithm.
        
        This is a wrapper around the function in fitness_evaluation.py
        
        Args:
            individual: Individual to evaluate
            config: Configuration object
            precomputed_features: Pre-extracted features
            labels: Ground truth labels
            n_patches_h: Number of patches in height
            n_patches_w: Number of patches in width  
            feature_weights: Weights for different features
            n_patches: Total number of patches
            max_possible_rules: Maximum number of rules allowed
            
        Returns:
            Tuple containing fitness score
        """
        return evaluate_ga_individual(
            individual, config, precomputed_features, labels,
            n_patches_h, n_patches_w, feature_weights, n_patches, max_possible_rules
        )