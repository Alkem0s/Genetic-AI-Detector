# optuna_config.py
"""
Configuration file for Optuna hyperparameter optimization.
Modify these values to control the optimization process.
"""

# =============================================================================
# OPTIMIZATION CONFIGURATION
# =============================================================================

# Number of trials for each optimization phase
feature_weight_trials = 50  # Trials for optimizing feature weights
ga_config_trials = 50       # Trials for optimizing GA configuration

# Study names (for organization/logging)
feature_weight_study_name = "feature_weights_optimization"
ga_config_study_name = "ga_configuration_optimization"

# Random seed for reproducibility (set to None for random)
optimization_seed = 42

# Verbosity level (0=silent, 1=minimal, 2=detailed)
verbosity = 1

# =============================================================================
# BASE CONFIGURATION (unchanged parameters)
# =============================================================================

# Image and patch configuration
image_size = 224
patch_size = 16

# Data configuration  
sample_size = 1000  # Number of images to use for evaluation during optimization
extraction_batch_size = 330

# Fixed feature extraction settings
use_feature_extraction = True

# Fixed fitness weights (these won't be optimized)
fitness_weights = {
    'balanced_accuracy': 0.3,
    'f1': 0.25,
    'mcc': 0.2,
    'precision': 0.05,
    'recall': 0.05,
    'efficiency_score': 0.05,
    'connectivity_score': 0.05,
    'simplicity_score': 0.05
}

# =============================================================================
# FEATURE WEIGHT OPTIMIZATION RANGES
# =============================================================================

# These define the search space for feature weights
# The optimizer will ensure they sum to 1.0
feature_weight_ranges = {
    'gradient': (0.05, 0.25),    # Measures unnatural gradient perfection
    'pattern': (0.05, 0.25),     # Detects repeating patterns/artifacts  
    'noise': (0.05, 0.25),       # Analyzes noise distribution
    'edge': (0.05, 0.20),        # Examines edge coherence and artifacts
    'symmetry': (0.08, 0.30),    # Measures unnatural symmetry
    'texture': (0.05, 0.25),     # Analyzes texture consistency
    'color': (0.05, 0.20),       # Detects color distribution anomalies
    'hash': (0.08, 0.30)         # Perceptual hash similarity to known AI patterns
}

# Default feature weights (fallback/starting point)
default_feature_weights = {
    'gradient': 0.10,
    'pattern': 0.12, 
    'noise': 0.14,
    'edge': 0.10,
    'symmetry': 0.16,
    'texture': 0.12,
    'color': 0.10,
    'hash': 0.16
}

# =============================================================================
# GA CONFIGURATION OPTIMIZATION RANGES
# =============================================================================

# Population and generation settings
population_size_range = (50, 150)
n_generations_range = (50, 250)

# Rule configuration
rules_per_individual_range = (3, 15)
max_possible_rules_range = (10, 30)

# Genetic operators
crossover_prob_range = (0.5, 0.9)
mutation_prob_range = (0.1, 0.5)
tournament_size_range = (2, 8)
num_elites_range = (1, 5)

# Default GA configuration (fallback/starting point)
default_ga_config = {
    'population_size': 100,
    'n_generations': 100,
    'rules_per_individual': 8,
    'max_possible_rules': 20,
    'crossover_prob': 0.7,
    'mutation_prob': 0.2,
    'tournament_size': 3,
    'num_elites': 2,
    'random_seed': 42,
    'verbose': False  # Keep verbose False during optimization
}

# =============================================================================
# PRUNING CONFIGURATION
# =============================================================================

# Enable pruning for faster optimization
enable_pruning = True
pruning_warmup_steps = 5  # Number of generations before pruning can occur
pruning_interval = 3      # Check for pruning every N generations

# Minimum fitness threshold for early stopping
min_fitness_threshold = 0.1  # Stop trial if fitness is too low

# =============================================================================
# OUTPUT CONFIGURATION  
# =============================================================================

# Where to save the best parameters
feature_weights_output_file = "best_feature_weights.json"
ga_config_output_file = "best_ga_config.json"
combined_config_output_file = "best_combined_config.json"

# Whether to print detailed progress
show_progress_bar = True
log_intermediate_results = True