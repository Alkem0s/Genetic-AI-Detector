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
# FEATURE WEIGHT OPTIMIZATION RANGES
# =============================================================================

# These define the search space for feature weights
# The optimizer will ensure they sum to 1.0
feature_weight_ranges = {
    'gradient': (0.0, 1.0),    # Measures unnatural gradient perfection
    'pattern': (0.0, 1.0),     # Detects repeating patterns/artifacts  
    'noise': (0.0, 1.0),       # Analyzes noise distribution
    'edge': (0.0, 1.0),        # Examines edge coherence and artifacts
    'symmetry': (0.0, 1.0),    # Measures unnatural symmetry
    'texture': (0.0, 1.0),     # Analyzes texture consistency
    'color': (0.0, 1.0),       # Detects color distribution anomalies
    'hash': (0.0, 1.0),        # Perceptual hash similarity to known AI patterns
    'dct': (0.0, 1.0),         # Analyzes DCT AC/DC energy ratios
    'channel_correlation': (0.0, 1.0), # Detects chromatic aberration vs AI alignment
    'glcm': (0.0, 1.0),        # GLCM contrast/homogeneity
    'noise_spectrum': (0.0, 1.0), # High frequency noise signature analysis
    'ycbcr_correlation': (0.0, 1.0) # Chrominance and Luminance correlation 
}


# =============================================================================
# GA CONFIGURATION OPTIMIZATION RANGES
# =============================================================================

# Population and generation settings
population_size_range = (120, 180)
n_generations_range = (100, 350)

# Rule configuration
rules_per_individual_range = (5, 15)
max_possible_rules_range = (15, 50)

# Genetic operators
crossover_prob_range = (0.6, 0.75)
mutation_prob_range = (0.2, 0.45)
tournament_size_range = (2, 8)
num_elites_range = (1, 5)


# =============================================================================
# PRUNING CONFIGURATION
# =============================================================================

# Enable pruning for faster optimization
enable_pruning = True
pruning_warmup_steps = 50  # Number of generations before pruning can occur
pruning_interval = 5      # Check for pruning every N generations

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