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

# Multi-run averaging for robust objective evaluation
num_ga_runs_per_trial = 3
deterministic_trial_seeding = True

# Feature weight usage regularization/penalty
optimize_weight_penalty = True
inactive_weight_penalty = 0.2  # Used as fixed penalty if optimize_weight_penalty is False

# Proxy GA Configuration used DURING Phase 1 optimization
# These will override best_ga_config.json for Phase 1.
use_proxy_ga_config = True
proxy_ga_config = {
    "population_size": 150,
    "n_generations": 150,
    "rules_per_individual": 12,
    "max_possible_rules": 50,
    "crossover_prob": 0.7,
    "mutation_prob": 0.3,
    "tournament_size": 3,
    "num_elites": 1,
    "inactive_weight_penalty": 0.2,
    "verbose": False
}

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
population_size_range = (100, 180)
n_generations_range = (100, 350)

# Rule configuration
rules_per_individual_range = (10, 25)
max_possible_rules_range = (15, 50)

# Genetic operators
crossover_prob_range = (0.5, 0.75)
mutation_prob_range = (0.2, 0.5)
tournament_size_range = (2, 8)
num_elites_range = (1, 5)
inactive_weight_penalty_range = (0.0, 0.5)


# =============================================================================
# PRUNING CONFIGURATION
# =============================================================================

# Enable pruning for faster optimization
enable_pruning = True
pruning_warmup_steps = 50  # Number of generations before pruning can occur
pruning_interval = 5      # Check for pruning every N generations

# Minimum fitness threshold for early stopping
# Set to 0 initially — the divergence score starts near 0 for random rule sets
# and the scale differs from the old balanced_accuracy + f1 scheme.
min_fitness_threshold = 0.0

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