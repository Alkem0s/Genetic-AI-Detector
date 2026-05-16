# optuna_config.py
"""
Configuration file for Optuna hyperparameter optimization.
Modify these values to control the optimization process.
"""

# =============================================================================
# OPTIMIZATION CONFIGURATION
# =============================================================================

# Number of trials for each optimization phase
feature_weight_trials = 100  # Trials for optimizing feature weights
ga_config_trials = 100       # Trials for optimizing GA configuration

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
    "population_size": 160,
    "n_generations": 120,
    "rules_per_individual": 14,
    "max_possible_rules": 50,
    "crossover_prob": 0.6,
    "mutation_prob": 0.35,
    "tournament_size": 5,
    "num_elites": 3,
    "inactive_weight_penalty": 0.05,
    "target_sparsity": 0.6,
    "sparsity_radius": 0.2,
    "verbose": False
}

# =============================================================================
# FEATURE WEIGHT OPTIMIZATION RANGES
# =============================================================================

# These define the search space for feature weights
# The optimizer will ensure they sum to 1.0
feature_weight_ranges = {
    # The Power Trio (Consistently > 0.7 in Phase 1)
    'gradient': (0.5, 1.0),
    'noise': (0.5, 1.0),
    'hash': (0.5, 1.0),
    
    # Support Signals (Consistently 0.2 - 0.6)
    'symmetry': (0.1, 0.6),
    'texture': (0.1, 0.6),
    'color': (0.1, 0.6),
    'noise_spectrum': (0.1, 0.6),

    # Mixed

    'glcm': (0.0, 0.6),
    'dct': (0.0, 0.6),
    'saturation_clipping': (0.0, 1.0),
    'local_entropy': (0.0, 1.0),

    # Weak
    'channel_correlation': (0.0, 0.3),
    'pattern': (0.0, 0.3)
    

}


# =============================================================================
# GA CONFIGURATION OPTIMIZATION RANGES
# =============================================================================

# Population and generation settings
population_size_range = (150, 220)
n_generations_range = (80, 150)

# Rule configuration
# We have 13 features, so we want at least 13-14 rules for full coverage
rules_per_individual_range = (14, 25)
max_possible_rules_range = (30, 60)

# Mask coverage (Sparsity) targets
target_sparsity_range = (0.4, 0.75)
sparsity_radius_range = (0.1, 0.20)

# Genetic operators
crossover_prob_range = (0.5, 0.7)
mutation_prob_range = (0.2, 0.4)
tournament_size_range = (2, 5)
num_elites_range = (2, 4)
inactive_weight_penalty_range = (0.05, 0.2)


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