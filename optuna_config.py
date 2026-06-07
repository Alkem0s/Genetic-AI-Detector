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
ga_config_trials = 150       # Trials for optimizing GA configuration

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
    "population_size": 180,
    "n_generations": 120,
    "rules_per_individual": 12,
    "max_possible_rules": 50,
    "crossover_prob": 0.6,
    "mutation_prob": 0.35,
    "tournament_size": 4,
    "num_elites": 2,
    "inactive_weight_penalty": 0.1,
    "target_sparsity": 0.45,
    "sparsity_radius": 0.25,
    "verbose": False
}

# =============================================================================
# FEATURE WEIGHT OPTIMIZATION RANGES
# =============================================================================

# These define the search space for feature weights
# The optimizer will ensure they sum to 1.0
# ORDER MUST MATCH THE CANONICAL STACK IN FEATURE EXTRACTOR
feature_weight_ranges = {
    'gradient': (0.5, 1.0),
    'pattern': (0.0, 0.3),
    'noise': (0.5, 1.0),
    'symmetry': (0.1, 0.6),
    'texture': (0.1, 0.6),
    'color': (0.1, 0.6),
    'hash': (0.5, 1.0),
    'dct': (0.0, 0.6),
    'glcm': (0.0, 0.6),
    'noise_spectrum': (0.1, 0.6),
    'local_entropy': (0.0, 1.0)
}


# =============================================================================
# GA CONFIGURATION OPTIMIZATION RANGES
# =============================================================================

# Population and generation settings
population_size_range = (100, 250)  # Expanded downwards to allow lighter GAs
n_generations_range = (50, 200)    # Expanded downwards to allow faster convergence

# Rule configuration
# We have 11 features, so we want at least 11-12 rules for full coverage
rules_per_individual_range = (12, 25)
max_possible_rules_range = (30, 60)

# Mask coverage (Sparsity) targets
target_sparsity_range = (0.25, 0.70)  # Expanded downwards to allow sparser/more efficient selective masks
sparsity_radius_range = (0.05, 0.20) # Expanded downwards to allow tighter constraints

# Genetic operators
crossover_prob_range = (0.5, 0.85)
mutation_prob_range = (0.2, 0.55)
tournament_size_range = (2, 5)
num_elites_range = (2, 4)
inactive_weight_penalty_range = (0.005, 0.2)

# Compute effort penalty coefficient (used in Phase 2 to discourage bloated populations/generations)
compute_penalty_coefficient = 5e-7  # Halved from 1e-6 (0.005 penalty per 10,000 extra evaluations)


# =============================================================================
# PRUNING CONFIGURATION
# =============================================================================

# Enable pruning for faster optimization
enable_pruning = True
pruning_warmup_steps = 50  # Number of generations before pruning can occur
pruning_interval = 5      # Check for pruning every N generations

# The percentile of trials to KEEP. Optuna's PercentilePruner keeps the top X%
# of trials and prunes the rest. Setting this to 75.0 keeps the top 75%
# (i.e. prunes the bottom 25%). Setting it to 25.0 would prune the bottom 75%.
prune_percentile = 75.0

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

# =============================================================================
# CNN HYPERPARAMETER OPTIMIZATION CONFIGURATION (Phase 3)
# =============================================================================
cnn_trials = 12
cnn_epochs = 40                  # Maximum epochs to train each CNN trial
cnn_early_stopping_patience = 5  # Patience for early stopping during tuning
num_cnn_runs_per_trial = 1       # Number of seeded runs to average per trial for stability

# Max images in total during CNN HPO to speed up trials (distributed automatically)
cnn_max_train_samples = 30000
cnn_max_val_samples = 5000

cnn_study_name_template = "cnn_optimization_{mask_mode}"
cnn_config_output_file_template = "best_cnn_config_{mask_mode}_scaled.json"

# CNN HPO Search Spaces
cnn_learning_rate_range = (1e-5, 1e-2)
cnn_dropout_1_range = (0.0, 0.8)
cnn_dropout_2_range = (0.0, 0.8)
cnn_dense_units_range = (64, 1024)
cnn_l2_reg_range = (1e-6, 1e-2)
cnn_optimizers = ['adam', 'rmsprop', 'sgd']

