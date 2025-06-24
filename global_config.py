# global_config.py

# --- Detector Configuration ---
data = "train.csv"
output_dir = "output"
cnn_batch_size = 64
extraction_batch_size = 330
max_images = 10000
epochs = 50
test_size = 0.2
random_seed = 42
model_path = "ai_detector_model.h5"
mask_path = "optimized_patch_mask.npy"
rules_path = "genetic_rules.pkl"
feature_cache_dir = "feature_cache"
mixed_precision = True
visualize = True
predict = False
predict_path = ""
skip_training = False
use_feature_extraction = True
use_augmentation = True

verbose = True  # Enable detailed debugging output

# --- Unified Patch and Image Sizes ---
image_size = 224
patch_size = 16
scale_factor = 1.0

# --- Genetic Algorithm Configuration ---
population_size = 100
n_generations = 100
sample_size = 1000
crossover_prob = 0.8
mutation_prob = 0.4
tournament_size = 6
num_elites = 5
use_multiprocessing = True
rules_per_individual = 10
max_possible_rules = 100

# --- Fitness Weights for Model Evaluation Metrics ---
# These weights determine the importance of each metric in the overall fitness score
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

# --- Feature Weights (should sum to 1.0) ---
feature_weights = {
    'gradient': 0.10, # Measures unnatural gradient perfection
    'pattern': 0.12, # Detects repeating patterns/artifacts
    'noise': 0.14, # Analyzes noise distribution
    'edge': 0.10, # Examines edge coherence and artifacts
    'symmetry': 0.16, # Measures unnatural symmetry
    'texture': 0.12, # Analyzes texture consistency
    'color': 0.10, # Detects color distribution anomalies
    'hash': 0.16 # Perceptual hash similarity to known AI patterns
}
