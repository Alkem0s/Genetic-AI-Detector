# global_config.py

# --- Detector Configuration ---
data = "train.csv"
output_dir = "output"
cnn_batch_size = 64
extraction_batch_size = 330
max_images = 100
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
population_size = 5
n_generations = 2
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
    'balanced_accuracy': 0.35,
    'f1': 0.35,
    'efficiency_score': 0.1,
    'connectivity_score': 0.1,
    'simplicity_score': 0.1
}

# --- Feature Weights (should sum to 1.0) ---
# Key ORDER must match the tf.stack order in feature_extractor._extract_single_patch_features
feature_weights = {
    'gradient':            0.08, # Measures unnatural gradient perfection
    'pattern':             0.10, # Detects repeating patterns/artifacts + phase + comb
    'noise':               0.09, # Analyzes noise distribution
    'edge':                0.08, # Examines edge coherence and artifacts
    'symmetry':            0.10, # Measures unnatural symmetry
    'texture':             0.08, # Analyzes LBP texture consistency
    'color':               0.08, # Detects color distribution anomalies
    'hash':                0.10, # Perceptual hash similarity to known AI patterns
    'dct':                 0.10, # Analyzes DCT AC/DC energy ratios (JPEG artifact)
    'channel_correlation': 0.10, # Detects chromatic aberration vs AI channel alignment
    'glcm':                0.09  # GLCM contrast/homogeneity (smooth AI textures)
}
