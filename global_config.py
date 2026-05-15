# global_config.py

# --- Detector Configuration ---
output_dir = "output"

# --- Experiment Control ---
# Mask mode: "none" (baseline CNN), "ga" (GA-generated masks), "random" (random masks)
mask_mode = "ga"
# JPEG robustness test quality levels
jpeg_quality_levels = [50, 75]
cnn_batch_size = 64
extraction_batch_size = 64
max_images = 10000
epochs = 50
test_size = 0.2
random_seed = 42
model_path = "ai_detector_model.h5"
mask_path = "optimized_patch_mask.npy"
rules_path = "genetic_rules.pkl"
feature_cache_dir = "feature_cache"
mixed_precision = True
predict = False
predict_path = ""
skip_training = False
use_feature_extraction = True
use_augmentation = True

verbose = False 
visualize = False
profile = False 
profile_log_dir = "profiler_logs" 

# --- Cross-Generator Matrix ---
# Generators used as training set (in-distribution experiment)
train_generators = ["ADM", "glide", "wukong"]
# Generators used as validation/test set (cross-generator experiment)
val_generators = ["sdv4", "vqdm"]
dataset_sampled_dir = "dataset_sampled"
max_train_per_gen = 10000  # Max AI/Real images to take from each train generator
max_val_per_gen = 2500     # Max AI/Real images to take from each val generator

# Explicit generator splits for run_experiment() calls (mirrors train/val by default)
generator_train = ["ADM", "glide", "wukong"]
generator_test  = ["sdv4", "vqdm"]

# --- Unified Patch and Image Sizes ---
image_size = 256
patch_size = 8
scale_factor = 1.0

# --- Genetic Algorithm Environment ---
sample_size = 6000
probe_sample_size = 3000
use_multiprocessing = True
use_feature_cache = False

# --- Fitness Weights ---
# divergence_score replaces the old balanced_accuracy + f1 components.
# The GA now rewards rule sets that maximise the statistical separation between
# AI and Human patch-feature distributions (FDR + Bhattacharyya composite)
# rather than performing linear classification.
# Efficiency weight is increased to prevent degenerate all-patch / no-patch masks.
fitness_weights = {
    'divergence_score':   0.75,
    'efficiency_score':   0.10,
    'connectivity_score': 0.05,
    'simplicity_score':   0.10,
}

# --- Mask Coverage Targets ---
# Any sparsity within [target_sparsity - radius, target_sparsity + radius] 
# will receive a perfect 1.0 efficiency score. 
# Defaults to [0.2, 0.6] range.
target_sparsity = 0.4
sparsity_radius = 0.2
