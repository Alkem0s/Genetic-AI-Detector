# global_config.py

# --- Detector Configuration ---
output_dir = "output"
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

verbose = False  # Enable detailed debugging output
visualize = False
profile = False  # Enable performance profiling
profile_log_dir = "profiler_logs" # Directory for profiling data

# --- Cross-Generator Matrix ---
# Leave empty to use the CSV-based approach, or specify folder names under dataset_sampled/
# Example: train_generators = ["wukong", "glide", "adm"]
train_generators = [] 
val_generators = []
dataset_sampled_dir = "dataset_sampled"
max_train_per_gen = 10000  # Max AI/Real images to take from each train generator
max_val_per_gen = 2500     # Max AI/Real images to take from each val generator

# --- Unified Patch and Image Sizes ---
image_size = 256
patch_size = 16
scale_factor = 1.0

# --- Genetic Algorithm Environment ---
sample_size = 8000
use_multiprocessing = True

# --- Fitness Weights (Fixed for now) ---
fitness_weights = {
    'balanced_accuracy': 0.35,
    'f1': 0.35,
    'efficiency_score': 0.1,
    'connectivity_score': 0.1,
    'simplicity_score': 0.1
}
