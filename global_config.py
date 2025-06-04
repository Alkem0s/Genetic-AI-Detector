# Structural features
gradient_threshold = 0.8
pattern_threshold = 0.7
edge_threshold = 0.65
symmetry_threshold = 0.8

# Texture features
noise_threshold = 0.7
texture_threshold = 0.65
color_threshold = 0.7

# Other features
hash_threshold = 0.85

# Unified patch sizes
default_patch_size = 16
large_feature_patch_size = 32

# Resolution scaling
image_size = 224  # Default image size for processing
min_resolution = 224
max_resolution = 1920
scale_factor = 1.0  # Will be dynamically adjusted

debug_timing = True  # Enable detailed timing for feature extraction 

# Feature weights (should sum to 1.0)
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