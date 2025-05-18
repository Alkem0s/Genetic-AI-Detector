class AIDetectionConfig():
    """Shared configuration for AI Detection components"""
    def __init__(self):
        # Structural features
        self.gradient_threshold = 0.8
        self.pattern_threshold = 0.7
        self.edge_threshold = 0.65
        self.symmetry_threshold = 0.8
        
        # Texture features
        self.noise_threshold = 0.7
        self.texture_threshold = 0.65
        self.color_threshold = 0.7
        
        # Other features
        self.hash_threshold = 0.85
        
        # Unified patch sizes
        self.default_patch_size = 16
        self.large_feature_patch_size = 32
        
        # Resolution scaling
        self.min_resolution = 224
        self.max_resolution = 1920
        self.scale_factor = 1.0  # Will be dynamically adjusted
        
        # Feature weights (should sum to 1.0)
        self.feature_weights = {
            'gradient': 0.10, # Measures unnatural gradient perfection
            'pattern': 0.12, # Detects repeating patterns/artifacts
            'noise': 0.14, # Analyzes noise distribution
            'edge': 0.10, # Examines edge coherence and artifacts
            'symmetry': 0.16, # Measures unnatural symmetry
            'texture': 0.12, # Analyzes texture consistency
            'color': 0.10, # Detects color distribution anomalies
            'hash': 0.16 # Perceptual hash similarity to known AI patterns
        }