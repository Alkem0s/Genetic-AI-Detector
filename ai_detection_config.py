class AIDetectionConfig:
    """Shared configuration for AI Detection components"""
    def __init__(self):
        # Patch sizes
        self.patch_size = 16
        
        # Feature thresholds
        self.gradient_threshold = 0.8
        self.pattern_threshold = 0.7
        self.edge_threshold = 0.65
        self.symmetry_threshold = 0.8
        self.hash_threshold = 0.85
        
        # Feature weights (should sum to 1.0)
        self.feature_weights = {
            'gradient': 0.15,
            'pattern': 0.15,
            'noise': 0.15,
            'edge': 0.1,
            'symmetry': 0.1,
            'texture': 0.1,
            'color': 0.1,
            'hash': 0.15
        }
        
        # Feature indexing
        self.feature_indices = {
            'gradient': 0,
            'pattern': 1,
            'noise': 2,
            'edge': 3,
            'symmetry': 4,
            'texture': 5,
            'color': 6,
            'hash': 7
        }