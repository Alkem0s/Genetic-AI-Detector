class ThresholdsConfig:
    """Unified configuration for feature extraction thresholds"""
    
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
    
    def set_dynamic_resolution(self, image_size):
        """Adjust scale factor based on image size"""
        max_dim = max(image_size)
        if max_dim > self.max_resolution:
            self.scale_factor = self.max_resolution / max_dim
        elif max_dim < self.min_resolution:
            self.scale_factor = self.min_resolution / max_dim
        else:
            self.scale_factor = 1.0