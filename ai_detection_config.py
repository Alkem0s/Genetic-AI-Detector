from thresholds_config import ThresholdsConfig

class AIDetectionConfig(ThresholdsConfig):
    """Shared configuration for AI Detection components"""
    def __init__(self):
        super().__init__()
        
        # Feature weights (should sum to 1.0)
        self.feature_weights = {
            'gradient': 0.10,
            'pattern': 0.12,
            'edge': 0.10,
            'symmetry': 0.16,
            'noise': 0.14,
            'texture': 0.12,
            'color': 0.10,
            'hash': 0.16
        }

        
        # Feature indexing
        self.feature_indices = {
            'gradient': 0,
            'pattern': 1,
            'edge': 2,
            'symmetry': 3,
            'noise': 4,
            'texture': 5,
            'color': 6,
            'hash': 7
        }