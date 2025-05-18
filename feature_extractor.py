import numpy as np
import cv2
from typing import Tuple, Dict, List, Any

from ai_detection_config import AIDetectionConfig

class FeatureExtractor:
    """
    Class for feature extraction in AI image detection system.
    Implements methods to extract various features that can identify AI-generated artifacts.
    
    The GeneticFeatureOptimizer expects this module to extract features
    from patches of images and return them in a consistent format.
    """
    
    def __init__(self, config=None):
        """
        Initialize the feature extractor with configuration parameters.
        
        Args:
            config: Configuration object or None to use defaults
        """
        if config is None:
            self.config = AIDetectionConfig()  # Default to AIDetectionConfig
        else:
            self.config = config
        
        # Each extracted feature should be normalized to range [0.0, 1.0]
        # where higher values typically indicate stronger AI artifacts presence
    
    def extract_all_features(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Extract all features from an image.
        
        This is the main method that the genetic algorithm will call.
        It should extract all features and return them in a standardized format.
        
        Args:
            image: Input image as numpy array (height, width, channels)
            
        Returns:
            Tuple containing:
                - visualization: Visualization image with highlighted features (can be None)
                - feature_stack: 3D array where each channel is a different feature map
                - metadata: Dictionary with additional metadata about extracted features
        """
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Initialize feature stack - MUST have the same spatial dimensions as the input image
        # with one channel for each feature type
        feature_stack = np.zeros((height, width, len(self.config.feature_weights.keys())), dtype=np.float32)

        # Extract individual features and populate the stack
        # Each feature should be normalized to [0, 1] range
        
        # Example for gradient feature extraction (implement your actual algorithm here)
        feature_stack[:, :, 0] = self._extract_gradient_feature(image)
        
        # Example for pattern detection
        feature_stack[:, :, 1] = self._extract_pattern_feature(image)
        
        # Noise analysis
        feature_stack[:, :, 2] = self._extract_noise_feature(image)
        
        # Edge coherence
        feature_stack[:, :, 3] = self._extract_edge_feature(image)
        
        # Symmetry detection
        feature_stack[:, :, 4] = self._extract_symmetry_feature(image)
        
        # Texture analysis
        feature_stack[:, :, 5] = self._extract_texture_feature(image)
        
        # Color distribution analysis
        feature_stack[:, :, 6] = self._extract_color_feature(image)
        
        # Perceptual hash comparison
        feature_stack[:, :, 7] = self._extract_hash_feature(image)
        
        # Create visualization image (optional)
        visualization = self._create_visualization(image, feature_stack)
        
        # Prepare metadata
        metadata = {
            "image_dimensions": (height, width),
            "feature_weights": self.config.feature_weights,
            "feature_statistics": self._compute_feature_statistics(feature_stack)
        }
        
        return visualization, feature_stack, metadata
    
    def _extract_gradient_feature(self, image: np.ndarray) -> np.ndarray:
        """
        Extract gradient perfection feature which identifies unnaturally perfect gradients.
        
        Args:
            image: Input image
            
        Returns:
            2D feature map same size as input image with values between 0 and 1
        """
        # Placeholder for actual implementation
        # This should return a 2D array with values normalized to [0,1]
        # Example: Calculate gradients and measure their regularity
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Compute gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Example metric: gradient consistency (simplified)
        # Higher values indicate more consistent/perfect gradients (potentially AI-generated)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        gradient_max = np.max(gradient_magnitude) if np.max(gradient_magnitude) > 0 else 1
        normalized_gradient = gradient_magnitude / gradient_max
        
        # Return a normalized feature map
        return normalized_gradient
    
    def _extract_pattern_feature(self, image: np.ndarray) -> np.ndarray:
        """
        Detect repeating patterns that may indicate AI artifacts.
        
        Args:
            image: Input image
            
        Returns:
            2D feature map same size as input image with values between 0 and 1
        """
        # Placeholder for actual implementation
        # This would detect repeating patterns using frequency domain analysis or other techniques
        height, width = image.shape[:2]
        feature_map = np.zeros((height, width), dtype=np.float32)
        
        # Example placeholder
        # Your actual implementation would look for repeating patterns
        return feature_map
    
    def _extract_noise_feature(self, image: np.ndarray) -> np.ndarray:
        """
        Analyze noise distribution to detect AI-generated inconsistencies.
        
        Args:
            image: Input image
            
        Returns:
            2D feature map same size as input image with values between 0 and 1
        """
        # Placeholder for actual implementation
        # This would analyze noise patterns, look for unnaturally clean areas, etc.
        height, width = image.shape[:2]
        feature_map = np.zeros((height, width), dtype=np.float32)
        
        # Example placeholder
        # Your actual implementation would analyze noise patterns
        return feature_map
    
    def _extract_edge_feature(self, image: np.ndarray) -> np.ndarray:
        """
        Examine edge coherence and artifacts typical in AI-generated images.
        
        Args:
            image: Input image
            
        Returns:
            2D feature map same size as input image with values between 0 and 1
        """
        # Placeholder for actual implementation
        height, width = image.shape[:2]
        feature_map = np.zeros((height, width), dtype=np.float32)
        
        # Example placeholder
        # Your actual implementation would detect edge artifacts
        return feature_map
    
    def _extract_symmetry_feature(self, image: np.ndarray) -> np.ndarray:
        """
        Measure unnatural symmetry often present in AI-generated images.
        
        Args:
            image: Input image
            
        Returns:
            2D feature map same size as input image with values between 0 and 1
        """
        # Placeholder for actual implementation
        height, width = image.shape[:2]
        feature_map = np.zeros((height, width), dtype=np.float32)
        
        # Example placeholder
        # Your actual implementation would detect symmetry anomalies
        return feature_map
    
    def _extract_texture_feature(self, image: np.ndarray) -> np.ndarray:
        """
        Analyze texture consistency and identify AI artifacts in textures.
        
        Args:
            image: Input image
            
        Returns:
            2D feature map same size as input image with values between 0 and 1
        """
        # Placeholder for actual implementation
        height, width = image.shape[:2]
        feature_map = np.zeros((height, width), dtype=np.float32)
        
        # Example placeholder
        # Your actual implementation would analyze texture patterns
        return feature_map
    
    def _extract_color_feature(self, image: np.ndarray) -> np.ndarray:
        """
        Detect color distribution anomalies common in AI-generated images.
        
        Args:
            image: Input image
            
        Returns:
            2D feature map same size as input image with values between 0 and 1
        """
        # Placeholder for actual implementation
        height, width = image.shape[:2]
        feature_map = np.zeros((height, width), dtype=np.float32)
        
        # Example placeholder
        # Your actual implementation would analyze color distributions
        return feature_map
    
    def _extract_hash_feature(self, image: np.ndarray) -> np.ndarray:
        """
        Compare perceptual hash similarity to known AI patterns.
        
        Args:
            image: Input image
            
        Returns:
            2D feature map same size as input image with values between 0 and 1
        """
        # Placeholder for actual implementation
        height, width = image.shape[:2]
        feature_map = np.zeros((height, width), dtype=np.float32)
        
        # Example placeholder
        # Your actual implementation would use perceptual hashing techniques
        return feature_map
    
    def _compute_feature_statistics(self, feature_stack: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Compute statistics for each feature map for metadata.
        
        Args:
            feature_stack: 3D array containing all feature maps
            
        Returns:
            Dictionary with statistics for each feature
        """
        stats = {}
        
        for i, feature_name in enumerate(self.config.feature_weights.keys()):
            if i < feature_stack.shape[2]:
                feature_map = feature_stack[:, :, i]
                stats[feature_name] = {
                    "mean": float(np.mean(feature_map)),
                    "min": float(np.min(feature_map)),
                    "max": float(np.max(feature_map)),
                    "std": float(np.std(feature_map))
                }
        
        return stats
    
    def _create_visualization(self, image: np.ndarray, feature_stack: np.ndarray) -> np.ndarray:
        """
        Create a visualization of the extracted features overlaid on the image.
        
        Args:
            image: Original input image
            feature_stack: 3D array containing all feature maps
            
        Returns:
            Visualization image with features highlighted
        """
        # Simple visualization: Combine features into a heatmap
        if len(image.shape) == 3:
            visualization = image.copy()
        else:
            # Convert grayscale to RGB for visualization
            visualization = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Create a combined feature map (average of all features)
        combined_feature = np.mean(feature_stack, axis=2)
        
        # Normalize to 0-1 range
        if np.max(combined_feature) > 0:
            combined_feature = combined_feature / np.max(combined_feature)
        
        # Create a heatmap overlay (higher values are more red)
        heatmap = np.zeros_like(visualization)
        heatmap[:, :, 2] = (combined_feature * 255).astype(np.uint8)  # Red channel
        
        # Blend with original image
        alpha = 0.6  # Transparency of overlay
        blended = cv2.addWeighted(visualization, 1 - alpha, heatmap, alpha, 0)
        
        return blended
    
    def extract_patch_features(self, image: np.ndarray, patch_size: int = 16) -> np.ndarray:
        """
        Extract average feature values for each patch in the image.
        This function is particularly useful for the genetic algorithm's rule application.
        
        Args:
            image: Input image
            patch_size: Size of patches to analyze
            
        Returns:
            4D array with shape (n_patches_h, n_patches_w, n_features, 1) containing
            average feature values for each patch
        """
        # Extract all features
        _, feature_stack, _ = self.extract_all_features(image)
        
        height, width = image.shape[:2]
        n_patches_h = height // patch_size
        n_patches_w = width // patch_size
        n_features = len(self.config.feature_weights.keys())
        
        # Initialize patch features array
        patch_features = np.zeros((n_patches_h, n_patches_w, n_features), dtype=np.float32)
        
        # For each patch
        for h in range(n_patches_h):
            for w in range(n_patches_w):
                # Calculate patch boundaries
                y_start = h * patch_size
                y_end = min((h + 1) * patch_size, height)
                x_start = w * patch_size
                x_end = min((w + 1) * patch_size, width)
                
                # Extract average feature values for this patch
                for f in range(min(n_features, feature_stack.shape[2])):
                    patch_features[h, w, f] = np.mean(feature_stack[y_start:y_end, x_start:x_end, f])
        
        return patch_features