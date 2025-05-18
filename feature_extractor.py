import numpy as np
import cv2
from typing import Tuple, Dict, List, Any

from ai_detection_config import AIDetectionConfig
from structural_features import StructuralFeatureExtractor
from texture_features import TextureFeatureExtractor

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
        feature_stack[:, :, 0] = StructuralFeatureExtractor()._extract_gradient_feature(image)

        # Example for pattern detection
        feature_stack[:, :, 1] = StructuralFeatureExtractor()._extract_pattern_feature(image)

        # Noise analysis
        feature_stack[:, :, 2] = TextureFeatureExtractor()._extract_noise_feature(image)

        # Edge coherence
        feature_stack[:, :, 3] = StructuralFeatureExtractor()._extract_edge_feature(image)

        # Symmetry detection
        feature_stack[:, :, 4] = StructuralFeatureExtractor()._extract_symmetry_feature(image)

        # Texture analysis
        feature_stack[:, :, 5] = TextureFeatureExtractor()._extract_texture_feature(image)

        # Color distribution analysis
        feature_stack[:, :, 6] = TextureFeatureExtractor()._extract_color_feature(image)

        # Perceptual hash comparison
        feature_stack[:, :, 7] = TextureFeatureExtractor()._extract_hash_feature(image)

        # Create visualization image (optional)
        visualization = self._create_visualization(image, feature_stack)
        
        # Prepare metadata
        metadata = {
            "image_dimensions": (height, width),
            "feature_weights": self.config.feature_weights,
            "feature_statistics": self._compute_feature_statistics(feature_stack)
        }
        
        return visualization, feature_stack, metadata
    
    
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
            3D array with shape (n_patches_h, n_patches_w, n_features) containing
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
    
    def extract_batch_patch_features(self, images: List[np.ndarray], patch_size: int = 16) -> np.ndarray:
        """
        Batch processing version of extract_patch_features
        Returns array shaped as (batch_size, n_patches_h, n_patches_w, n_features)
        """
        if not images:
            return np.array([])
        
        # Validate consistent dimensions
        ref_height, ref_width = images[0].shape[:2]
        for img in images:
            if img.shape[:2] != (ref_height, ref_width):
                raise ValueError("All images in batch must have identical dimensions")
        
        # Pre-calculate patch grid
        n_patches_h = ref_height // patch_size
        n_patches_w = ref_width // patch_size
        trunc_h = n_patches_h * patch_size
        trunc_w = n_patches_w * patch_size
        
        # Pre-allocate arrays
        batch_size = len(images)
        n_features = len(self.config.feature_weights)
        batch_patches = np.zeros((batch_size, n_patches_h, n_patches_w, n_features), dtype=np.float32)
        
        # Vectorized processing
        for batch_idx, img in enumerate(images):
            # Extract features for current image
            _, feature_stack, _ = self.extract_all_features(img)
            
            # Truncate to integer patch multiples and limit to n_features
            truncated = feature_stack[:trunc_h, :trunc_w, :n_features]  # Fix here
            
            # Reshape into patches and compute means
            reshaped = truncated.reshape(n_patches_h, patch_size, n_patches_w, patch_size, n_features)
            batch_patches[batch_idx] = reshaped.mean(axis=(1, 3))  # Average over spatial dimensions
        
        return batch_patches