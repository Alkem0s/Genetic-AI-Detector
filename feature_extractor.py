import tensorflow as tf
from typing import Tuple, Dict, List, Any

import global_config
from structural_features import StructuralFeatureExtractor
from texture_features import TextureFeatureExtractor

class FeatureExtractor:
    """
    Class for feature extraction in AI image detection system.
    Implements methods to extract various features that can identify AI-generated artifacts.
    
    The GeneticFeatureOptimizer expects this module to extract features
    from patches of images and return them in a consistent format.
    """
    def extract_all_features(self, image: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, Dict[str, Any]]:
        """
        Extract all features from an image.
        
        This is the main method that the genetic algorithm will call.
        It should extract all features and return them in a standardized format.
        
        Args:
            image: Input image as tensorflow tensor (height, width, channels)
            
        Returns:
            Tuple containing:
                - visualization: Visualization image with highlighted features (can be None)
                - feature_stack: 3D tensor where each channel is a different feature map
                - metadata: Dictionary with additional metadata about extracted features
        """
        # Ensure the image is a tensor with float32 data type
        image = tf.cast(image, tf.float32)
        
        # Get image dimensions
        height, width = tf.shape(image)[0], tf.shape(image)[1]
        
        # Create a list to hold all feature maps
        feature_maps = []

        structural_features_extractor = StructuralFeatureExtractor()
        texture_features_extractor = TextureFeatureExtractor()

        # Add each feature to the list
        feature_maps.append(structural_features_extractor._extract_gradient_feature(image))
        feature_maps.append(structural_features_extractor._extract_pattern_feature(image))
        feature_maps.append(texture_features_extractor._extract_noise_feature(image))
        feature_maps.append(structural_features_extractor._extract_edge_feature(image))
        feature_maps.append(structural_features_extractor._extract_symmetry_feature(image))
        feature_maps.append(texture_features_extractor._extract_texture_feature(image))
        feature_maps.append(texture_features_extractor._extract_color_feature(image))
        feature_maps.append(texture_features_extractor._extract_hash_feature(image))

        # Stack all feature maps along the channel dimension
        feature_stack = tf.stack(feature_maps, axis=-1)
        
        return feature_stack
    
    def extract_patch_features(self, image: tf.Tensor, patch_size: int = 16) -> tf.Tensor:
        """
        Extract average feature values for each patch in the image.
        This function is particularly useful for the genetic algorithm's rule application.
        
        Args:
            image: Input image as tensor
            patch_size: Size of patches to analyze
            
        Returns:
            3D tensor with shape (n_patches_h, n_patches_w, n_features) containing
            average feature values for each patch
        """
        # Extract all features
        feature_stack = self.extract_all_features(image)
        
        height, width = tf.shape(feature_stack)[0], tf.shape(feature_stack)[1]
        n_patches_h = height // patch_size
        n_patches_w = width // patch_size
        n_features = tf.shape(feature_stack)[-1]
        
        # Truncate to integer patch multiples
        truncated = feature_stack[:n_patches_h * patch_size, :n_patches_w * patch_size, :]
        
        # Reshape into patches and compute means using TensorFlow operations
        reshaped = tf.reshape(truncated, [n_patches_h, patch_size, n_patches_w, patch_size, n_features])
        patch_features = tf.reduce_mean(reshaped, axis=[1, 3])  # Average over spatial dimensions within each patch
        
        return patch_features
    
    def extract_batch_patch_features(self, images: List[tf.Tensor], patch_size: int = 16) -> tf.Tensor:
        """
        Batch processing version of extract_patch_features
        Returns tensor shaped as (batch_size, n_patches_h, n_patches_w, n_features)
        """
        if isinstance(images, tf.Tensor):
            if tf.size(images) == 0:
                return tf.zeros([0])
        elif not images:
            return tf.zeros([0])
        
        # Validate consistent dimensions
        ref_height, ref_width = tf.shape(images[0])[0], tf.shape(images[0])[1]
        
        # Process each image and collect results in a list
        all_patch_features = []
        for img in images:
            # Validate dimensions
            current_height, current_width = tf.shape(img)[0], tf.shape(img)[1]
            tf.debugging.assert_equal(
                current_height, 
                ref_height, 
                message="All images in batch must have identical height"
            )
            tf.debugging.assert_equal(
                current_width, 
                ref_width, 
                message="All images in batch must have identical width"
            )
            
            # Extract features for current image
            patch_features = self.extract_patch_features(img, patch_size)
            all_patch_features.append(patch_features)
        
        # Stack all patch features into a batch
        batch_patches = tf.stack(all_patch_features, axis=0)
        
        return batch_patches