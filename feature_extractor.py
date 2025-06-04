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
    def __init__(self):
        self.structural_extractor = StructuralFeatureExtractor()
        self.texture_extractor = TextureFeatureExtractor()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[global_config.image_size, global_config.image_size, 3], dtype=tf.float32)
    ])
    def extract_patch_features(self, image: tf.Tensor) -> tf.Tensor:
        image_expanded = tf.expand_dims(image, 0)

        static_patch_size = global_config.default_patch_size

        patches = tf.image.extract_patches(
            images=image_expanded,
            sizes=[1, static_patch_size, static_patch_size, 1],
            strides=[1, static_patch_size, static_patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        
        patches_shape = tf.shape(patches)
        n_patches_h = patches_shape[1]
        n_patches_w = patches_shape[2]

        patches = tf.reshape(patches, [-1, static_patch_size, static_patch_size, 3])

        patch_features = tf.map_fn(
            lambda single_patch: self._extract_single_patch_features(single_patch),
            patches,
            fn_output_signature=tf.TensorSpec(shape=[static_patch_size, static_patch_size, len(global_config.feature_weights)], dtype=tf.float32)
        )

        patch_features = tf.reduce_mean(patch_features, axis=[1, 2])
        
        num_features = len(global_config.feature_weights)
        patch_features = tf.reshape(patch_features, [n_patches_h, n_patches_w, num_features])
        
        return patch_features
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, global_config.image_size, global_config.image_size, 3], dtype=tf.float32) # Batch of images (batch_size, H, W, C)
    ])
    def extract_batch_patch_features(self, images: tf.Tensor) -> tf.Tensor:
        """
        Batch processing version of extract_patch_features.
        Returns tensor shaped as (batch_size, n_patches_h, n_patches_w, n_features).

        Args:
            images: Input batch of images as a single TensorFlow tensor (batch_size, height, width, channels).
            patch_size: Size of patches to analyze.

        Returns:
            4D tensor with shape (batch_size, n_patches_h, n_patches_w, n_features) containing
            average feature values for each patch across the batch.
        """
        if tf.size(images) == 0:
            # Use static patch size for calculations
            static_patch_size = global_config.default_patch_size
            dummy_image_height = tf.constant(global_config.image_size, dtype=tf.float32)
            dummy_image_width = tf.constant(global_config.image_size, dtype=tf.float32)
            patch_size_float = tf.constant(static_patch_size, dtype=tf.float32)

            # Number of patches in height/width direction using floor division (mimicking 'VALID' padding)
            n_patches_h = tf.cast(tf.floor(dummy_image_height / patch_size_float), tf.int32)
            n_patches_w = tf.cast(tf.floor(dummy_image_width / patch_size_float), tf.int32)
            
            return tf.zeros([0, n_patches_h, n_patches_w, len(global_config.feature_weights)], dtype=tf.float32)

        # Apply extract_patch_features to each image in the batch
        batch_patches = tf.map_fn(
            lambda img: self.extract_patch_features(img),
            images,
            fn_output_signature=tf.TensorSpec(shape=[None, None, len(global_config.feature_weights)], dtype=tf.float32)
        )
        return batch_patches


    @tf.function(input_signature=[
        tf.TensorSpec(shape=[global_config.default_patch_size, global_config.default_patch_size, 3], dtype=tf.float32)
    ])
    def _extract_single_patch_features(self, patch: tf.Tensor) -> tf.Tensor:
        """
        Extracts all individual features from a single patch.
        Returns a 3D tensor where each channel is a feature map for the patch.
        Shape: [patch_size, patch_size, num_features]
        """
        gradient_feature = self.structural_extractor._extract_gradient_feature(patch)
        pattern_feature = self.structural_extractor._extract_pattern_feature(patch)
        edge_feature = self.structural_extractor._extract_edge_feature(patch)
        symmetry_feature = self.structural_extractor._extract_symmetry_feature(patch)

        noise_feature = self.texture_extractor._extract_noise_feature(patch)
        texture_feature = self.texture_extractor._extract_texture_feature(patch)
        color_feature = self.texture_extractor._extract_color_feature(patch)
        hash_feature = self.texture_extractor._extract_hash_feature(patch)

        # Stack all feature maps. Each feature map is [patch_size, patch_size].
        # Stacking along a new axis to get [patch_size, patch_size, num_features]
        feature_stack = tf.stack([
            gradient_feature,
            pattern_feature,
            edge_feature,
            symmetry_feature,
            noise_feature,
            texture_feature,
            color_feature,
            hash_feature
        ], axis=-1)
        
        return feature_stack