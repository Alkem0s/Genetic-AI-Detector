# base_features.py

import tensorflow as tf
import utils


class BaseFeatureExtractor:
    """
    Base class for patch-level feature extractors.
    """

    EPSILON_TF: tf.Tensor = tf.constant(1e-8, dtype=tf.float32)

    @staticmethod
    @tf.function
    def _rgb_to_grayscale(image: tf.Tensor) -> tf.Tensor:
        """
        Convert an RGB/BGR float32 image patch to grayscale.

        Args:
            image: Float32 tensor of shape (H, W, 3), values in [0, 1].

        Returns:
            Float32 tensor of shape (H, W).
        """
        bgr_to_gray_weights = tf.constant([0.114, 0.587, 0.299], dtype=tf.float32)
        return tf.reduce_sum(image * bgr_to_gray_weights, axis=-1)

    @staticmethod
    @tf.function
    def _create_content_mask(image: tf.Tensor) -> tf.Tensor:
        """
        Creates a binary mask where 1 indicates non-black (content) pixels
        and 0 indicates black (border) pixels. Assumes black borders are (0,0,0) or 0.
        
        Args:
            image: Input image tf.Tensor (H, W, C or H, W, dtype=tf.float32, normalized to 0-1 or 0-255).
                It's important that black borders are exactly 0.

        Returns:
            tf.Tensor: A 2D binary mask (H, W, dtype=tf.float32).
        """
        # Sum absolute values across all channels (last axis). Works for any channel
        # count (1, 3, N) inside @tf.function without Python-level branching.
        pixel_intensity_sum = tf.reduce_sum(tf.abs(image), axis=-1)

        # Create a binary mask: 1 where sum > 0 (content), 0 where sum == 0 (black border)
        content_mask = tf.cast(pixel_intensity_sum > 0, tf.float32)
        
        return content_mask

    @staticmethod
    @tf.function
    def _apply_mask(feature_map: tf.Tensor, image: tf.Tensor) -> tf.Tensor:
        """
        Zero out feature-map values that fall inside black border regions.

        Args:
            feature_map: Float32 tensor of shape (H, W).
            image:       Original float32 image patch of shape (H, W, 3),
                         used to derive the content mask.

        Returns:
            Float32 tensor of shape (H, W) with border pixels zeroed.
        """
        content_mask = BaseFeatureExtractor._create_content_mask(image)
        return feature_map * content_mask