import tensorflow as tf
import numpy as np
from typing import Tuple
import global_config as config
import utils

class StructuralFeatureExtractor:
    # Define constants as class-level tf.constants
    EPSILON_TF = tf.constant(1e-8, dtype=tf.float32)
    
    # Pre-computed constants for better performance
    BGR_TO_GRAY_WEIGHTS = tf.constant([0.114, 0.587, 0.299], dtype=tf.float32)
    SOBEL_X_KERNEL = tf.constant([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], 
                                dtype=tf.float32, shape=[3, 3, 1, 1])
    SOBEL_Y_KERNEL = tf.constant([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], 
                                dtype=tf.float32, shape=[3, 3, 1, 1])
    
    # Normalization constants
    GRADIENT_MAX_VAL = tf.constant(4.0 * 255.0, dtype=tf.float32)
    FEATURE_MAX_VAL = tf.constant(100.0, dtype=tf.float32)
    
    # Canny edge detection kernels (TensorFlow-native implementation)
    GAUSSIAN_KERNEL = tf.constant([
        [1., 4., 7., 4., 1.],
        [4., 16., 26., 16., 4.],
        [7., 26., 41., 26., 7.],
        [4., 16., 26., 16., 4.],
        [1., 4., 7., 4., 1.]
    ], dtype=tf.float32) / 273.0
    GAUSSIAN_KERNEL = tf.reshape(GAUSSIAN_KERNEL, [5, 5, 1, 1])

    @staticmethod
    @tf.function
    def _rgb_to_grayscale(image: tf.Tensor) -> tf.Tensor:
        """Convert RGB/BGR image to grayscale using TensorFlow operations."""
        return tf.reduce_sum(image * StructuralFeatureExtractor.BGR_TO_GRAY_WEIGHTS, axis=-1)

    @staticmethod
    @tf.function
    def _apply_sobel_filters(gray: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply Sobel filters for gradient computation."""
        # Expand dims for conv2d (add batch and channel dimensions)
        gray_expanded = tf.expand_dims(tf.expand_dims(gray, 0), -1)
        
        grad_x = tf.nn.conv2d(gray_expanded, StructuralFeatureExtractor.SOBEL_X_KERNEL, 
                             strides=[1, 1, 1, 1], padding='SAME')
        grad_y = tf.nn.conv2d(gray_expanded, StructuralFeatureExtractor.SOBEL_Y_KERNEL, 
                             strides=[1, 1, 1, 1], padding='SAME')
        
        # Remove batch and channel dimensions
        grad_x = tf.squeeze(grad_x, [0, 3])
        grad_y = tf.squeeze(grad_y, [0, 3])
        
        return grad_x, grad_y

    @staticmethod
    @tf.function
    def _tf_native_canny(gray: tf.Tensor, low_threshold: float = 50.0, high_threshold: float = 150.0) -> tf.Tensor:
        """
        TensorFlow-native Canny edge detection implementation.
        Simplified but graph-compliant version.
        """
        # Step 1: Gaussian blur
        gray_expanded = tf.expand_dims(tf.expand_dims(gray, 0), -1)
        blurred = tf.nn.conv2d(gray_expanded, StructuralFeatureExtractor.GAUSSIAN_KERNEL,
                              strides=[1, 1, 1, 1], padding='SAME')
        blurred = tf.squeeze(blurred, [0, 3])
        
        # Step 2: Gradient computation
        grad_x, grad_y = StructuralFeatureExtractor._apply_sobel_filters(blurred)
        
        # Step 3: Magnitude and direction
        magnitude = tf.sqrt(grad_x**2 + grad_y**2 + StructuralFeatureExtractor.EPSILON_TF)
        
        # Step 4: Non-maximum suppression (simplified)
        # For graph compliance, we use a simplified version
        magnitude_padded = tf.pad(magnitude, [[1, 1], [1, 1]], mode='REFLECT')
        
        # Check if current pixel is local maximum in 3x3 neighborhood
        max_pooled = tf.nn.max_pool2d(
            tf.expand_dims(tf.expand_dims(magnitude_padded, 0), -1),
            ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID'
        )
        max_pooled = tf.squeeze(max_pooled, [0, 3])
        
        # Non-maximum suppression
        suppressed = tf.where(tf.equal(magnitude, max_pooled), magnitude, 0.0)
        
        # Step 5: Double thresholding
        strong_edges = tf.cast(suppressed > high_threshold, tf.float32)
        weak_edges = tf.cast(tf.logical_and(suppressed > low_threshold, 
                                          suppressed <= high_threshold), tf.float32) * 0.5
        
        edges = strong_edges + weak_edges
        
        return tf.clip_by_value(edges, 0.0, 1.0)

    @staticmethod
    @tf.function
    def _create_radial_gradient_map(h: tf.Tensor, w: tf.Tensor, score: tf.Tensor) -> tf.Tensor:
        """Create a radial gradient map for feature visualization."""
        center_y = tf.cast(h, tf.float32) / 2.0
        center_x = tf.cast(w, tf.float32) / 2.0
        
        y_coords = tf.range(tf.cast(h, tf.int32), dtype=tf.float32)
        x_coords = tf.range(tf.cast(w, tf.int32), dtype=tf.float32)
        
        yy, xx = tf.meshgrid(y_coords, x_coords, indexing='ij')
        
        distance = tf.sqrt((yy - center_y)**2 + (xx - center_x)**2)
        max_distance = tf.sqrt(center_y**2 + center_x**2) + StructuralFeatureExtractor.EPSILON_TF
        
        gradient = 1.0 - tf.minimum(distance / max_distance, 1.0)
        return gradient * score

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[config.patch_size, config.patch_size, 3], dtype=tf.float32)
    ])
    def _extract_gradient_feature(self, image: tf.Tensor) -> tf.Tensor:
        """
        Extract gradient perfection feature using pure TensorFlow operations.
        """
        with tf.name_scope('gradient_feature'):
            # Convert to grayscale
            gray = self._rgb_to_grayscale(image)
            gray = tf.cast(gray, tf.float32)
            
            # Compute gradients
            grad_x, grad_y = self._apply_sobel_filters(gray)
            
            # Compute gradient magnitude
            magnitude = tf.sqrt(grad_x**2 + grad_y**2 + self.EPSILON_TF)
            
            # Normalize magnitude
            magnitude_normalized = tf.clip_by_value(
                magnitude / (self.GRADIENT_MAX_VAL + self.EPSILON_TF), 0.0, 1.0
            )
            
            # Compute variance-based perfection score
            mean_magnitude = tf.reduce_mean(magnitude_normalized)
            variance_magnitude = tf.reduce_mean(tf.square(magnitude_normalized - mean_magnitude))
            
            # Invert variance for perfection score
            perfection_score = 1.0 / (variance_magnitude + self.EPSILON_TF)
            
            # Create feature map using radial gradient
            height = tf.shape(image)[0]
            width = tf.shape(image)[1]
            
            feature_map = self._create_radial_gradient_map(
                tf.cast(height, tf.float32), 
                tf.cast(width, tf.float32), 
                perfection_score
            )
            
            # Normalize and clip
            feature_map = tf.clip_by_value(feature_map / self.FEATURE_MAX_VAL, 0.0, 1.0)
            
            # Apply content mask
            content_mask = utils.create_content_mask(image)
            feature_map = feature_map * content_mask
            
            return feature_map

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[config.patch_size, config.patch_size, 3], dtype=tf.float32)
    ])
    def _extract_pattern_feature(self, image: tf.Tensor) -> tf.Tensor:
        """
        Extract repetitive pattern features using FFT (pure TensorFlow).
        """
        with tf.name_scope('pattern_feature'):
            # Convert to grayscale
            gray = self._rgb_to_grayscale(image)
            gray = tf.cast(gray, tf.float32)
            
            # Apply 2D FFT
            f_transform = tf.signal.fft2d(tf.cast(gray, tf.complex64))
            
            # Shift zero-frequency to center
            f_shift = tf.signal.fftshift(f_transform)
            
            # Compute magnitude spectrum
            magnitude_spectrum = tf.math.log(tf.abs(f_shift) + self.EPSILON_TF)
            
            # Calculate pattern score based on spectral energy distribution
            # High-frequency energy vs low-frequency energy ratio
            h, w = tf.shape(magnitude_spectrum)[0], tf.shape(magnitude_spectrum)[1]
            center_h, center_w = h // 2, w // 2
            
            # Create frequency masks
            y_coords = tf.range(h, dtype=tf.float32)
            x_coords = tf.range(w, dtype=tf.float32)
            yy, xx = tf.meshgrid(y_coords, x_coords, indexing='ij')
            
            # Distance from center
            dist_from_center = tf.sqrt(
                (yy - tf.cast(center_h, tf.float32))**2 + 
                (xx - tf.cast(center_w, tf.float32))**2
            )
            
            # High frequency mask (outer region)
            max_dist = tf.sqrt(tf.cast(center_h**2 + center_w**2, tf.float32))
            high_freq_mask = tf.cast(dist_from_center > max_dist * 0.3, tf.float32)
            low_freq_mask = 1.0 - high_freq_mask
            
            # Calculate energy in different frequency bands
            high_freq_energy = tf.reduce_sum(magnitude_spectrum * high_freq_mask)
            low_freq_energy = tf.reduce_sum(magnitude_spectrum * low_freq_mask)
            
            # Pattern score: ratio of high to low frequency energy
            pattern_score = high_freq_energy / (low_freq_energy + self.EPSILON_TF)
            
            # Create feature map
            height = tf.shape(image)[0]
            width = tf.shape(image)[1]
            
            feature_map = self._create_radial_gradient_map(
                tf.cast(height, tf.float32), 
                tf.cast(width, tf.float32), 
                pattern_score
            )
            
            # Normalize
            feature_map = tf.clip_by_value(feature_map / self.FEATURE_MAX_VAL, 0.0, 1.0)
            
            # Apply content mask
            content_mask = utils.create_content_mask(image)
            feature_map = feature_map * content_mask
            
            return feature_map

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[config.patch_size, config.patch_size, 3], dtype=tf.float32)
    ])
    def _extract_edge_feature(self, image: tf.Tensor) -> tf.Tensor:
        """
        Extract edge features using TensorFlow-native Canny implementation.
        """
        with tf.name_scope('edge_feature'):
            # Convert to grayscale
            gray = self._rgb_to_grayscale(image)
            gray = tf.cast(gray, tf.float32) * 255.0  # Scale to 0-255 range
            
            # Apply TensorFlow-native Canny edge detection
            edge_map = self._tf_native_canny(gray, low_threshold=50.0, high_threshold=150.0)
            
            # Apply content mask
            content_mask = utils.create_content_mask(image)
            feature_map = edge_map * content_mask
            
            return feature_map

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[config.patch_size, config.patch_size, 3], dtype=tf.float32)
    ])
    def _extract_symmetry_feature(self, image: tf.Tensor) -> tf.Tensor:
        """
        Extract symmetry features using TensorFlow operations.
        Simplified approach that detects horizontal and vertical symmetry.
        """
        with tf.name_scope('symmetry_feature'):
            # Convert to grayscale
            gray = self._rgb_to_grayscale(image)
            gray = tf.cast(gray, tf.float32)
            
            # Horizontal symmetry: compare left and right halves
            h, w = tf.shape(gray)[0], tf.shape(gray)[1]
            left_half = gray[:, :w//2]
            right_half = tf.reverse(gray[:, w//2:], axis=[1])
            
            # Ensure both halves have the same width
            min_width = tf.minimum(tf.shape(left_half)[1], tf.shape(right_half)[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
            
            # Calculate horizontal symmetry score
            h_diff = tf.abs(left_half - right_half)
            h_symmetry = 1.0 - tf.reduce_mean(h_diff) / (tf.reduce_max(gray) + self.EPSILON_TF)
            
            # Vertical symmetry: compare top and bottom halves
            top_half = gray[:h//2, :]
            bottom_half = tf.reverse(gray[h//2:, :], axis=[0])
            
            # Ensure both halves have the same height
            min_height = tf.minimum(tf.shape(top_half)[0], tf.shape(bottom_half)[0])
            top_half = top_half[:min_height, :]
            bottom_half = bottom_half[:min_height, :]
            
            # Calculate vertical symmetry score
            v_diff = tf.abs(top_half - bottom_half)
            v_symmetry = 1.0 - tf.reduce_mean(v_diff) / (tf.reduce_max(gray) + self.EPSILON_TF)
            
            # Combine symmetry scores
            symmetry_score = (h_symmetry + v_symmetry) / 2.0
            symmetry_score = tf.clip_by_value(symmetry_score, 0.0, 1.0)
            
            # Create feature map
            height = tf.shape(image)[0]
            width = tf.shape(image)[1]
            
            feature_map = self._create_radial_gradient_map(
                tf.cast(height, tf.float32), 
                tf.cast(width, tf.float32), 
                symmetry_score
            )
            
            # Apply content mask
            content_mask = utils.create_content_mask(image)
            feature_map = feature_map * content_mask
            
            return feature_map