import tensorflow as tf
import numpy as np # Keep for utility, but reduce direct usage where possible
from skimage.metrics import structural_similarity as ssim
from skimage.transform import radon
from skimage.feature import graycoprops, graycomatrix
import cv2

import global_config as config
import utils # For Canny edge detection and image resizing in NumPy parts

class StructuralFeatureExtractor:
    # Define EPSILON as a class-level tf.constant for TensorFlow operations
    EPSILON_TF = tf.constant(1e-8, dtype=tf.float32)
    # Define a separate NumPy version for use within tf.py_function calls (if any remain)
    EPSILON_NP = 1e-8

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[config.patch_size, config.patch_size, 3], dtype=tf.float32)
    ])
    def _extract_gradient_feature(self, image: tf.Tensor) -> tf.Tensor:
        """
        Extract gradient perfection feature which identifies unnaturally perfect gradients.

        Args:
            image: Input image (tf.Tensor, assumed to be in BGR format if color)

        Returns:
            2D feature map same size as input image with values between 0 and 1 (tf.Tensor)
        """
        with tf.name_scope('gradient_feature'):
            image_tensor = image # Already a tf.Tensor due to input_signature

            # Always convert to grayscale as the input signature specifies 3 channels
            gray_weights = tf.constant([0.114, 0.587, 0.299], dtype=tf.float32) # BGR to grayscale weights
            gray = tf.reduce_sum(image_tensor * gray_weights, axis=-1)

            # Ensure grayscale image is 2D
            gray = tf.cast(gray, tf.float32)
            
            # Compute gradients using Sobel filters
            sobel_x = tf.constant([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=tf.float32)
            sobel_y = tf.constant([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=tf.float32)
            
            # Reshape kernels for conv2d
            sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
            sobel_y = tf.reshape(sobel_y, [3, 3, 1, 1])
            
            # Expand dims for batch and channel for conv2d
            gray_expanded = tf.expand_dims(tf.expand_dims(gray, 0), -1)

            grad_x = tf.nn.conv2d(gray_expanded, sobel_x, strides=[1, 1, 1, 1], padding='SAME')
            grad_y = tf.nn.conv2d(gray_expanded, sobel_y, strides=[1, 1, 1, 1], padding='SAME')
            
            # Compute gradient magnitude and direction
            magnitude = tf.sqrt(grad_x**2 + grad_y**2 + StructuralFeatureExtractor.EPSILON_TF)
            # Normalize magnitude to 0-1 range based on typical max gradient value (e.g., 4 * 255 for 8-bit image range)
            magnitude_normalized = magnitude / (4.0 * 255.0 + StructuralFeatureExtractor.EPSILON_TF) # Max value for 8-bit image
            magnitude_normalized = tf.clip_by_value(magnitude_normalized, 0, 1)

            # Using variance of magnitude (lower variance means more uniform gradient, i.e., "perfect")
            magnitude_flat = tf.reshape(magnitude, [-1])
            variance_magnitude = tf.math.reduce_variance(magnitude_flat)

            # Invert variance to get a perfection score (higher score for lower variance)
            # Add epsilon to prevent division by zero for perfectly uniform (zero variance) areas
            perfection_score = 1.0 / (variance_magnitude + StructuralFeatureExtractor.EPSILON_TF)
            
            # Get original H, W from input image
            height, width = tf.shape(image)[0], tf.shape(image)[1]
            
            # Broadcast the scalar perfection_score to the image dimensions
            feature_map = tf.fill([height, width], perfection_score) # <<< Change: Direct broadcast
            
            # Normalize to 0-1 (example - adjust max_val based on empirical data)
            max_val = tf.constant(100.0, dtype=tf.float32) # Heuristic, adjust as needed based on empirical 'perfection_score' values
            feature_map = tf.clip_by_value(feature_map / max_val, 0.0, 1.0)
            
            content_mask = utils.create_content_mask(image) # Use the original input image for mask creation
            feature_map = feature_map * content_mask # Zero out features in black border regions

            return feature_map

    @staticmethod
    def _create_horizontal_gradient_map(h, w, score):
        center_x = tf.cast(w, tf.float32) / 2.0
        x_coords = tf.range(w, dtype=tf.float32)
        x_dist = tf.abs(x_coords - center_x) / (tf.cast(w, tf.float32) / 2.0 + StructuralFeatureExtractor.EPSILON_TF) # Use EPSILON_TF
        gradient = 1.0 - tf.minimum(x_dist, 1.0)
        gradient_2d = tf.ones([h, 1], dtype=tf.float32) * tf.reshape(gradient, [1, -1])
        return gradient_2d * score

    @staticmethod
    def _create_vertical_gradient_map(h, w, score):
        center_y = tf.cast(h, tf.float32) / 2.0
        y_coords = tf.range(h, dtype=tf.float32)
        y_dist = tf.abs(y_coords - center_y) / (tf.cast(h, tf.float32) / 2.0 + StructuralFeatureExtractor.EPSILON_TF) # Use EPSILON_TF
        gradient = 1.0 - tf.minimum(y_dist, 1.0)
        gradient_2d = tf.reshape(gradient, [-1, 1]) * tf.ones([1, w], dtype=tf.float32)
        return gradient_2d * score

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[config.patch_size, config.patch_size, 3], dtype=tf.float32)
    ])
    def _extract_pattern_feature(self, image: tf.Tensor) -> tf.Tensor:
        """
        Extract repetitive pattern features using FFT.

        Args:
            image: Input image (tf.Tensor)

        Returns:
            2D feature map same size as input image with values between 0 and 1 (tf.Tensor)
        """
        with tf.name_scope('pattern_feature'):
            image_tf = image

            gray_weights = tf.constant([0.114, 0.587, 0.299], dtype=tf.float32)
            gray = tf.reduce_sum(image_tf * gray_weights, axis=-1)

            # Ensure grayscale image is 2D
            gray = tf.cast(gray, tf.float32)

            # Apply 2D FFT
            f_transform = tf.signal.fft2d(tf.cast(gray, tf.complex64))
            
            # Shift the zero-frequency component to the center
            f_shift = tf.signal.fftshift(f_transform)
            
            # Compute magnitude spectrum (log scale for visualization/analysis)
            magnitude_spectrum = tf.math.log(tf.abs(f_shift) + StructuralFeatureExtractor.EPSILON_TF)

            # Calculate variance of the magnitude spectrum as a pattern score
            pattern_variance = tf.math.reduce_variance(tf.reshape(magnitude_spectrum, [-1]))

            # Invert variance to get a pattern score (higher score for lower variance, indicating more dominant patterns)
            pattern_score = 1.0 / (pattern_variance + StructuralFeatureExtractor.EPSILON_TF)

            # Broadcast the scalar pattern_score to the image dimensions
            height, width = tf.shape(image)[0], tf.shape(image)[1]
            feature_map = tf.fill([height, width], pattern_score)

            # Normalize to 0-1 (heuristic max_val)
            max_val = tf.constant(100.0, dtype=tf.float32) # Heuristic, adjust based on empirical 'pattern_score' values
            feature_map = tf.clip_by_value(feature_map / max_val, 0.0, 1.0)
            
            content_mask = utils.create_content_mask(image) # Use the original input image for mask creation
            feature_map = feature_map * content_mask # Zero out features in black border regions

            return feature_map

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[config.patch_size, config.patch_size, 3], dtype=tf.float32)
    ])
    def _extract_edge_feature(self, image: tf.Tensor) -> tf.Tensor:
        """
        Extract edge features to detect unnatural edge patterns.
        Using Canny edge detection.

        Args:
            image: Input image (tf.Tensor, BGR format if color)

        Returns:
            2D feature map same size as input image with values between 0 and 1 (tf.Tensor)
        """
        with tf.name_scope('edge_feature'):
            image_tf = image
            
            image_tf = tf.image.rgb_to_grayscale(tf.cast(image_tf, tf.uint8)) # Convert to grayscale

            # Canny edge detection is not directly available in TensorFlow.
            # We must use tf.py_function for now or implement a custom Canny-like filter.
            # For production, consider a custom tf.keras.layers.Conv2D based edge detector
            # or pre-train a small CNN for edge detection.
            
            def py_canny(image_tf_input): # Renamed arg to avoid confusion, it's a TF Tensor initially
                try:
                    image_np = image_tf_input.numpy() # Convert to NumPy array
                    
                    # Ensure image_np is 2D grayscale for Canny
                    if image_np.ndim == 3:
                        if image_np.shape[-1] == 3:
                            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) # Canny expects RGB if color
                        else: # Assuming single channel in last dim
                            image_np = np.squeeze(image_np, axis=-1)

                    # Convert to uint8 if not already
                    image_np = image_np.astype(np.uint8)

                    # Ensure we have the expected shape
                    if image_np.shape != (config.patch_size, config.patch_size):
                        # Resize if shape doesn't match
                        image_np = cv2.resize(image_np, (config.patch_size, config.patch_size))

                    # Apply Canny, thresholds can be tuned or made dynamic
                    edges = cv2.Canny(image_np, threshold1=100, threshold2=200)
                    
                    # Ensure output shape is correct
                    if edges.shape != (config.patch_size, config.patch_size):
                        edges = cv2.resize(edges, (config.patch_size, config.patch_size))
                    
                    # Normalize to 0-1
                    result = edges.astype(np.float32) / 255.0
                    
                    # Final shape guarantee
                    if result.shape != (config.patch_size, config.patch_size):
                        result = np.full((config.patch_size, config.patch_size), 0.0, dtype=np.float32)
                    
                    return result
                    
                except Exception as e:
                    # Return a default array if any error occurs
                    print(f"Warning: py_canny failed with error {e}, returning default array")
                    return np.full((config.patch_size, config.patch_size), 0.0, dtype=np.float32)

            edge_map = tf.py_function(py_canny, [image_tf], tf.float32)
            edge_map.set_shape(tf.TensorShape([config.patch_size, config.patch_size])) # Set shape info for graph compilation
            
            # Expand dims back if needed for stacking
            feature_map = edge_map
            
            content_mask = utils.create_content_mask(image) # Use the original input image for mask creation
            feature_map = feature_map * content_mask # Zero out features in black border regions

            return feature_map

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[config.patch_size, config.patch_size, 3], dtype=tf.float32)
    ])
    def _extract_symmetry_feature(self, image: tf.Tensor) -> tf.Tensor:
        """
        Extract symmetry features. AI-generated images might exhibit unnatural symmetries.
        This feature uses Radon transform to detect dominant angles and then assesses symmetry.

        Args:
            image: Input image (tf.Tensor)

        Returns:
            2D feature map same size as input image with values between 0 and 1 (tf.Tensor)
        """
        with tf.name_scope('symmetry_feature'):
            image_tf = image

            gray_weights = tf.constant([0.114, 0.587, 0.299], dtype=tf.float32)
            gray = tf.reduce_sum(image_tf * gray_weights, axis=-1)
            
            gray = tf.cast(gray, tf.float32)

            # Radon transform is not directly available in TensorFlow.
            # Using tf.py_function.
            def py_radon_symmetry(gray_tf_input): # Renamed arg to avoid confusion, it's a TF Tensor initially
                try:
                    gray_np = gray_tf_input.numpy() # Convert to NumPy array
                    
                    # Ensure input shape is correct
                    expected_shape = (config.patch_size, config.patch_size)
                    if gray_np.shape != expected_shape:
                        # Resize if shape doesn't match
                        gray_np = cv2.resize(gray_np, expected_shape)
                    
                    # Ensure input is float64 as skimage.radon expects it for accuracy
                    gray_np = gray_np.astype(np.float64)
                    
                    # Apply Radon transform with safe parameters
                    theta = np.linspace(0., 180., max(gray_np.shape), endpoint=False)
                    
                    # Ensure theta has reasonable length
                    if len(theta) == 0:
                        theta = np.linspace(0., 180., 180, endpoint=False)
                    
                    sinogram = radon(gray_np, theta=theta, circle=False)
                    
                    # Analyze sinogram for symmetry.
                    # A simple approach: Compare left half with right half, or symmetry around 90/180 degrees.
                    # For this example, let's consider the variance of projections across angles.
                    # Perfectly symmetric patterns might have very low variance across certain projection angles.
                    
                    # Calculate variance for each angle's projection with bounds checking
                    if sinogram.shape[1] > 0:
                        variance_per_angle = np.var(sinogram, axis=0)
                    else:
                        variance_per_angle = np.array([1.0])  # Default value
                    
                    # Ensure variance_per_angle is not empty
                    if len(variance_per_angle) == 0:
                        variance_per_angle = np.array([1.0])
                    
                    # Invert variance to get a "symmetry score"
                    # A lower variance suggests more uniform projections, potentially due to symmetry.
                    # Normalize and clip.
                    symmetry_score = 1.0 / (variance_per_angle + StructuralFeatureExtractor.EPSILON_NP)
                    
                    # Normalize symmetry_score to 0-1 with bounds checking
                    min_score = np.min(symmetry_score)
                    max_score = np.max(symmetry_score)
                    score_range = max_score - min_score
                    
                    if score_range > StructuralFeatureExtractor.EPSILON_NP:
                        normalized_symmetry_score = (symmetry_score - min_score) / score_range
                    else:
                        normalized_symmetry_score = np.zeros_like(symmetry_score)
                    
                    # As a feature map, we can broadcast this score to the image dimensions.
                    # For a more localized feature, one would analyze local patches.
                    # For simplicity, returning a global score broadcasted.
                    global_symmetry_score = np.mean(normalized_symmetry_score) # Average across angles
                    
                    # Ensure global_symmetry_score is finite
                    if not np.isfinite(global_symmetry_score):
                        global_symmetry_score = 0.0
                    
                    # Create a feature map of the same size as the input image
                    feature_map_np = np.full(expected_shape, global_symmetry_score, dtype=np.float32)
                    
                    # Final shape guarantee
                    if feature_map_np.shape != expected_shape:
                        feature_map_np = np.full(expected_shape, 0.0, dtype=np.float32)
                    
                    return feature_map_np
                    
                except Exception as e:
                    # Return a default array if any error occurs
                    print(f"Warning: py_radon_symmetry failed with error {e}, returning default array")
                    expected_shape = (config.patch_size, config.patch_size)
                    return np.full(expected_shape, 0.0, dtype=np.float32)

            feature_map = tf.py_function(py_radon_symmetry, [gray], tf.float32)
            
            # Get original H, W from input image to set exact shape
            feature_map.set_shape(tf.TensorShape([config.patch_size, config.patch_size])) # Set exact shape info

            content_mask = utils.create_content_mask(image) # Use the original input image for mask creation
            feature_map = feature_map * content_mask # Zero out features in black border regions

            return feature_map