import tensorflow as tf
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from skimage.feature import graycoprops, graycomatrix
from PIL import Image
import imagehash

import global_config
import utils

class TextureFeatureExtractor:
    # Define EPSILON as a class-level tf.constant for TensorFlow operations
    EPSILON_TF = tf.constant(1e-8, dtype=tf.float32)
    # Define a separate NumPy version for use within tf.py_function calls
    EPSILON_NP = 1e-8

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[global_config.default_patch_size, global_config.default_patch_size, 3], dtype=tf.float32)
    ])
    def _extract_noise_feature(self, image: tf.Tensor) -> tf.Tensor:
        """
        Analyze noise distribution to detect AI-generated inconsistencies.

        Args:
            image: Input image (tf.Tensor, BGR format if color)

        Returns:
            2D feature map same size as input image with values between 0 and 1 (tf.Tensor)
        """
        with tf.name_scope('noise_feature'):
            image_tf = image # Already a tf.Tensor due to input_signature
            
            gray_weights = tf.constant([0.114, 0.587, 0.299], dtype=tf.float32) # BGR to grayscale weights
            gray = tf.reduce_sum(image_tf * gray_weights, axis=-1)

            # Compute noise (e.g., using a high-pass filter or simple difference)
            # Example: Laplacian filter for noise
            laplacian_kernel = tf.constant([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=tf.float32)
            laplacian_kernel = tf.reshape(laplacian_kernel, [3, 3, 1, 1])
            
            # Expand dims for batch and channel for conv2d
            gray_expanded = tf.expand_dims(tf.expand_dims(gray, 0), -1)
            
            noise = tf.nn.conv2d(gray_expanded, laplacian_kernel, strides=[1, 1, 1, 1], padding='SAME')
            noise = tf.abs(noise) # Absolute value of noise response
            noise = tf.squeeze(noise) # Remove batch and channel dims

            # --- TensorFlow native implementation for median and MAD ---
            # Median implementation for a flattened tensor
            def tf_median(tensor_flat):
                sorted_tensor = tf.sort(tensor_flat)
                size = tf.shape(sorted_tensor)[0]
                # For even size, take average of middle two; for odd, take middle element
                median_val = tf.cond(tf.equal(size % 2, 0),
                                     lambda: (sorted_tensor[size // 2 - 1] + sorted_tensor[size // 2]) / 2.0,
                                     lambda: sorted_tensor[size // 2])
                return median_val

            # Global Noise Analysis
            flat_noise = tf.reshape(noise, [-1])
            global_median_noise = tf_median(flat_noise)
            
            # Median Absolute Deviation (MAD)
            # abs_diff_from_median = tf.abs(flat_noise - global_median_noise)
            # global_mad = tf_median(abs_diff_from_median)

            # NOTE: Implementing MAD purely with tf_median can be tricky for arbitrary batching/patching.
            # If the MAD calculation is critical and `tf_median` is too slow or complex
            # for `tf.map_fn` over patches, consider keeping `tf.py_function` for MAD or
            # re-evaluating the specific noise metric.
            # For this example, we'll implement a simplified noise variance feature instead of MAD.
            global_noise_variance = tf.math.reduce_variance(flat_noise)
            
            # Normalize noise variance to get a feature value (e.g., invert, then clip)
            # Higher variance -> more "natural" noise, lower variance -> potentially "AI-generated" noise
            # So, perfection_score = 1 - normalized_variance.
            # Max expected variance heuristic.
            max_noise_variance_heuristic = tf.constant(5000.0, dtype=tf.float32) # Adjust based on data
            normalized_variance = global_noise_variance / (max_noise_variance_heuristic + TextureFeatureExtractor.EPSILON_TF)
            normalized_variance = tf.clip_by_value(normalized_variance, 0, 1)
            
            # Feature: Uniformity of noise (1 - normalized_variance)
            noise_uniformity_score = 1.0 - normalized_variance

            # Create a feature map of the same size as the input image by broadcasting the score
            feature_map = tf.fill(tf.shape(gray), noise_uniformity_score)
            
            content_mask = utils.create_content_mask(image) # Use the original input image for mask creation
            feature_map = feature_map * content_mask # Zero out features in black border regions

            return feature_map

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[global_config.default_patch_size, global_config.default_patch_size, 3], dtype=tf.float32)
    ])
    def _extract_texture_feature(self, image: tf.Tensor) -> tf.Tensor:
        """
        Extract texture features (e.g., using GLCM - Gray Level Co-occurrence Matrix)
        or Local Binary Patterns (LBP).

        Args:
            image: Input image (tf.Tensor)

        Returns:
            2D feature map same size as input image with values between 0 and 1 (tf.Tensor)
        """
        with tf.name_scope('texture_feature'):
            image_tf = image

            gray_weights = tf.constant([0.114, 0.587, 0.299], dtype=tf.float32)
            gray = tf.reduce_sum(image_tf * gray_weights, axis=-1)
            
            # Cast to uint8 for skimage functions
            gray_uint8 = tf.cast(gray, tf.uint8)

            # LBP is used here as GLCM is more complex to implement in TF natively
            # LBP is not available in TensorFlow natively. Using tf.py_function.
            def py_lbp(image_tf_input): # Renamed arg to avoid confusion, it's a TF Tensor initially
                image_np_uint8 = image_tf_input.numpy() # Convert to NumPy array
                
                # Normalize image to 0-255 for LBP if not already
                # For robust LBP, consider different methods and radii.
                # Using 'default' method and radius 1, n_points 8
                radius = 1
                n_points = 8 * radius
                lbp_image = local_binary_pattern(image_np_uint8, n_points, radius, method='uniform')
                
                # Normalize LBP output to 0-1 range based on max possible LBP value
                # Max value for 'uniform' method with 8 points is 58 (for 8-bit image)
                max_lbp_val = np.max(lbp_image) # dynamic based on image
                if max_lbp_val == 0:
                    return np.zeros_like(lbp_image, dtype=np.float32)
                return (lbp_image / max_lbp_val).astype(np.float32)

            texture_map = tf.py_function(py_lbp, [gray_uint8], tf.float32)
            texture_map.set_shape(tf.TensorShape([global_config.default_patch_size, global_config.default_patch_size])) # Set shape info
            
            # Rescale to original image size if LBP changed dims (it usually doesn't)
            # feature_map = tf.image.resize(tf.expand_dims(texture_map, -1), 
            #                               tf.shape(image_tf)[:2], 
            #                               method=tf.image.ResizeMethod.BILINEAR)
            # feature_map = tf.squeeze(feature_map, axis=-1)
            
            feature_map = texture_map

            content_mask = utils.create_content_mask(image) # Use the original input image for mask creation
            feature_map = feature_map * content_mask # Zero out features in black border regions

            return feature_map

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[global_config.default_patch_size, global_config.default_patch_size, 3], dtype=tf.float32)
    ])
    def _extract_color_feature(self, image: tf.Tensor) -> tf.Tensor:
        """
        Extract color anomaly features. AI-generated images might have unusual color distributions.

        Args:
            image: Input image (tf.Tensor, BGR format if color)

        Returns:
            2D feature map same size as input image with values between 0 and 1 (tf.Tensor)
        """
        with tf.name_scope('color_feature'):
            image_tf = image # Assumed to be in BGR or RGB as input

            # If grayscale, color features are not applicable; return zero map
            if tf.shape(image_tf)[-1] != 3:
                return tf.zeros(tf.shape(image_tf)[:2], dtype=tf.float32)

            # Convert to HSV color space for better color analysis
            # TF's rgb_to_hsv expects RGB, so convert BGR to RGB first if necessary
            # Assuming input is BGR based on comments in other functions.
            image_rgb = image_tf[...,::-1] # Swap B and R channels
            hsv_image = tf.image.rgb_to_hsv(image_rgb)
            
            # Analyze saturation and hue distributions for anomalies
            # High saturation variance or unusual hue clusters might indicate AI generation.
            hue = hsv_image[..., 0] # 0-1 range
            saturation = hsv_image[..., 1] # 0-1 range
            value = hsv_image[..., 2] # 0-1 range

            # Example anomaly detection: very high saturation variance or unnatural hue shifts
            # A simple metric could be local saturation variance or hue uniformity.
            # For simplicity, calculate the variance of saturation across the image.
            saturation_variance = tf.math.reduce_variance(tf.reshape(saturation, [-1]))
            
            # Invert variance to get a "color perfection" score (lower variance -> more "perfect" color)
            # Normalize to 0-1 based on heuristic max variance
            max_sat_variance_heuristic = tf.constant(0.1, dtype=tf.float32) # Adjust based on data
            normalized_sat_variance = saturation_variance / (max_sat_variance_heuristic + TextureFeatureExtractor.EPSILON_TF)
            normalized_sat_variance = tf.clip_by_value(normalized_sat_variance, 0, 1)
            
            color_perfection_score = 1.0 - normalized_sat_variance
            
            # Create a feature map of the same size as the input image by broadcasting the score
            feature_map = tf.fill(tf.shape(image_tf)[:2], color_perfection_score)
            
            content_mask = utils.create_content_mask(image) # Use the original input image for mask creation
            feature_map = feature_map * content_mask # Zero out features in black border regions

            return feature_map

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[global_config.default_patch_size, global_config.default_patch_size, 3], dtype=tf.float32)
    ])
    def _extract_hash_feature(self, image: tf.Tensor) -> tf.Tensor:
        """
        Extract perceptual hash features. AI-generated images might have unusual hash patterns.
        This feature is typically a global score.

        Args:
            image: Input image (tf.Tensor, BGR format if color)

        Returns:
            2D feature map same size as input image with values between 0 and 1 (tf.Tensor)
        """
        with tf.name_scope('hash_feature'):
            # Imagehashing libraries are not TensorFlow native. Use tf.py_function.
            # Make sure the input image is converted to appropriate format for numpy/PIL.
            
            def py_hash_analysis(image_tf_input, epsilon_np_val): # Renamed arg
                try:
                    if not hasattr(Image, 'ANTIALIAS'):
                        Image.ANTIALIAS = Image.LANCZOS

                    image_np = image_tf_input.numpy() # Convert to NumPy array
                    
                    if image_np.shape[-1] == 3:
                        # imagehash expects RGB, so convert if input is BGR
                        image_pil = Image.fromarray(cv2.cvtColor(image_np.astype(np.uint8), cv2.COLOR_BGR2RGB))
                    else:
                        image_pil = Image.fromarray(image_np.astype(np.uint8))
                    
                    phash = imagehash.phash(image_pil, hash_size=8)
                    dhash = imagehash.dhash(image_pil, hash_size=8)
                    
                    # For simplicity, returning a scalar based on a simple combination.
                    # In a real scenario, you'd compare these hashes to a database.
                    # Example: a score based on Hamming distance to some 'ideal' hash.
                    # For a feature map, this global score is broadcast.
                    
                    # Dummy score: average of phash and dhash (as integers, then normalized)
                    # A 64-bit hash (8x8) has 64 bits. Sum of set bits can be up to 64.
                    # Normalizing by 64 means score is between 0 and 1.
                    dummy_score = (phash.hash.sum() + dhash.hash.sum()) / (2 * 64) 
                    return np.float32(dummy_score) # <<< Change: Return a scalar np.float32
                except Exception as e:
                    tf.print(f"Error in py_hash_analysis: {e}")
                    return np.float32(0.0) # <<< Change: Return a scalar np.float32

            hash_score_tensor = tf.py_function(
                func=py_hash_analysis,
                inp=[image, TextureFeatureExtractor.EPSILON_NP], # Pass image_tf directly
                Tout=tf.float32
            )
            # Ensure the output shape is known for graph compilation.
            # Since it's now guaranteed to be a scalar, it will be a 0D tensor (scalar).
            hash_score_tensor.set_shape([]) # <<< Change: Set shape to scalar (rank 0)
            
            # Broadcast the scalar hash score to create a 2D feature map of the image's size
            height, width = tf.shape(image)[0], tf.shape(image)[1]
            feature_map = tf.fill([height, width], hash_score_tensor) # <<< Change: Use the scalar value directly to fill
            
            content_mask = utils.create_content_mask(image) # Use the original input image for mask creation
            feature_map = feature_map * content_mask # Zero out features in black border regions

            return feature_map