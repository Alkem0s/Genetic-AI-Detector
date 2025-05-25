import tensorflow as tf
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.transform import radon
from skimage.feature import graycoprops, graycomatrix
import cv2 # For Canny edge detection and image resizing in NumPy parts

class StructuralFeatureExtractor:
    # Define EPSILON as a class-level tf.constant for TensorFlow operations
    EPSILON_TF = tf.constant(1e-8, dtype=tf.float32)
    # Define a separate NumPy version for use within tf.py_function calls
    EPSILON_NP = 1e-8

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32) # Allow dynamic H, W but fixed C
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

            if tf.shape(image_tensor)[-1] == 3:
                gray_weights = tf.constant([0.114, 0.587, 0.299], dtype=tf.float32)
                gray = tf.reduce_sum(image_tensor * gray_weights, axis=-1)
            else:
                gray = image_tensor
            
            gray = tf.cond(tf.reduce_max(gray) <= 1.0, 
                           lambda: gray * 255.0, 
                           lambda: gray)
                
            scharr_x = tf.constant([
                [3, 0, -3],
                [10, 0, -10],
                [3, 0, -3]
            ], dtype=tf.float32)
            
            scharr_y = tf.constant([
                [3, 10, 3],
                [0, 0, 0],
                [-3, -10, -3]
            ], dtype=tf.float32)
            
            scharr_x = tf.reshape(scharr_x, [3, 3, 1, 1])
            scharr_y = tf.reshape(scharr_y, [3, 3, 1, 1])
            
            gray_expanded = tf.expand_dims(tf.expand_dims(gray, 0), -1)
            
            scharrx = tf.nn.conv2d(gray_expanded, scharr_x, strides=[1, 1, 1, 1], padding='SAME')
            scharry = tf.nn.conv2d(gray_expanded, scharr_y, strides=[1, 1, 1, 1], padding='SAME')
            
            scharrx = tf.squeeze(scharrx, [0])
            scharry = tf.squeeze(scharry, [0])
            
            scharrx = tf.squeeze(scharrx, [-1])
            scharry = tf.squeeze(scharry, [-1])

            magnitude = tf.sqrt(tf.square(scharrx) + tf.square(scharry))
            direction = tf.atan2(scharry, scharrx)

            patch_size = 16
            h, w = tf.shape(gray)[0], tf.shape(gray)[1]
            
            # Using tf.cast(h / patch_size, tf.int32) directly for map dimensions
            pm_h = h // patch_size
            pm_w = w // patch_size
            
            magnitude_reshaped = tf.expand_dims(tf.expand_dims(magnitude, 0), -1)
            direction_reshaped = tf.expand_dims(tf.expand_dims(direction, 0), -1)

            patches_magnitude = tf.image.extract_patches(
                images=magnitude_reshaped,
                sizes=[1, patch_size, patch_size, 1],
                strides=[1, patch_size, patch_size, 1],
                rates=[1, 1, 1, 1],
                padding='VALID'
            )
            patches_direction = tf.image.extract_patches(
                images=direction_reshaped,
                sizes=[1, patch_size, patch_size, 1],
                strides=[1, patch_size, patch_size, 1],
                rates=[1, 1, 1, 1],
                padding='VALID'
            )

            patches_magnitude = tf.reshape(patches_magnitude, [-1, patch_size * patch_size])
            patches_direction = tf.reshape(patches_direction, [-1, patch_size * patch_size])

            patch_mean_mag = tf.reduce_mean(patches_magnitude, axis=1)
            patch_std_mag = tf.math.reduce_std(patches_magnitude, axis=1)

            coef_var = patch_std_mag / (patch_mean_mag + StructuralFeatureExtractor.EPSILON_TF) # Use EPSILON_TF
            coef_var_clipped = tf.minimum(coef_var, 1.0)
            perfection_score = 1 - coef_var_clipped

            complex_dir_patches = tf.complex(tf.cos(patches_direction), tf.sin(patches_direction))
            
            complex_dir_patches_reshaped = tf.reshape(complex_dir_patches, [-1, patch_size, patch_size])
            
            fft_dir = tf.signal.fft2d(complex_dir_patches_reshaped)
            power_spectrum = tf.square(tf.abs(fft_dir))
            
            total_power = tf.reduce_sum(power_spectrum, axis=[1, 2])

            center_power = tf.cond(
                tf.greater_equal(patch_size, 3),
                lambda: tf.reduce_sum(power_spectrum[:, :3, :3], axis=[1, 2]),
                lambda: total_power
            )

            spectral_concentration = tf.where(total_power > 0, center_power / total_power, 0.0)

            perfection_map_flat = perfection_score * (0.5 + 0.5 * spectral_concentration)
            perfection_map = tf.reshape(perfection_map_flat, [pm_h, pm_w])
            
            high_perfection_ratio = tf.reduce_sum(
                tf.cast(perfection_map > 0.7, tf.float32)
            ) / tf.cast(tf.size(perfection_map), tf.float32)
            
            perfection_map = tf.cond(
                high_perfection_ratio > 0.5,
                lambda: perfection_map * (0.5 + high_perfection_ratio * 0.5),
                lambda: perfection_map
            )
            
            feature_map = tf.image.resize(
                tf.expand_dims(tf.expand_dims(perfection_map, 0), -1),
                [h, w],
                method=tf.image.ResizeMethod.BILINEAR
            )
            
            feature_map = tf.squeeze(feature_map)
            feature_map = tf.clip_by_value(feature_map, 0, 1)
            
            return feature_map # Return tf.Tensor
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32)
    ])
    def _extract_pattern_feature(self, image: tf.Tensor) -> tf.Tensor:
        """
        Detect repeating patterns that may indicate AI artifacts.
        
        Args:
            image: Input image (tf.Tensor, BGR format if color)
            
        Returns:
            2D feature map same size as input image with values between 0 and 1 (tf.Tensor)
        """
        with tf.name_scope('pattern_feature'):
            image_tensor = image # Already a tf.Tensor

            if tf.shape(image_tensor)[-1] == 3:
                gray_weights = tf.constant([0.114, 0.587, 0.299], dtype=tf.float32)
                gray = tf.reduce_sum(image_tensor * gray_weights, axis=-1)
            else:
                gray = image_tensor
            
            gray = tf.cond(tf.reduce_max(gray) <= 1.0, 
                           lambda: gray * 255.0, 
                           lambda: gray)
            
            def py_pattern_analysis(gray_np_uint8_tensor, epsilon_np_val): # Pass EPSILON_NP
                gray_np_uint8 = gray_np_uint8_tensor.numpy() # Convert EagerTensor to NumPy array
                f_transform = np.fft.fft2(gray_np_uint8)
                f_shift = np.fft.fftshift(f_transform)
                magnitude_spectrum = 20 * np.log(np.abs(f_shift) + epsilon_np_val)
                
                mag_min = magnitude_spectrum.min()
                mag_max = magnitude_spectrum.max()
                magnitude_spectrum_norm = ((magnitude_spectrum - mag_min) / 
                                          (mag_max - mag_min)) if (mag_max - mag_min) > 0 else np.zeros_like(magnitude_spectrum)
            
                mean_spectrum = np.mean(magnitude_spectrum_norm)
                std_spectrum = np.std(magnitude_spectrum_norm)
                adaptive_threshold = mean_spectrum + 0.7 * std_spectrum
                peaks = magnitude_spectrum_norm > adaptive_threshold
            
                h_np, w_np = gray_np_uint8.shape
                Y, X = np.ogrid[:h_np, :w_np]
                center = (h_np//2, w_np//2)
                radius = min(h_np, w_np) // 20
                peaks[np.sqrt((X - center[1])**2 + (Y - center[0])**2) <= radius] = False
            
                peak_count = np.sum(peaks)
                pattern_score = 0.0
                
                if peak_count > 5:
                    py, px = np.where(peaks)
                    distances = np.sqrt((px - center[1])**2 + (py - center[0])**2)
                    radial_hist, _ = np.histogram(distances, bins=20)
                    radial_hist = radial_hist / radial_hist.sum() if radial_hist.sum() > 0 else radial_hist
                    radial_entropy = -np.sum(radial_hist * np.log2(radial_hist + epsilon_np_val))
                    max_entropy = np.log2(len(radial_hist))
                    radial_score = 1 - (radial_entropy / max_entropy) if max_entropy > 0 else 0
            
                    angles = np.arctan2(py - center[0], px - center[1])
                    angle_hist, _ = np.histogram(angles, bins=16, range=(-np.pi, np.pi))
                    angle_hist = angle_hist / angle_hist.sum() if angle_hist.sum() > 0 else angle_hist
                    angle_entropy = -np.sum(angle_hist * np.log2(angle_hist + epsilon_np_val))
                    angle_score = 1 - (angle_entropy / np.log2(16)) if angle_hist.size > 0 else 0
            
                    pattern_score = radial_score * (1 - 0.5 * angle_score)
            
                f_shift_filtered = f_shift * peaks.astype(np.complex64)
                spatial_pattern = np.abs(np.fft.ifft2(np.fft.ifftshift(f_shift_filtered)))
                spatial_pattern_min = spatial_pattern.min()
                spatial_pattern_ptp = np.ptp(spatial_pattern)
                spatial_pattern_norm = (spatial_pattern - spatial_pattern_min) / (spatial_pattern_ptp + epsilon_np_val)
                
                feature_map_np = np.clip(spatial_pattern_norm * pattern_score, 0, 1)
                return feature_map_np.astype(np.float32)

            feature_map = tf.py_function(
                py_pattern_analysis,
                [tf.cast(gray, tf.uint8), StructuralFeatureExtractor.EPSILON_NP], # Pass EPSILON_NP
                tf.float32
            )
            # Use tf.TensorShape(gray.shape) to ensure the shape is explicitly set based on the input tensor's shape
            feature_map.set_shape(tf.TensorShape(gray.shape))
            
            return feature_map # Return tf.Tensor
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32)
    ])
    def _extract_edge_feature(self, image: tf.Tensor) -> tf.Tensor:
        """
        Examine edge coherence and artifacts typical in AI-generated images.

        Args:
            image: Input image (tf.Tensor, BGR format if color)

        Returns:
            2D feature map same size as input image with values between 0 and 1 (tf.Tensor)
        """
        with tf.name_scope('edge_feature'):
            image_tensor = image # Already a tf.Tensor

            if tf.shape(image_tensor)[-1] == 3:
                gray_weights = tf.constant([0.114, 0.587, 0.299], dtype=tf.float32)
                gray = tf.reduce_sum(image_tensor * gray_weights, axis=-1)
            else:
                gray = image_tensor
            
            gray = tf.cond(tf.reduce_max(gray) <= 1.0, 
                             lambda: gray * 255.0, 
                             lambda: gray)
            
            def py_edge_analysis(gray_np_uint8_tensor, epsilon_np_val):
                # Convert the EagerTensor to a NumPy array inside the py_function
                gray_np_uint8 = gray_np_uint8_tensor.numpy() 
                
                pyramid_levels = 3
                edge_pyramids = []
                current_img = gray_np_uint8.copy()

                for _ in range(pyramid_levels):
                    edges = cv2.Canny(current_img, 100, 200)
                    edge_pyramids.append(edges)
                    current_img = cv2.pyrDown(current_img)

                combined_edges = edge_pyramids[0].copy()
                for edges in edge_pyramids[1:]:
                    resized = cv2.resize(edges, (gray_np_uint8.shape[1], gray_np_uint8.shape[0]))
                    combined_edges = cv2.bitwise_or(combined_edges, resized)

                theta = np.linspace(0, 180, 36)
                global_radon = radon(combined_edges, theta=theta, circle=True)
                global_profile = np.sum(global_radon, axis=0)
                global_profile /= (np.sum(global_profile) + epsilon_np_val)

                glcm = graycomatrix(gray_np_uint8, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
                global_contrast = np.mean(graycoprops(glcm, 'contrast'))

                patch_size = 16
                h_np, w_np = gray_np_uint8.shape
                map_h, map_w = h_np // patch_size, w_np // patch_size
                edge_scores = np.zeros((map_h, map_w))

                # Extract patches once in NumPy, then process
                patches_combined_edges = tf.image.extract_patches(
                    images=tf.expand_dims(tf.expand_dims(tf.constant(combined_edges, dtype=tf.float32), 0), -1),
                    sizes=[1, patch_size, patch_size, 1],
                    strides=[1, patch_size, patch_size, 1],
                    rates=[1, 1, 1, 1],
                    padding='VALID'
                )
                patches_combined_edges = tf.squeeze(patches_combined_edges, axis=0)
                patches_combined_edges = tf.reshape(patches_combined_edges, [-1, patch_size, patch_size]) # Convert to numpy inside map_fn

                patches_gray_uint8 = tf.image.extract_patches(
                    images=tf.expand_dims(tf.expand_dims(tf.constant(gray_np_uint8, dtype=tf.float32), 0), -1),
                    sizes=[1, patch_size, patch_size, 1],
                    strides=[1, patch_size, patch_size, 1],
                    rates=[1, 1, 1, 1],
                    padding='VALID'
                )
                patches_gray_uint8 = tf.squeeze(patches_gray_uint8, axis=0)
                patches_gray_uint8 = tf.reshape(patches_gray_uint8, [-1, patch_size, patch_size]) # Convert to numpy inside map_fn

                def process_edge_patch(patch_tuple):
                    edge_patch_tf, gray_patch_tf = patch_tuple
                    edge_patch = edge_patch_tf.numpy().astype(np.uint8) # Convert to numpy
                    gray_patch = gray_patch_tf.numpy().astype(np.uint8) # Convert to numpy

                    score = 0.0
                    if np.sum(edge_patch) > patch_size:
                        patch_radon = radon(edge_patch, theta=theta, circle=True)
                        patch_profile = np.sum(patch_radon, axis=0)
                        patch_profile /= (np.sum(patch_profile) + epsilon_np_val)

                        entropy = -np.sum(patch_profile * np.log2(patch_profile + epsilon_np_val))
                        max_entropy = np.log2(len(theta))
                        entropy_score = 1 - (entropy / max_entropy) if max_entropy > 0 else 0

                        min_len_profile = min(len(patch_profile), len(global_profile))
                        correlation = np.corrcoef(patch_profile[:min_len_profile], global_profile[:min_len_profile])[0, 1]
                        correlation_score = 0.5 * (1 - abs(correlation))

                        score = entropy_score * 0.7 + correlation_score * 0.3

                        patch_glcm = graycomatrix(gray_patch, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
                        patch_contrast = np.mean(graycoprops(patch_glcm, 'contrast'))
                        contrast_ratio = patch_contrast / (global_contrast + epsilon_np_val)

                        if 0.7 < contrast_ratio < 1.3:
                            score *= 0.7
                    return np.float32(score)

                # Use tf.map_fn to process patches
                patch_scores_flat = tf.map_fn(
                    process_edge_patch,
                    (patches_combined_edges, patches_gray_uint8),
                    fn_output_signature=tf.float32,
                    parallel_iterations=10 # Adjust as needed
                )
                
                return patch_scores_flat.numpy().astype(np.float32).reshape((map_h, map_w))

            edge_scores_tensor_flat = tf.py_function(
                py_edge_analysis,
                [tf.cast(gray, tf.uint8), StructuralFeatureExtractor.EPSILON_NP],
                tf.float32
            )
            
            # Dynamically determine the output shape based on gray's shape
            h_tf, w_tf = tf.shape(gray)[0], tf.shape(gray)[1]
            map_h_tf = h_tf
            map_w_tf = w_tf
            
            # Set the shape of the py_function output
            edge_scores_tensor_flat.set_shape([map_h_tf * map_w_tf]) # Since py_function returns flat array

            # Reshape using the dynamically calculated map_h_tf and map_w_tf
            edge_scores_tensor = tf.reshape(edge_scores_tensor_flat, [map_h_tf, map_w_tf])
            
            feature_map = tf.image.resize(
                tf.expand_dims(tf.expand_dims(edge_scores_tensor, 0), -1),
                [h_tf, w_tf],
                method=tf.image.ResizeMethod.BILINEAR
            )
            
            feature_map = tf.squeeze(feature_map)
            feature_map = tf.clip_by_value(feature_map, 0, 1)
            
            return feature_map
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32)
    ])
    def _extract_symmetry_feature(self, image: tf.Tensor) -> tf.Tensor:
        """
        Measure unnatural symmetry often present in AI-generated images.
        
        Args:
            image: Input image (tf.Tensor, BGR format if color)
            
        Returns:
            2D feature map same size as input image with values between 0 and 1 (tf.Tensor)
        """
        with tf.name_scope('symmetry_feature'):
            image_tensor = image # Already a tf.Tensor

            if tf.shape(image_tensor)[-1] == 3:
                gray_weights = tf.constant([0.114, 0.587, 0.299], dtype=tf.float32)
                gray = tf.reduce_sum(image_tensor * gray_weights, axis=-1)
            else:
                gray = image_tensor
            
            gray = tf.cond(tf.reduce_max(gray) <= 1.0, 
                             lambda: gray * 255.0, 
                             lambda: gray)
            
            def py_symmetry_analysis(gray_np_uint8_tensor, epsilon_np_val): # Pass EPSILON_NP
                gray_np_uint8 = gray_np_uint8_tensor.numpy() 
                
                h_np, w_np = gray_np_uint8.shape
                
                scales = [1.0, 0.5]
                scale_weights = [0.7, 0.3]
                symmetry_scores = []
                axis_preferences = []
                
                for scale, weight in zip(scales, scale_weights):
                    if scale != 1.0:
                        scaled_h = int(h_np*scale)
                        scaled_w = int(w_np*scale)
                        scaled_gray = cv2.resize(gray_np_uint8, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
                    else:
                        scaled_gray = gray_np_uint8
                
                    half_w = scaled_gray.shape[1] // 2
                    left = scaled_gray[:, :half_w]
                    right = np.fliplr(scaled_gray[:, half_w:])
                    min_width = min(left.shape[1], right.shape[1])
                    h_ssim = ssim(left[:, :min_width], right[:, :min_width], data_range=255)
                    
                    half_h = scaled_gray.shape[0] // 2
                    top = scaled_gray[:half_h, :]
                    bottom = np.flipud(scaled_gray[half_h:, :])
                    min_height = min(top.shape[0], bottom.shape[0])
                    v_ssim = ssim(top[:min_height, :], bottom[:min_height, :], data_range=255)
                
                    axis_score = max(h_ssim, v_ssim)
                    symmetry_scores.append(axis_score * weight)
                    axis_preferences.append(h_ssim > v_ssim)
                
                combined_score = np.sum(symmetry_scores) / np.sum(scale_weights)
                primary_horizontal = np.mean(axis_preferences) > 0.5
                
                combined_score_np = np.float32(combined_score)
                
                edges = cv2.Canny(gray_np_uint8, 100, 200)
                if primary_horizontal:
                    half_w = w_np // 2
                    left_edges = edges[:, :half_w]
                    right_edges = np.fliplr(edges[:, half_w:])
                    min_width = min(left_edges.shape[1], right_edges.shape[1])
                    edge_match = np.mean(left_edges[:, :min_width] == right_edges[:, :min_width])
                else:
                    half_h = h_np // 2
                    top_edges = edges[:half_h, :]
                    bottom_edges = np.flipud(edges[half_h:, :])
                    min_height = min(top_edges.shape[0], bottom_edges.shape[0])
                    edge_match = np.mean(top_edges[:min_height, :] == bottom_edges[:min_height, :])
                
                if edge_match > 0.9:
                    combined_score_np = min(combined_score_np * 1.2, 1.0)

                return combined_score_np, primary_horizontal

            combined_score_tf, primary_horizontal_tf = tf.py_function(
                py_symmetry_analysis,
                [tf.cast(gray, tf.uint8), StructuralFeatureExtractor.EPSILON_NP], # Pass EPSILON_NP
                [tf.float32, tf.bool]
            )
            combined_score_tf.set_shape([])
            primary_horizontal_tf.set_shape([])
        
            h_tensor = tf.shape(gray)[0]
            w_tensor = tf.shape(gray)[1]

            feature_map = tf.cond(
                primary_horizontal_tf,
                lambda: StructuralFeatureExtractor._create_horizontal_gradient_map(h_tensor, w_tensor, combined_score_tf),
                lambda: StructuralFeatureExtractor._create_vertical_gradient_map(h_tensor, w_tensor, combined_score_tf)
            )
            
            feature_map = tf.clip_by_value(feature_map, 0, 1)
            
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