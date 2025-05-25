import tensorflow as tf
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from skimage.feature import graycoprops, graycomatrix
from PIL import Image
import imagehash

class TextureFeatureExtractor:
    # Define EPSILON as a class-level tf.constant for TensorFlow operations
    EPSILON_TF = tf.constant(1e-8, dtype=tf.float32)
    # Define a separate NumPy version for use within tf.py_function calls
    EPSILON_NP = 1e-8

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32) # Allow dynamic H, W but fixed C
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
            
            is_float_input = tf.constant(image_tf.dtype.is_floating)

            if tf.rank(image_tf) == 3 and tf.shape(image_tf)[-1] == 3:
                gray_raw = tf.cond(is_float_input,
                                   lambda: tf.image.rgb_to_grayscale(tf.cast(image_tf * 255, tf.uint8)),
                                   lambda: tf.image.rgb_to_grayscale(tf.cast(image_tf, tf.uint8)))
                gray = tf.squeeze(gray_raw)
            else:
                gray = tf.identity(image_tf)
                gray = tf.cond(is_float_input,
                               lambda: tf.cast(gray * 255, tf.uint8),
                               lambda: tf.cast(gray, tf.uint8))
            
            gray = tf.cast(gray, tf.float32)
            
            # Create Gaussian kernel
            kernel_size = 5
            sigma_val_init = 0.0

            def gaussian_kernel(size, sigma_val):
                x = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
                x_grid, y_grid = tf.meshgrid(x, x)
                kernel = tf.exp(-(x_grid**2 + y_grid**2) / (2.0 * sigma_val**2 + TextureFeatureExtractor.EPSILON_TF))
                return kernel / (tf.reduce_sum(kernel) + TextureFeatureExtractor.EPSILON_TF)
                
            sigma_tf = tf.constant(sigma_val_init, dtype=tf.float32)
            
            sigma = tf.cond(sigma_tf <= 0, 
                            lambda: tf.constant(0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8, dtype=tf.float32), 
                            lambda: sigma_tf)
                
            kernel = gaussian_kernel(kernel_size, sigma)
            kernel = kernel[:, :, tf.newaxis, tf.newaxis]
            
            gray_4d = tf.expand_dims(tf.expand_dims(gray, 0), -1)
            
            blurred = tf.nn.conv2d(gray_4d, kernel, strides=[1, 1, 1, 1], padding='SAME')
            blurred = tf.squeeze(blurred)
            
            noise = gray - blurred
            noise_std = tf.math.reduce_std(noise) + TextureFeatureExtractor.EPSILON_TF
            noise_normalized = noise / noise_std
            
            patch_size = 16
            h, w = tf.shape(gray)[0], tf.shape(gray)[1]
            
            noise_4d = tf.expand_dims(tf.expand_dims(noise, 0), -1)
            noise_normalized_4d = tf.expand_dims(tf.expand_dims(noise_normalized, 0), -1)

            patches = tf.image.extract_patches(
                images=noise_4d,
                sizes=[1, patch_size, patch_size, 1],
                strides=[1, patch_size, patch_size, 1],
                rates=[1, 1, 1, 1],
                padding='VALID'
            )
            patches = tf.reshape(patches, [-1, patch_size, patch_size])

            patches_norm = tf.image.extract_patches(
                images=noise_normalized_4d,
                sizes=[1, patch_size, patch_size, 1],
                strides=[1, patch_size, patch_size, 1],
                rates=[1, 1, 1, 1],
                padding='VALID'
            )
            patches_norm = tf.reshape(patches_norm, [-1, patch_size, patch_size])

            gray_4d_patches = tf.image.extract_patches(
                images=tf.expand_dims(tf.expand_dims(gray, 0), -1),
                sizes=[1, patch_size, patch_size, 1],
                strides=[1, patch_size, patch_size, 1],
                rates=[1, 1, 1, 1],
                padding='VALID'
            )
            gray_patches = tf.reshape(gray_4d_patches, [-1, patch_size, patch_size])

            # Calculate global statistics (once for the whole image)
            global_var = tf.math.reduce_variance(noise)
            flat_noise = tf.reshape(noise, [-1])

            def py_median(arr_tf):
                arr_np = arr_tf.numpy()
                return np.median(arr_np).astype(np.float32)
            
            def py_mad(arr_tf, median_val_tf):
                arr_np = arr_tf.numpy()
                median_val_np = median_val_tf.numpy()
                return np.median(np.abs(arr_np - median_val_np)).astype(np.float32)

            global_median_noise = tf.py_function(py_median, [flat_noise], tf.float32)
            global_median_noise.set_shape([])

            global_mad = tf.py_function(py_mad, [flat_noise, global_median_noise], tf.float32)
            global_mad.set_shape([])

            # Define a function to process a single patch
            def process_noise_patch(patch_tuple):
                patch, patch_norm, gray_patch = patch_tuple
                
                local_std = tf.math.reduce_std(patch)
                flat_patch = tf.reshape(patch, [-1])
                
                patch_median = tf.py_function(py_median, [flat_patch], tf.float32)
                patch_median.set_shape([])

                local_mad = tf.py_function(py_mad, [flat_patch, patch_median], tf.float32)
                local_mad.set_shape([])

                local_var = tf.math.reduce_variance(patch)
                
                abs_patch_norm = tf.abs(patch_norm)
                entropy = -tf.reduce_sum(abs_patch_norm * tf.math.log(abs_patch_norm + TextureFeatureExtractor.EPSILON_TF) / tf.math.log(2.0 + TextureFeatureExtractor.EPSILON_TF))
                
                flat_patch_norm = tf.reshape(patch_norm, [-1])
                sorted_patch = tf.sort(flat_patch_norm)
                size = tf.size(sorted_patch)
                q1_idx = tf.cast(tf.round(0.25 * tf.cast(size, tf.float32)), tf.int32)
                q3_idx = tf.cast(tf.round(0.75 * tf.cast(size, tf.float32)), tf.int32)
                
                q1_idx = tf.clip_by_value(q1_idx, 0, size - 1)
                q3_idx = tf.clip_by_value(q3_idx, 0, size - 1)
                
                q1 = sorted_patch[q1_idx]
                q3 = sorted_patch[q3_idx]
                
                global_mad_factor = tf.cond(global_mad > 0, lambda: 2 * global_mad, lambda: TextureFeatureExtractor.EPSILON_TF)
                
                robust_kurtosis = tf.cond(
                    local_mad > 0, 
                    lambda: (q3 - q1) / (global_mad_factor), 
                    lambda: tf.constant(1.0, dtype=tf.float32)
                )
                
                var_ratio = tf.cond(global_var > 0, 
                                    lambda: local_var / (global_var + TextureFeatureExtractor.EPSILON_TF),
                                    lambda: tf.constant(0.0, dtype=tf.float32))
                var_diff = tf.abs(1 - var_ratio)
                
                score = tf.constant(0.0, dtype=tf.float32)

                score = tf.where(local_std < 0.005, 0.8, score)
                score = tf.where(tf.logical_and(local_std >= 0.005, entropy < 0.1), 0.7, score)
                score = tf.where(
                    tf.logical_and(
                        tf.logical_and(local_std >= 0.005, entropy >= 0.1), 
                        tf.abs(robust_kurtosis - (q3 - q1)/global_mad_factor) > 1.5
                    ), 
                    0.6, 
                    score
                )
                score = tf.where(
                    tf.logical_and(
                        tf.logical_and(
                            tf.logical_and(local_std >= 0.005, entropy >= 0.1), 
                            tf.abs(robust_kurtosis - (q3 - q1)/global_mad_factor) <= 1.5
                        ),
                        var_diff > 0.7
                    ), 
                    0.5, 
                    score
                )
                
                patch_mean = tf.reduce_mean(gray_patch)
                score = tf.where(
                    tf.logical_or(patch_mean > 240, patch_mean < 30),
                    score * 0.3,
                    score
                )
                
                return tf.minimum(score, 1.0)
            
            # Use tf.map_fn to process all patches
            patch_scores_tensor = tf.map_fn(
                process_noise_patch,
                (patches, patches_norm, gray_patches),
                fn_output_signature=tf.float32,
                parallel_iterations=10 # Adjust for performance
            )
            
            rows = h // patch_size
            cols = w // patch_size
            patch_scores_grid = tf.reshape(patch_scores_tensor, [rows, cols])

            feature_map = tf.image.resize(
                tf.expand_dims(tf.expand_dims(patch_scores_grid, 0), -1),
                [h, w],
                method=tf.image.ResizeMethod.BILINEAR
            )
            feature_map = tf.squeeze(feature_map)
        
            global_variance_magnitude = tf.math.log1p(global_var * 1000)
            feature_map = tf.cond(
                global_variance_magnitude < -5,
                lambda: tf.clip_by_value(feature_map * 1.2, 0, 1),
                lambda: feature_map
            )
            feature_map = tf.cond(
                global_variance_magnitude > 2,
                lambda: tf.clip_by_value(feature_map * 0.7, 0, 1),
                lambda: feature_map
            )
            
            return feature_map
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32)
    ])
    def _extract_texture_feature(self, image: tf.Tensor) -> tf.Tensor:
        """
        Analyze texture consistency and identify AI artifacts in textures.

        Args:
            image: Input image (tf.Tensor, BGR format if color)

        Returns:
            2D feature map same size as input image with values between 0 and 1 (tf.Tensor)
        """
        with tf.name_scope('texture_feature'):
            if tf.rank(image) == 3 and tf.shape(image)[-1] == 3:
                is_float_input = tf.constant(image.dtype.is_floating)
                gray_raw = tf.cond(is_float_input,
                                   lambda: tf.image.rgb_to_grayscale(tf.cast(image * 255, tf.uint8)),
                                   lambda: tf.image.rgb_to_grayscale(tf.cast(image, tf.uint8)))
                gray = tf.squeeze(gray_raw)
            else:
                gray = tf.identity(tf.cast(image, tf.uint8))
                is_float_input = tf.constant(image.dtype.is_floating)
                gray = tf.cond(is_float_input,
                               lambda: tf.cast(gray * 255, tf.uint8),
                               lambda: gray)

            h, w = tf.shape(gray)[0], tf.shape(gray)[1]
            patch_size = 32
            
            radius = 2
            n_points = 8 * radius

            def py_lbp(image_tf_uint8, n_points_tf, radius_tf):
                image_np_uint8 = image_tf_uint8.numpy()
                n_points_np = n_points_tf.numpy()
                radius_np = radius_tf.numpy()
                
                lbp_result = local_binary_pattern(image_np_uint8, n_points_np, radius_np, method='uniform')
                return lbp_result.astype(np.float32)

            lbp = tf.py_function(py_lbp, [gray, tf.constant(n_points, dtype=tf.int32), tf.constant(radius, dtype=tf.int32)], tf.float32)
            lbp.set_shape(gray.shape)
            
            sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
            sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tf.float32)
            
            sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
            sobel_y = tf.reshape(sobel_y, [3, 3, 1, 1])
            
            gray_float = tf.cast(gray, tf.float32)
            gray_4d = tf.expand_dims(tf.expand_dims(gray_float, 0), -1)
            
            edge_x = tf.nn.conv2d(gray_4d, sobel_x, strides=[1, 1, 1, 1], padding='SAME')
            edge_y = tf.nn.conv2d(gray_4d, sobel_y, strides=[1, 1, 1, 1], padding='SAME')
            
            edge_magnitude = tf.sqrt(tf.square(edge_x) + tf.square(edge_y))
            edge_magnitude = tf.squeeze(edge_magnitude)
            
            edges = tf.cast(edge_magnitude > 50, tf.float32)
            
            kernel_size_box = patch_size
            box_kernel = tf.ones((kernel_size_box, kernel_size_box), dtype=tf.float32) / tf.cast((kernel_size_box * kernel_size_box), tf.float32)
            box_kernel = tf.reshape(box_kernel, [kernel_size_box, kernel_size_box, 1, 1])
            
            edges_4d = tf.expand_dims(tf.expand_dims(edges, 0), -1)
            edge_density_raw = tf.nn.conv2d(edges_4d, box_kernel, strides=[1, 1, 1, 1], padding='SAME')
            edge_density = tf.squeeze(edge_density_raw)
            
            def py_glcm_props(image_tf_uint8, epsilon_np_val):
                image_np_uint8 = image_tf_uint8.numpy()
                glcm = graycomatrix(image_np_uint8, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
                contrast = graycoprops(glcm, 'contrast').mean()
                energy = graycoprops(glcm, 'energy').mean()
                return np.array([contrast, energy], dtype=np.float32)

            global_glcm_stats = tf.py_function(py_glcm_props, [gray, TextureFeatureExtractor.EPSILON_NP], tf.float32)
            global_glcm_stats.set_shape([2])
            global_contrast = global_glcm_stats[0]
            global_energy = global_glcm_stats[1]

            rows_start = tf.range(0, h - patch_size + 1, patch_size // 2)
            cols_start = tf.range(0, w - patch_size + 1, patch_size // 2)
            
            # Create all patch coordinates
            all_patch_coords = tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=False)
            k = 0
            for i in rows_start:
                for j in cols_start:
                    all_patch_coords = all_patch_coords.write(k, tf.stack([i, j]))
                    k += 1
            all_patch_coords_stacked = all_patch_coords.stack()

            # Define a function to process a single patch
            def process_texture_patch(coords):
                pi, pj = coords[0], coords[1]
                
                patch_gray = gray[pi:pi+patch_size, pj:pj+patch_size]
                patch_tf = tf.cast(patch_gray, tf.float32)

                patch_std = tf.math.reduce_std(patch_tf)
                
                lbp_patch = lbp[pi:pi+patch_size, pj:pj+patch_size]
                lbp_patch_flat = tf.reshape(lbp_patch, [-1])
                
                lbp_hist = tf.histogram_fixed_width(lbp_patch_flat, [0, n_points + 2], nbins=n_points + 2)
                lbp_hist = tf.cast(lbp_hist, tf.float32) / (tf.cast(patch_size**2, tf.float32) + TextureFeatureExtractor.EPSILON_TF)
                
                lbp_entropy = -tf.reduce_sum(lbp_hist * tf.math.log(lbp_hist + TextureFeatureExtractor.EPSILON_TF) / tf.math.log(2.0 + TextureFeatureExtractor.EPSILON_TF))
                
                local_glcm_stats = tf.py_function(py_glcm_props, [patch_gray, TextureFeatureExtractor.EPSILON_NP], tf.float32)
                local_glcm_stats.set_shape([2])
                patch_contrast = local_glcm_stats[0]
                patch_energy = local_glcm_stats[1]
                
                score = tf.constant(0.0, dtype=tf.float32)
                score = tf.where(patch_std < 5, score + 0.4, score)
                score = tf.where(lbp_entropy < 2.0, score + 0.3, score)
                score = tf.where(tf.abs(patch_contrast - global_contrast) > global_contrast * 0.5, score + 0.2, score)
                score = tf.where(tf.abs(patch_energy - global_energy) > global_energy * 0.5, score + 0.1, score)
                
                local_edge_density = tf.reduce_mean(edge_density[pi:pi+patch_size, pj:pj+patch_size])
                score = tf.where(local_edge_density < 0.05, score * 0.5, score)
                score = tf.where(local_edge_density > 0.2, score * 1.2, score)
                
                return tf.minimum(score, 1.0)
            
            # Use tf.map_fn to process all patches
            scores_gathered = tf.map_fn(
                process_texture_patch,
                all_patch_coords_stacked,
                fn_output_signature=tf.float32,
                parallel_iterations=10 # Adjust for performance
            )
            
            final_feature_map = tf.zeros((h, w), dtype=tf.float32)

            kernel_size_gaussian = patch_size
            x = tf.range(0, kernel_size_gaussian, dtype=tf.float32) - tf.cast(kernel_size_gaussian, tf.float32) / 2
            x_grid, y_grid = tf.meshgrid(x, x)
            gaussian_kernel_patch = tf.exp(-(x_grid**2 + y_grid**2) / (2.0 * (tf.cast(patch_size, tf.float32) / 3)**2 + TextureFeatureExtractor.EPSILON_TF))
            gaussian_kernel_patch = gaussian_kernel_patch / (tf.reduce_max(gaussian_kernel_patch) + TextureFeatureExtractor.EPSILON_TF)

            # Use a tf.TensorArray to accumulate updates more efficiently than scatter_nd_add in a loop
            # This can still be a source of retracing if map_fn output shape changes, but less likely with fixed patch_size
            # and constant input images.
            # A more advanced approach would be to use tf.nn.conv2d with a stride for aggregation.
            # However, for overlapping patches with max pooling, the current approach is conceptually sound.

            # Re-implementing the aggregation using tf.TensorArray to collect patches and then batching
            # This is complex to do purely in TF graph mode due to variable number of patches and dynamic updates.
            # For simplicity and to avoid retracing caused by Python loops within @tf.function,
            # it's best if the number of patches is fixed. If input images have varying sizes,
            # this part will still cause re-tracing. Assuming fixed input image dimensions.

            # Alternative for aggregation: create an output grid and fill based on coordinates.
            # The previous scatter_nd_add in a loop is already graph-compatible, but tf.maximum needs careful handling.
            
            # Use a loop over known patch coordinates which is now graph-compatible.
            for k_iter in tf.range(tf.shape(scores_gathered)[0]):
                score = scores_gathered[k_iter]
                pi, pj = all_patch_coords_stacked[k_iter][0], all_patch_coords_stacked[k_iter][1]

                weighted_score_patch = score * gaussian_kernel_patch

                # Define the slice where this patch's update goes
                patch_slice = tf.slice(final_feature_map, [pi, pj], [patch_size, patch_size])
                
                # Apply the update using tf.maximum
                updated_patch_slice = tf.maximum(patch_slice, weighted_score_patch)
                
                # Create indices for tensor_scatter_nd_update (for the full map update)
                # This is still relatively expensive and can cause retracing if shapes are not perfectly static.
                # The ideal solution would be a single vectorized operation.
                
                # For fixed patch_size and fixed image dimensions, this should work.
                # If image dimensions vary, this will still be a challenge.
                
                # An alternative for aggregation, if you're willing to approximate, is to resize the score_grid
                # and then apply a final blur.
                
                # For now, sticking with the current logic but emphasizing fixed dimensions to avoid re-tracing here.
                # If image sizes are truly dynamic, this section is a potential point of future optimization.
                final_feature_map = tf.tensor_scatter_nd_update(
                    tensor=final_feature_map,
                    indices=tf.stack(tf.meshgrid(tf.range(pi, pi + patch_size), tf.range(pj, pj + patch_size), indexing='ij'), axis=-1),
                    updates=tf.reshape(updated_patch_slice, [-1])
                )


            kernel_size_smooth = 15
            sigma_smooth = 3.0
            
            x_smooth = tf.range(-(kernel_size_smooth//2), kernel_size_smooth//2 + 1, dtype=tf.float32)
            x_grid_smooth, y_grid_smooth = tf.meshgrid(x_smooth, x_smooth)
            gaussian_kernel_smooth = tf.exp(-(x_grid_smooth**2 + y_grid_smooth**2) / (2.0 * sigma_smooth**2 + TextureFeatureExtractor.EPSILON_TF))
            gaussian_kernel_smooth = gaussian_kernel_smooth / (tf.reduce_sum(gaussian_kernel_smooth) + TextureFeatureExtractor.EPSILON_TF)
            gaussian_kernel_smooth = tf.reshape(gaussian_kernel_smooth, [kernel_size_smooth, kernel_size_smooth, 1, 1])
            
            feature_map_4d = tf.expand_dims(tf.expand_dims(final_feature_map, 0), -1)
            feature_map_smoothed = tf.nn.conv2d(feature_map_4d, gaussian_kernel_smooth, strides=[1, 1, 1, 1], padding='SAME')
            feature_map_smoothed = tf.squeeze(feature_map_smoothed)
            
            feature_min = tf.reduce_min(feature_map_smoothed)
            feature_range = tf.reduce_max(feature_map_smoothed) - feature_min + TextureFeatureExtractor.EPSILON_TF
            feature_map_normalized = (feature_map_smoothed - feature_min) / feature_range
            
            feature_map_final = tf.clip_by_value(feature_map_normalized, 0, 1)
            
            return feature_map_final
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32) # Assume color input for this feature
    ])
    def _extract_color_feature(self, image: tf.Tensor) -> tf.Tensor:
        """
        Detect color distribution anomalies common in AI-generated images.

        Args:
            image: Input image (tf.Tensor, BGR format if color)

        Returns:
            2D feature map same size as input image with values between 0 and 1 (tf.Tensor)
        """
        with tf.name_scope('color_feature'):
            # Handle non-color images by returning a zero map
            # This check is now redundant if input_signature enforces 3 channels for this function
            if tf.shape(image)[2] != 3:
                h, w = tf.shape(image)[0], tf.shape(image)[1]
                return tf.zeros((h, w), dtype=tf.float32)
            
            is_float_input = tf.constant(image.dtype.is_floating)

            img_uint8 = tf.cond(is_float_input,
                                lambda: tf.cast(image * 255, tf.uint8),
                                lambda: tf.cast(image, tf.uint8))
            
            def py_bgr_to_hsv(img_tf_uint8):
                img_np_uint8 = img_tf_uint8.numpy()
                return cv2.cvtColor(img_np_uint8, cv2.COLOR_BGR2HSV).astype(np.float32)

            hsv_img = tf.py_function(py_bgr_to_hsv, [img_uint8], tf.float32)
            # Set shape explicitly for hsv_img
            hsv_img.set_shape(tf.TensorShape([None, None, 3]))

            h, w = tf.shape(img_uint8)[0], tf.shape(img_uint8)[1]
            feature_map = tf.zeros((h, w), dtype=tf.float32)
            patch_size = 16
            
            hue = hsv_img[:, :, 0]
            hue_rad = hue * (np.pi / 180.0)

            mean_sin = tf.reduce_mean(tf.sin(hue_rad))
            mean_cos = tf.reduce_mean(tf.cos(hue_rad))
            global_hue_var = 1.0 - tf.sqrt(mean_sin**2 + mean_cos**2 + TextureFeatureExtractor.EPSILON_TF)
            limited_palette = global_hue_var < 0.3
            
            def py_bgr_to_gray(img_tf_uint8):
                img_np_uint8 = img_tf_uint8.numpy()
                return cv2.cvtColor(img_np_uint8, cv2.COLOR_BGR2GRAY).astype(np.float32)
            
            gray = tf.py_function(py_bgr_to_gray, [img_uint8], tf.float32)
            # Set shape explicitly for gray
            gray.set_shape(tf.TensorShape([None, None]))

            laplacian_kernel = tf.constant([
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]
            ], dtype=tf.float32)
            laplacian_kernel = tf.reshape(laplacian_kernel, [3, 3, 1, 1])
            
            gray_4d = tf.expand_dims(tf.expand_dims(gray, 0), -1)
            laplacian = tf.nn.conv2d(gray_4d, laplacian_kernel, strides=[1, 1, 1, 1], padding='SAME')
            laplacian = tf.squeeze(laplacian)
            
            global_lap_var = tf.math.reduce_variance(laplacian)
            
            # Extract patches for both HSV and Laplacian
            hsv_patches_extracted = tf.image.extract_patches(
                images=tf.expand_dims(hsv_img, 0),
                sizes=[1, patch_size, patch_size, 1],
                strides=[1, patch_size, patch_size, 1],
                rates=[1, 1, 1, 1],
                padding='VALID'
            )
            hsv_patches_reshaped = tf.reshape(hsv_patches_extracted, [-1, patch_size, patch_size, 3])

            laplacian_patches_extracted = tf.image.extract_patches(
                images=tf.expand_dims(tf.expand_dims(laplacian, 0), -1),
                sizes=[1, patch_size, patch_size, 1],
                strides=[1, patch_size, patch_size, 1],
                rates=[1, 1, 1, 1],
                padding='VALID'
            )
            laplacian_patches_reshaped = tf.reshape(laplacian_patches_extracted, [-1, patch_size, patch_size])
            
            # Get patch coordinates (top-left) for mapping back to the full image
            # Calculate grid dimensions first
            num_patches_h = h // patch_size
            num_patches_w = w // patch_size
            
            # Generate all patch top-left coordinates
            patch_indices_flat = tf.stack(tf.meshgrid(
                tf.range(0, h - patch_size + 1, patch_size),
                tf.range(0, w - patch_size + 1, patch_size),
                indexing='ij'
            ), axis=-1)
            patch_indices_flat = tf.reshape(patch_indices_flat, [-1, 2])


            # Define a function to process a single patch
            def process_color_patch(patch_tuple):
                hsv_patch, laplacian_patch, patch_coords = patch_tuple
                
                hue_patch = hsv_patch[:, :, 0]
                sat_patch = hsv_patch[:, :, 1]
                
                hue_rad_patch = hue_patch * (np.pi / 180.0)
                mean_sin_p = tf.reduce_mean(tf.sin(hue_rad_patch))
                mean_cos_p = tf.reduce_mean(tf.cos(hue_rad_patch))
                local_hue_var = 1.0 - tf.sqrt(mean_sin_p**2 + mean_cos_p**2 + TextureFeatureExtractor.EPSILON_TF)
                
                mean_sat = tf.reduce_mean(sat_patch)
                
                lap_ratio = tf.math.reduce_variance(laplacian_patch) / (global_lap_var + TextureFeatureExtractor.EPSILON_TF)
                smooth_gradient = tf.logical_and(lap_ratio > 0.1, lap_ratio < 0.5)
                
                unnatural = tf.constant(False)
                if_limited_palette = tf.logical_and(local_hue_var < 0.05, tf.logical_not(smooth_gradient))
                
                var_ratio_cond = tf.cond(global_hue_var > 0, 
                                         lambda: local_hue_var / (global_hue_var + TextureFeatureExtractor.EPSILON_TF),
                                         lambda: tf.constant(0.0, dtype=tf.float32))
                if_not_limited_palette = tf.logical_and(var_ratio_cond < 0.3, tf.logical_not(smooth_gradient))
                
                unnatural = tf.cond(limited_palette, lambda: if_limited_palette, lambda: if_not_limited_palette)
                
                should_update = tf.logical_and(unnatural, mean_sat > 30)
                
                score_val = tf.cond(should_update, lambda: 1.0, lambda: 0.0)
                return score_val

            # Use tf.map_fn to process all patches
            patch_scores_tensor = tf.map_fn(
                process_color_patch,
                (hsv_patches_reshaped, laplacian_patches_reshaped, patch_indices_flat),
                fn_output_signature=tf.float32,
                parallel_iterations=10 # Adjust for performance
            )
            
            # Reshape scores into a grid corresponding to patches
            scores_grid = tf.reshape(patch_scores_tensor, [num_patches_h, num_patches_w])

            feature_map_patches = tf.image.resize(
                tf.expand_dims(tf.expand_dims(scores_grid, 0), -1),
                [h, w],
                method=tf.image.ResizeMethod.BILINEAR
            )
            feature_map = tf.squeeze(feature_map_patches)
            
            return feature_map
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32) # Allow dynamic H, W, C
    ])
    def _extract_hash_feature(self, image: tf.Tensor) -> tf.Tensor:
        """
        Compare perceptual hash similarity to known AI patterns.

        Args:
            image: Input image (tf.Tensor, BGR format if color)

        Returns:
            2D feature map same size as input image with values between 0 and 1 (tf.Tensor)
        """
        h, w = tf.shape(image)[0], tf.shape(image)[1]
        
        def py_hash_analysis(image_tf, epsilon_np_val):
            if not hasattr(Image, 'ANTIALIAS'):
                Image.ANTIALIAS = Image.LANCZOS
            
            image_np = image_tf.numpy()
            try:
                if image_np.dtype == np.float32 or image_np.dtype == np.float64:
                    image_uint8 = (image_np * 255).astype(np.uint8)
                else:
                    image_uint8 = image_np.astype(np.uint8)
                
                if len(image_uint8.shape) == 3 and image_uint8.shape[2] == 3:
                    image_pil = Image.fromarray(cv2.cvtColor(image_uint8, cv2.COLOR_BGR2RGB))
                else:
                    image_pil = Image.fromarray(image_uint8)
                
                phash = imagehash.phash(image_pil, hash_size=8)
                dhash = imagehash.dhash(image_pil, hash_size=8)
                
                # For simplicity, returning a scalar based on a simple combination.
                # In a real scenario, you'd compare these hashes to a database.
                # Example: a score based on Hamming distance to some 'ideal' hash.
                # For a feature map, this global score is broadcast.
                
                # Dummy score: average of phash and dhash (as integers, then normalized)
                # This is just an example to demonstrate returning a scalar.
                dummy_score = (phash.hash.sum() + dhash.hash.sum()) / (2 * 64) # Max 64 for each hash.sum()
                return np.array([dummy_score], dtype=np.float32)
            except Exception as e:
                tf.print(f"Error in py_hash_analysis: {e}")
                return np.array([0.0], dtype=np.float32)

        hash_score_tensor = tf.py_function(
            func=py_hash_analysis,
            inp=[image, TextureFeatureExtractor.EPSILON_NP],
            Tout=tf.float32
        )
        hash_score_tensor.set_shape([1])

        feature_map = tf.fill(tf.stack([h, w]), hash_score_tensor[0])
        
        return feature_map