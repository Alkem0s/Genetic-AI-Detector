import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from skimage.feature import graycoprops

from utils import get_edge_detection, cached_glcm, image_hash_to_array

class TextureFeaturesExtractor:
    @staticmethod
    def detect_noise_patterns(image, patch_size=16):
        """
        Detect unnatural noise patterns or lack of natural noise
        Natural images have characteristic noise patterns that AI often fails to replicate
        """
        if len(image.shape) > 2 and image.shape[2] > 1:
            # For color images, analyze each channel
            # Vectorized channel processing
            channels = np.dsplit(image, image.shape[2])
            blurred_channels = [cv2.GaussianBlur(ch.squeeze(), (5, 5), 0) for ch in channels]
            noise_maps = [ch.squeeze() - blurred for ch, blurred in zip(channels, blurred_channels)]
            noise = np.mean(noise_maps, axis=0)
        else:
            # For grayscale
            blurred = cv2.GaussianBlur(image, (5, 5), 0)
            noise = image - blurred
        
        # Convert to standard scale for more consistent analysis
        noise_std = np.std(noise)
        noise_normalized = noise / noise_std if noise_std > 0 else noise
        
        h, w = noise.shape
        feature_map = np.zeros((h, w))
        
        # Calculate global noise statistics
        global_mean = np.mean(noise)
        global_std = np.std(noise)
        global_mad = np.median(np.abs(noise - np.median(noise)))  # Median Absolute Deviation
        
        # Calculate robust global kurtosis estimator (less sensitive to outliers)
        sorted_noise = np.sort(noise_normalized.flatten())
        n = len(sorted_noise)
        q1_pos = int(n * 0.25)
        q3_pos = int(n * 0.75)
        q1 = sorted_noise[q1_pos]
        q3 = sorted_noise[q3_pos]
        robust_kurtosis = (q3 - q1) / (2 * global_mad) if global_mad > 0 else 1
        
        # Calculate noise variance
        global_var = np.var(noise)
        
        # Prepare image for processing
        gray_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) if len(image.shape) > 2 else (image * 255).astype(np.uint8)
        
        # Vectorized patch creation using strided arrays
        # Pre-allocate arrays for patch scores
        rows = np.arange(0, h - patch_size, patch_size)
        cols = np.arange(0, w - patch_size, patch_size)
        num_rows = len(rows)
        num_cols = len(cols)
        
        noise_pattern_map = np.zeros((num_rows, num_cols))
        all_stds = np.zeros((num_rows, num_cols))
        all_entropies = np.zeros((num_rows, num_cols))
        all_mads = np.zeros((num_rows, num_cols))
        all_kurtosis = np.ones((num_rows, num_cols))
        all_vars = np.zeros((num_rows, num_cols))
        var_diffs = np.zeros((num_rows, num_cols))
        gray_means = np.zeros((num_rows, num_cols))
        
        # Optimized vectorized patch analysis
        for idx_i, i in enumerate(rows):
            for idx_j, j in enumerate(cols):
                patch = noise[i:i+patch_size, j:j+patch_size]
                patch_norm = noise_normalized[i:i+patch_size, j:j+patch_size]
                
                # Calculate patch statistics
                all_stds[idx_i, idx_j] = np.std(patch)
                all_entropies[idx_i, idx_j] = -np.sum(np.abs(patch) * np.log2(np.abs(patch) + 1e-10))
                
                # Calculate median absolute deviation
                local_median = np.median(patch)
                local_mad = np.median(np.abs(patch - local_median))
                all_mads[idx_i, idx_j] = local_mad
                
                # Calculate robust kurtosis
                sorted_patch = np.sort(patch_norm.flatten())
                n_local = len(sorted_patch)
                if n_local > 4:
                    q1_pos_local = int(n_local * 0.25)
                    q3_pos_local = int(n_local * 0.75)
                    q1_local = sorted_patch[q1_pos_local]
                    q3_local = sorted_patch[q3_pos_local]
                    all_kurtosis[idx_i, idx_j] = (q3_local - q1_local) / (2 * local_mad) if local_mad > 0 else 1
                
                # Calculate variance and relative variance
                all_vars[idx_i, idx_j] = np.var(patch)
                if global_var > 0:
                    var_ratio = all_vars[idx_i, idx_j] / global_var
                    var_diffs[idx_i, idx_j] = abs(1 - var_ratio)
                else:
                    var_diffs[idx_i, idx_j] = 0 if all_vars[idx_i, idx_j] == 0 else 1
                
                # Gray mean for content analysis
                gray_patch = gray_image[i:i+patch_size, j:j+patch_size]
                gray_means[idx_i, idx_j] = np.mean(gray_patch)
        
        # Vectorized scoring based on multiple criteria
        unnatural_scores = np.zeros_like(all_stds)
        
        # Apply scoring masks vectorized
        unnatural_mask = (all_stds < 0.005)
        unnatural_scores[unnatural_mask] = 0.8
        
        entropy_mask = (all_entropies < 0.1)
        unnatural_scores[entropy_mask & ~unnatural_mask] = 0.7
        
        kurtosis_mask = (np.abs(all_kurtosis - robust_kurtosis) > 1.5)
        unnatural_scores[kurtosis_mask & ~(unnatural_mask | entropy_mask)] = 0.6
        
        var_mask = (var_diffs > 0.7)
        unnatural_scores[var_mask & ~(unnatural_mask | entropy_mask | kurtosis_mask)] = 0.5
        
        # Account for image content - don't penalize naturally smooth areas
        bright_mask = (gray_means > 240)
        dark_mask = (gray_means < 30)
        unnatural_scores[bright_mask | dark_mask] *= 0.3
        
        # Fill maps
        noise_pattern_map = unnatural_scores.copy()
        
        # Create full feature map from patch scores
        for idx_i, i in enumerate(rows):
            for idx_j, j in enumerate(cols):
                score = unnatural_scores[idx_i, idx_j]
                if score > 0:
                    feature_map[i:i+patch_size, j:j+patch_size] = score
        
        # Adjust overall score based on global variance assessment
        global_variance_magnitude = np.log1p(global_var * 1000)  # Logarithmic scaling
        if global_variance_magnitude < -5:  # Unnaturally low variance
            global_adjustment = 1.2  # Increase scores
        elif global_variance_magnitude > 2:  # High variance (likely natural image noise)
            global_adjustment = 0.7  # Decrease scores
        else:
            global_adjustment = 1.0  # No change
        
        feature_map = np.clip(feature_map * global_adjustment, 0, 1)
        
        return feature_map, noise_pattern_map
    
    @staticmethod
    def analyze_texture_quality(image, patch_size=32):
        """
        Analyze texture quality and consistency
        AI-generated images often have repeating textures or lack natural texture variation
        """
        if len(image.shape) > 2 and image.shape[2] > 1:
            # Convert to grayscale for texture analysis
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            # Keep color for content analysis
            color_img = (image * 255).astype(np.uint8)
        else:
            gray = (image * 255).astype(np.uint8)
            color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        h, w = gray.shape
        rows = np.arange(0, h - patch_size, patch_size)
        cols = np.arange(0, w - patch_size, patch_size)
        num_rows = len(rows)
        num_cols = len(cols)
        
        all_stds = np.zeros((num_rows, num_cols))
        all_entropies = np.zeros((num_rows, num_cols))
        texture_consistency_map = np.zeros((num_rows, num_cols))
        feature_map = np.zeros((h, w))

        # 1. Local Binary Patterns for texture encoding
        radius = 2
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')

        # Simulate CNN embeddings with simple downsampled patches
        def get_cnn_like_features(img, scale=0.25):
            small_img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            sobelx = get_edge_detection(small_img, method='sobel', dx=1, dy=0, ksize=3)
            sobely = get_edge_detection(small_img, method='sobel', dx=0, dy=1, ksize=3)
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            features = [
                np.mean(small_img),
                np.std(small_img),
                np.mean(magnitude),
                np.std(magnitude),
                np.percentile(magnitude, 90)
            ]
            return np.array(features)

        # Get global CNN-like features
        global_cnn_features = get_cnn_like_features(gray)

        # 3. Gray-Level Co-occurrence Matrix (GLCM) features with simplified angles
        # Use cached GLCM computation
        downsampled = cv2.resize(gray, (max(w//4, 50), max(h//4, 50)))
        
        # Hash the image for caching
        img_hash = tuple(downsampled.flatten())
        distances = [1, 2]
        angles = [0, np.pi/2]
        
        # Use cached GLCM computation
        try:
            glcm = cached_glcm(img_hash, tuple(distances), tuple(angles), 256, True, True)
            glcm_contrast = graycoprops(glcm, 'contrast').mean()
            glcm_dissim = graycoprops(glcm, 'dissimilarity').mean()
            glcm_energy = graycoprops(glcm, 'energy').mean()
            glcm_corr = graycoprops(glcm, 'correlation').mean()
        except:
            # Fallback if caching fails
            glcm = cv2.resize(gray, (max(w//8, 25), max(h//8, 25)))
            glcm_contrast = np.std(glcm)
            glcm_energy = 1.0 / (1.0 + np.var(glcm))
            glcm_dissim = np.mean(np.abs(glcm - np.mean(glcm)))
            glcm_corr = 0.5

        global_lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points+2, range=(0, n_points+2), density=True)
        global_lbp_entropy = -np.sum(global_lbp_hist * np.log2(global_lbp_hist + 1e-10))

        edges = get_edge_detection(gray, method='canny', threshold1=50, threshold2=150)
        edge_density = np.sum(edges > 0) / (h * w)

        is_smooth_scene = edge_density < 0.05
        is_textured_scene = edge_density > 0.2

        # Vectorized computation for std and entropy across all patches
        for idx_i, i in enumerate(rows):
            for idx_j, j in enumerate(cols):
                patch = gray[i:i+patch_size, j:j+patch_size]
                all_stds[idx_i, idx_j] = np.std(patch)
                all_entropies[idx_i, idx_j] = -np.sum(np.abs(patch) * np.log2(np.abs(patch) + 1e-10))

        # Set initial unnatural scores based on std
        unnatural_mask = (all_stds < 0.005)
        unnatural_scores = np.zeros_like(all_stds)
        unnatural_scores[unnatural_mask] = 0.8

        # Continue with more detailed patch analysis
        for idx_i, i in enumerate(rows):
            for idx_j, j in enumerate(cols):
                patch = gray[i:i+patch_size, j:j+patch_size]
                patch_lbp = lbp[i:i+patch_size, j:j+patch_size]
                patch_lbp_hist, _ = np.histogram(patch_lbp.ravel(), bins=n_points+2, range=(0, n_points+2), density=True)
                patch_lbp_entropy = -np.sum(patch_lbp_hist * np.log2(patch_lbp_hist + 1e-10))
                patch_cnn_features = get_cnn_like_features(patch)
                patch_edges = edges[i:i+patch_size, j:j+patch_size]
                patch_edge_density = np.sum(patch_edges > 0) / (patch_size * patch_size)
                
                # Resize patch for GLCM analysis
                small_patch = cv2.resize(patch, (20, 20))
                
                # Try cached GLCM or compute directly with error handling
                try:
                    # Hash the patch for caching
                    patch_hash = tuple(small_patch.flatten())
                    patch_glcm = cached_glcm(patch_hash, (1,), (0, np.pi/2), 256, True, True)
                    patch_contrast = graycoprops(patch_glcm, 'contrast').mean()
                    patch_energy = graycoprops(patch_glcm, 'energy').mean()
                except:
                    # Fallback method if caching fails
                    patch_contrast = np.std(small_patch)
                    patch_energy = 1.0 / (1.0 + np.var(small_patch))

                # Calculate KL divergence for LBP histograms
                kl_div_lbp = np.sum(patch_lbp_hist * np.log2((patch_lbp_hist + 1e-10) / (global_lbp_hist + 1e-10)))
                kl_div_lbp = min(kl_div_lbp, 10)
                normalized_kl_lbp = kl_div_lbp / 10

                # Calculate feature difference
                feature_diff = np.linalg.norm(patch_cnn_features - global_cnn_features)
                max_diff = np.linalg.norm(np.ones_like(global_cnn_features) * 255)
                normalized_feature_diff = feature_diff / max_diff if max_diff > 0 else 0

                # Set expectation ranges based on image content
                if is_smooth_scene or (patch_edge_density < 0.05 and np.mean(patch) > 200):
                    expected_contrast_range = (0, 20)
                    expected_energy_range = (0.1, 0.5)
                    expected_lbp_entropy_range = (0, 2)
                elif is_textured_scene or patch_edge_density > 0.3:
                    expected_contrast_range = (20, 100)
                    expected_energy_range = (0.01, 0.2)
                    expected_lbp_entropy_range = (3, 6)
                else:
                    expected_contrast_range = (5, 50)
                    expected_energy_range = (0.05, 0.3)
                    expected_lbp_entropy_range = (1, 4)

                # Check if values are within natural ranges
                contrast_natural = expected_contrast_range[0] <= patch_contrast <= expected_contrast_range[1]
                energy_natural = expected_energy_range[0] <= patch_energy <= expected_energy_range[1]
                entropy_natural = expected_lbp_entropy_range[0] <= patch_lbp_entropy <= expected_lbp_entropy_range[1]

                # Collect indicators of unnatural textures
                unnatural_indicators = []
                if normalized_kl_lbp < 0.1:
                    unnatural_indicators.append(0.6)
                if normalized_feature_diff < 0.1:
                    unnatural_indicators.append(0.7)
                if not (contrast_natural and energy_natural and entropy_natural):
                    failed = sum([not contrast_natural, not energy_natural, not entropy_natural])
                    unnatural_indicators.append(0.5 * failed / 3)
                if patch_lbp_entropy < 1.0:
                    unnatural_indicators.append(0.6)

                # Calculate final texture score
                if unnatural_indicators:
                    texture_score = sum(unnatural_indicators) / len(unnatural_indicators)
                    if is_smooth_scene or (patch_edge_density < 0.05 and np.mean(patch) > 200):
                        texture_score *= 0.5
                    texture_consistency_map[idx_i, idx_j] = texture_score
                    feature_map[i:i+patch_size, j:j+patch_size] = texture_score

        return feature_map, texture_consistency_map

    @staticmethod
    def analyze_color_coherence(image, patch_size=16):
        """
        Analyze color coherence and distribution with improved false positive handling.
        AI-generated images sometimes have unnatural color relationships.
        """
        if len(image.shape) <= 2 or image.shape[2] < 3:
            # Not a color image
            return np.zeros(image.shape[:2]), 0
        
        # Convert to HSV color space
        hsv_img = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        
        h, w = image.shape[:2]
        num_rows = h // patch_size
        num_cols = w // patch_size
        color_coherence_map = np.zeros((num_rows, num_cols))
        feature_map = np.zeros((h, w))
        
        # Global hue statistics - using variance metrics instead of entropy
        hue_data = hsv_img[:,:,0].astype(float)
        
        # Handle circular nature of hue (0 and 180 are adjacent)
        hue_sin = np.sin(2 * np.pi * hue_data / 180)
        hue_cos = np.cos(2 * np.pi * hue_data / 180)
        mean_sin = np.mean(hue_sin)
        mean_cos = np.mean(hue_cos)
        
        # Calculate circular variance of hue
        global_hue_variance = 1 - np.sqrt(mean_sin**2 + mean_cos**2)
        
        # Detect if image has limited color palette (natural in some photos)
        limited_palette = global_hue_variance < 0.3
        
        # Detect gradients using Laplacian variance (simpler method)
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = np.var(laplacian)
        
        # Pre-compute patch coordinates
        patch_coords = [(i, j) for i in range(0, h - patch_size, patch_size) 
                        for j in range(0, w - patch_size, patch_size)]
        total_patches = len(patch_coords)
        suspicious_patches = 0
        
        # Process each patch
        for i, j in patch_coords:
            patch = hsv_img[i:i+patch_size, j:j+patch_size]
            
            # Calculate local hue variance using circular statistics
            hue_patch = patch[:,:,0].astype(float)
            hue_sin_patch = np.sin(2 * np.pi * hue_patch / 180)
            hue_cos_patch = np.cos(2 * np.pi * hue_patch / 180)
            mean_sin_patch = np.mean(hue_sin_patch)
            mean_cos_patch = np.mean(hue_cos_patch)
            local_hue_variance = 1 - np.sqrt(mean_sin_patch**2 + mean_cos_patch**2)
            
            # Check local saturation - very low saturation areas should be ignored
            local_mean_sat = np.mean(patch[:,:,1])
            
            # Skip very low saturation areas (naturally occurring in photos)
            if local_mean_sat < 20:
                continue
            
            # Calculate local Laplacian variance for gradient detection
            local_lap = laplacian[i:i+patch_size, j:j+patch_size]
            local_lap_var = np.var(local_lap)
            
            # Detect smooth, natural gradients using Laplacian variance ratio
            smooth_gradient = False
            if laplacian_var > 0:
                lap_var_ratio = local_lap_var / laplacian_var
                smooth_gradient = lap_var_ratio < 0.5 and lap_var_ratio > 0.1
            
            # AI images often have unnatural hue variance patterns
            # For limited palette images, expect more uniform hue variance
            unnatural_hue = False
            if limited_palette:
                unnatural_hue = local_hue_variance < 0.05 and not smooth_gradient
            else:
                # Compare local variance to global - too uniform is suspicious
                if global_hue_variance > 0:
                    hue_var_ratio = local_hue_variance / global_hue_variance
                    unnatural_hue = hue_var_ratio < 0.3 and not smooth_gradient
            
            if unnatural_hue and local_mean_sat > 30:
                row_idx = i // patch_size
                col_idx = j // patch_size
                if row_idx < num_rows and col_idx < num_cols:  # Safety check
                    color_coherence_map[row_idx, col_idx] = 1
                    feature_map[i:i+patch_size, j:j+patch_size] = 1
                    suspicious_patches += 1
        
        # Calculate overall score, adjusted to reduce false positives
        overall_score = min(suspicious_patches / max(total_patches, 1) * 1.5, 1.0) if total_patches > 0 else 0
        
        return feature_map, overall_score