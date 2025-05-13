import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops

class TextureFeaturesExtractor:
    @staticmethod
    def detect_noise_patterns(image, patch_size=16):
        """
        Detect unnatural noise patterns or lack of natural noise
        Natural images have characteristic noise patterns that AI often fails to replicate
        """
        if len(image.shape) > 2 and image.shape[2] > 1:
            # For color images, analyze each channel
            noise_maps = []
            for channel in range(image.shape[2]):
                channel_data = image[:,:,channel]
                # Apply high-pass filter to extract noise
                blurred = cv2.GaussianBlur(channel_data, (5, 5), 0)
                noise = channel_data - blurred
                noise_maps.append(noise)
            
            noise = np.mean(noise_maps, axis=0)
        else:
            # For grayscale
            blurred = cv2.GaussianBlur(image, (5, 5), 0)
            noise = image - blurred
        
        # Convert to standard scale for more consistent analysis
        noise_std = np.std(noise)
        if noise_std > 0:
            noise_normalized = noise / noise_std
        else:
            noise_normalized = noise
        
        # Analyze noise statistics in patches
        h, w = noise.shape
        feature_map = np.zeros((h, w))
        noise_pattern_map = np.zeros((h // patch_size, w // patch_size))
        
        # Calculate global noise statistics for comparison
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
        
        # Calculate noise variance to replace histogram comparison
        global_var = np.var(noise)
        
        gray_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) if len(image.shape) > 2 else (image * 255).astype(np.uint8)
        
        for i in range(0, h - patch_size, patch_size):
            for j in range(0, w - patch_size, patch_size):
                patch = noise[i:i+patch_size, j:j+patch_size]
                patch_norm = noise_normalized[i:i+patch_size, j:j+patch_size]
                
                # Calculate enhanced noise statistics
                std_dev = np.std(patch)
                entropy = -np.sum(np.abs(patch) * np.log2(np.abs(patch) + 1e-10))
                
                # Calculate local MAD (Median Absolute Deviation)
                local_median = np.median(patch)
                local_mad = np.median(np.abs(patch - local_median))
                
                # Calculate simplified local kurtosis using quartile method
                sorted_patch = np.sort(patch_norm.flatten())
                n_local = len(sorted_patch)
                if n_local > 4:  # Ensure enough points for quartile calculation
                    q1_pos_local = int(n_local * 0.25)
                    q3_pos_local = int(n_local * 0.75)
                    q1_local = sorted_patch[q1_pos_local]
                    q3_local = sorted_patch[q3_pos_local]
                    local_robust_kurtosis = (q3_local - q1_local) / (2 * local_mad) if local_mad > 0 else 1
                else:
                    local_robust_kurtosis = 1
                
                # Calculate local variance
                local_var = np.var(patch)
                
                # Variance comparison metric (replaces histogram comparison)
                if global_var > 0:
                    var_ratio = local_var / global_var
                    var_diff = abs(1 - var_ratio)
                else:
                    var_diff = 0 if local_var == 0 else 1
                
                # AI images might have unnaturally consistent noise patterns
                # or lack the random variations found in natural images
                is_unnatural = False
                
                # Combine multiple factors for more robust detection
                if std_dev < 0.005:  # Very low noise
                    unnatural_score = 0.8
                    is_unnatural = True
                elif entropy < 0.1:  # Low entropy
                    unnatural_score = 0.7
                    is_unnatural = True
                elif abs(local_robust_kurtosis - robust_kurtosis) > 1.5:  # Different kurtosis
                    unnatural_score = 0.6
                    is_unnatural = True
                elif var_diff > 0.7:  # Very different variance
                    unnatural_score = 0.5
                    is_unnatural = True
                else:
                    unnatural_score = 0
                
                # Account for image content - don't penalize naturally smooth areas
                if np.mean(gray_image[i:i+patch_size, j:j+patch_size]) > 240:
                    # Likely a naturally smooth bright area (sky, white wall, etc.)
                    unnatural_score *= 0.3
                elif np.mean(gray_image[i:i+patch_size, j:j+patch_size]) < 30:
                    # Likely a naturally smooth dark area (shadows, etc.)
                    unnatural_score *= 0.3
                
                if is_unnatural:
                    noise_pattern_map[i // patch_size, j // patch_size] = unnatural_score
                    feature_map[i:i+patch_size, j:j+patch_size] = unnatural_score
        
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
        texture_consistency_map = np.zeros((h // patch_size, w // patch_size))
        feature_map = np.zeros((h, w))
        
        # 1. Local Binary Patterns for texture encoding
        radius = 2
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # Simulate CNN embeddings with simple downsampled patches
        # This replaces Gabor filter bank with a simple feature extraction approach
        def get_cnn_like_features(img, scale=0.25):
            # Downsample for computational efficiency
            small_img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            # Apply Sobel filters as a simple edge detector in multiple directions
            sobelx = cv2.Sobel(small_img, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(small_img, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            # Get simple statistics as features
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
        # Downsample image for GLCM computation to save time
        downsampled = cv2.resize(gray, (max(w//4, 50), max(h//4, 50)))
        distances = [1, 2]  # Different distance offsets
        angles = [0, np.pi/2]  # Simplified: use only 2 angles instead of 4
        
        glcm = graycomatrix(downsampled, distances, angles, 256, symmetric=True, normed=True)
        glcm_contrast = graycoprops(glcm, 'contrast').mean()
        glcm_dissim = graycoprops(glcm, 'dissimilarity').mean()
        glcm_energy = graycoprops(glcm, 'energy').mean()
        glcm_corr = graycoprops(glcm, 'correlation').mean()
        
        # Get global texture statistics for comparison
        global_lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points+2, range=(0, n_points+2), density=True)
        global_lbp_entropy = -np.sum(global_lbp_hist * np.log2(global_lbp_hist + 1e-10))
        
        # Edge detection for structure analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)
        
        # Simple scene type classification
        is_smooth_scene = edge_density < 0.05
        is_textured_scene = edge_density > 0.2
        
        # Calculate detailed texture metrics for each patch
        for i in range(0, h - patch_size, patch_size):
            for j in range(0, w - patch_size, patch_size):
                patch = gray[i:i+patch_size, j:j+patch_size]
                
                # Calculate LBP histogram for patch
                patch_lbp = lbp[i:i+patch_size, j:j+patch_size]
                patch_lbp_hist, _ = np.histogram(patch_lbp.ravel(), bins=n_points+2, range=(0, n_points+2), density=True)
                
                # Calculate entropy of LBP histogram (texture complexity measure)
                patch_lbp_entropy = -np.sum(patch_lbp_hist * np.log2(patch_lbp_hist + 1e-10))
                
                # Calculate CNN-like features for this patch
                patch_cnn_features = get_cnn_like_features(patch)
                
                # Calculate edge content in this patch
                patch_edges = edges[i:i+patch_size, j:j+patch_size]
                patch_edge_density = np.sum(patch_edges > 0) / (patch_size * patch_size)
                
                # Calculate GLCM for this patch (on a smaller version to save computation)
                small_patch = cv2.resize(patch, (20, 20))
                try:
                    # Use only 2 angles to simplify computation
                    patch_glcm = graycomatrix(small_patch, [1], [0, np.pi/2], levels=256, symmetric=True, normed=True)
                    patch_contrast = graycoprops(patch_glcm, 'contrast').mean()
                    patch_energy = graycoprops(patch_glcm, 'energy').mean()
                except:
                    # Fallback if GLCM fails
                    patch_contrast = 0
                    patch_energy = 0
                
                # Compare patch texture to global texture
                # LBP comparison - KL divergence between local and global histograms
                kl_div_lbp = np.sum(patch_lbp_hist * np.log2((patch_lbp_hist + 1e-10) / (global_lbp_hist + 1e-10)))
                kl_div_lbp = min(kl_div_lbp, 10)  # Cap at 10 to avoid extreme values
                normalized_kl_lbp = kl_div_lbp / 10
                
                # CNN feature comparison - normalized euclidean distance
                feature_diff = np.linalg.norm(patch_cnn_features - global_cnn_features)
                max_diff = np.linalg.norm(np.ones_like(global_cnn_features) * 255)  # Maximum possible difference
                normalized_feature_diff = feature_diff / max_diff if max_diff > 0 else 0
                
                # Content-aware analysis
                # Set expectations based on content type
                if is_smooth_scene or (patch_edge_density < 0.05 and np.mean(patch) > 200):
                    # Likely sky, water, or other smooth area
                    expected_contrast_range = (0, 20)
                    expected_energy_range = (0.1, 0.5)
                    expected_lbp_entropy_range = (0, 2)
                elif is_textured_scene or patch_edge_density > 0.3:
                    # Highly textured area
                    expected_contrast_range = (20, 100)
                    expected_energy_range = (0.01, 0.2)
                    expected_lbp_entropy_range = (3, 6)
                else:
                    # Mixed content
                    expected_contrast_range = (5, 50)
                    expected_energy_range = (0.05, 0.3)
                    expected_lbp_entropy_range = (1, 4)
                
                # Check if patch meets expectations for natural textures
                contrast_natural = expected_contrast_range[0] <= patch_contrast <= expected_contrast_range[1]
                energy_natural = expected_energy_range[0] <= patch_energy <= expected_energy_range[1]
                entropy_natural = expected_lbp_entropy_range[0] <= patch_lbp_entropy <= expected_lbp_entropy_range[1]
                
                # Final unnatural texture score combining metrics
                unnatural_indicators = []
                
                # 1. Texture consistency with global image
                if normalized_kl_lbp < 0.1:  # Too similar to global pattern
                    unnatural_indicators.append(0.6)
                
                # 2. CNN feature consistency
                if normalized_feature_diff < 0.1:  # Too uniform response
                    unnatural_indicators.append(0.7)
                
                # 3. Content-appropriate texture check
                if not (contrast_natural and energy_natural and entropy_natural):
                    # Count how many checks failed
                    failed = sum([not contrast_natural, not energy_natural, not entropy_natural])
                    unnatural_indicators.append(0.5 * failed / 3)
                
                # 4. LBP entropy check (too low entropy often indicates AI textures)
                if patch_lbp_entropy < 1.0:
                    unnatural_indicators.append(0.6)
                
                # Calculate final score - average of all indicators
                if unnatural_indicators:
                    texture_score = sum(unnatural_indicators) / len(unnatural_indicators)
                    
                    # Reduce score for naturally smooth areas
                    if is_smooth_scene or (patch_edge_density < 0.05 and np.mean(patch) > 200):
                        texture_score *= 0.5
                    
                    texture_consistency_map[i // patch_size, j // patch_size] = texture_score
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
        color_coherence_map = np.zeros((h // patch_size, w // patch_size))
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
        
        # Count suspicious patches
        suspicious_patches = 0
        total_patches = 0
        
        for i in range(0, h - patch_size, patch_size):
            for j in range(0, w - patch_size, patch_size):
                patch = hsv_img[i:i+patch_size, j:j+patch_size]
                total_patches += 1
                
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
                    color_coherence_map[i // patch_size, j // patch_size] = 1
                    feature_map[i:i+patch_size, j:j+patch_size] = 1
                    suspicious_patches += 1
        
        # Calculate overall score, adjusted to reduce false positives
        overall_score = min(suspicious_patches / max(total_patches, 1) * 1.5, 1.0) if total_patches > 0 else 0
        
        return feature_map, overall_score