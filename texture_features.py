import imagehash
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from skimage.feature import graycoprops
from PIL import Image
from skimage.feature import graycoprops, graycomatrix

class TextureFeatureExtractor:
    def _extract_noise_feature(self, image: np.ndarray) -> np.ndarray:
        """
        Analyze noise distribution to detect AI-generated inconsistencies.

        Args:
            image: Input image (BGR format if color)

        Returns:
            2D feature map same size as input image with values between 0 and 1
        """
        # Handle input normalization and color conversion
        if len(image.shape) == 3:
            # Convert BGR to grayscale and handle float inputs
            if np.issubdtype(image.dtype, np.floating):
                gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            if np.issubdtype(gray.dtype, np.floating):
                gray = (gray * 255).astype(np.uint8)

        # Extract noise pattern using Gaussian residual
        blurred = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
        noise = gray.astype(np.float32) - blurred
        noise_normalized = noise / (np.std(noise) + 1e-6)

        # Initialize parameters
        patch_size = 16
        h, w = gray.shape
        rows = np.arange(0, h - patch_size, patch_size)
        cols = np.arange(0, w - patch_size, patch_size)
        feature_map = np.zeros((h, w), dtype=np.float32)

        # Calculate global statistics
        global_var = np.var(noise)
        global_mad = np.median(np.abs(noise - np.median(noise)))

        # Process patches in vectorized manner
        for i in rows:
            for j in cols:
                # Extract noise patch and normalized version
                patch = noise[i:i+patch_size, j:j+patch_size]
                patch_norm = noise_normalized[i:i+patch_size, j:j+patch_size]

                # Calculate local statistics
                local_std = np.std(patch)
                local_mad = np.median(np.abs(patch - np.median(patch)))
                local_var = np.var(patch)

                # Calculate entropy of absolute values
                abs_patch = np.abs(patch_norm)
                entropy = -np.sum(abs_patch * np.log2(abs_patch + 1e-10))

                # Calculate robust kurtosis estimate
                sorted_patch = np.sort(patch_norm.flatten())
                q1, q3 = np.quantile(sorted_patch, [0.25, 0.75])
                robust_kurtosis = (q3 - q1) / (2 * local_mad) if local_mad > 0 else 1

                # Calculate variance ratio
                var_ratio = local_var / global_var if global_var > 0 else 0
                var_diff = abs(1 - var_ratio)

                # Calculate initial score based on multiple criteria
                score = 0
                if local_std < 0.005:
                    score = 0.8
                elif entropy < 0.1:
                    score = 0.7
                elif abs(robust_kurtosis - (q3 - q1)/(2*global_mad)) > 1.5:
                    score = 0.6
                elif var_diff > 0.7:
                    score = 0.5

                # Adjust for bright/dark areas
                gray_patch = gray[i:i+patch_size, j:j+patch_size]
                patch_mean = np.mean(gray_patch)
                if patch_mean > 240 or patch_mean < 30:
                    score *= 0.3

                # Apply score to feature map
                feature_map[i:i+patch_size, j:j+patch_size] = score

        # Global variance adjustment
        global_variance_magnitude = np.log1p(global_var * 1000)
        if global_variance_magnitude < -5:
            feature_map = np.clip(feature_map * 1.2, 0, 1)
        elif global_variance_magnitude > 2:
            feature_map = np.clip(feature_map * 0.7, 0, 1)

        return feature_map
    
    def _extract_texture_feature(self, image: np.ndarray) -> np.ndarray:
        """
        Analyze texture consistency and identify AI artifacts in textures.

        Args:
            image: Input image (BGR format if color)

        Returns:
            2D feature map same size as input image with values between 0 and 1
        """
        # Convert to grayscale and handle input formats
        if len(image.shape) == 3:
            if np.issubdtype(image.dtype, np.floating):
                gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            if np.issubdtype(gray.dtype, np.floating):
                gray = (gray * 255).astype(np.uint8)

        h, w = gray.shape
        patch_size = 32
        feature_map = np.zeros((h, w), dtype=np.float32)

        # Local Binary Patterns texture analysis
        radius = 2
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')

        # Edge density analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = cv2.resize(
            (cv2.boxFilter(edges, -1, (patch_size, patch_size))) / 255,
            (w, h)
        )

        # GLCM texture analysis
        glcm = graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast').mean()
        energy = graycoprops(glcm, 'energy').mean()

        # Process texture patches
        for i in range(0, h - patch_size, patch_size//2):
            for j in range(0, w - patch_size, patch_size//2):
                # Extract patch with overlap handling
                pi, pj = min(i, h-patch_size), min(j, w-patch_size)
                patch = gray[pi:pi+patch_size, pj:pj+patch_size]

                # Local texture statistics
                patch_std = np.std(patch)
                lbp_hist = np.histogram(lbp[pi:pi+patch_size, pj:pj+patch_size], 
                                       bins=n_points+2, range=(0, n_points+2))[0]
                lbp_hist = lbp_hist / (patch_size**2 + 1e-10)
                lbp_entropy = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-10))

                # Local GLCM features
                patch_glcm = graycomatrix(patch, [1], [0], levels=256, symmetric=True, normed=True)
                patch_contrast = graycoprops(patch_glcm, 'contrast').mean()
                patch_energy = graycoprops(patch_glcm, 'energy').mean()

                # Texture anomaly detection
                score = 0
                # Check for low variation
                if patch_std < 5:
                    score += 0.4
                # Check for unusual LBP patterns
                if lbp_entropy < 2.0:
                    score += 0.3
                # Check GLCM consistency
                if abs(patch_contrast - contrast) > contrast*0.5:
                    score += 0.2
                if abs(patch_energy - energy) > energy*0.5:
                    score += 0.1

                # Apply edge density weighting
                local_edge_density = np.mean(edge_density[pi:pi+patch_size, pj:pj+patch_size])
                if local_edge_density < 0.05:
                    score *= 0.5  # Reduce score in smooth areas
                elif local_edge_density > 0.2:
                    score *= 1.2  # Increase in textured areas

                # Apply to feature map with Gaussian weighting
                feature_map[pi:pi+patch_size, pj:pj+patch_size] = np.maximum(
                    feature_map[pi:pi+patch_size, pj:pj+patch_size],
                    np.minimum(score, 1.0) * cv2.getGaussianKernel(patch_size, patch_size/3)[:, np.newaxis]
                )

        # Normalize and smooth final result
        feature_map = cv2.GaussianBlur(feature_map, (15, 15), 3)
        feature_map -= feature_map.min()
        feature_map /= feature_map.max() + 1e-10

        return np.clip(feature_map, 0, 1).astype(np.float32)
    
    def _extract_color_feature(self, image: np.ndarray) -> np.ndarray:
        """
        Detect color distribution anomalies common in AI-generated images.

        Args:
            image: Input image (BGR format if color)

        Returns:
            2D feature map same size as input image with values between 0 and 1
        """
        # Handle non-color images
        if len(image.shape) != 3 or image.shape[2] != 3:
            return np.zeros(image.shape[:2], dtype=np.float32)

        # Convert input to proper format
        if np.issubdtype(image.dtype, np.floating):
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image.astype(np.uint8)

        # Convert to HSV color space
        hsv_img = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2HSV)
        h, w = img_uint8.shape[:2]
        feature_map = np.zeros((h, w), dtype=np.float32)
        patch_size = 16

        # Global hue variance calculation
        hue = hsv_img[:, :, 0].astype(np.float32)
        hue_rad = np.deg2rad(hue)
        mean_sin = np.mean(np.sin(hue_rad))
        mean_cos = np.mean(np.cos(hue_rad))
        global_hue_var = 1 - np.sqrt(mean_sin**2 + mean_cos**2)
        limited_palette = global_hue_var < 0.3

        # Edge preservation analysis
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        global_lap_var = np.var(laplacian)

        # Process patches with vectorized operations
        for i in range(0, h - patch_size + 1, patch_size):
            for j in range(0, w - patch_size + 1, patch_size):
                patch = hsv_img[i:i+patch_size, j:j+patch_size]
                hue_patch = patch[:, :, 0].astype(np.float32)
                sat_patch = patch[:, :, 1]

                # Local hue variance
                hue_rad_patch = np.deg2rad(hue_patch)
                mean_sin_p = np.mean(np.sin(hue_rad_patch))
                mean_cos_p = np.mean(np.cos(hue_rad_patch))
                local_hue_var = 1 - np.sqrt(mean_sin_p**2 + mean_cos_p**2)

                # Skip low saturation areas
                if np.mean(sat_patch) < 20:
                    continue

                # Local texture analysis
                local_lap = laplacian[i:i+patch_size, j:j+patch_size]
                lap_ratio = np.var(local_lap) / (global_lap_var + 1e-6)
                smooth_gradient = 0.1 < lap_ratio < 0.5

                # Detect unnatural color consistency
                unnatural = False
                if limited_palette:
                    unnatural = (local_hue_var < 0.05) and not smooth_gradient
                else:
                    var_ratio = local_hue_var / (global_hue_var + 1e-6)
                    unnatural = (var_ratio < 0.3) and not smooth_gradient

                if unnatural and np.mean(sat_patch) > 30:
                    feature_map[i:i+patch_size, j:j+patch_size] = 1.0

        return feature_map
    
    def _extract_hash_feature(self, image: np.ndarray) -> np.ndarray:
        """
        Compare perceptual hash similarity to known AI patterns.

        Args:
            image: Input image (BGR format if color)

        Returns:
            2D feature map same size as input image with values between 0 and 1
        """
        h, w = image.shape[:2]
        feature_map = np.zeros((h, w), dtype=np.float32)

        try:
            # Convert to PIL Image with proper color space handling
            if image.dtype == np.float32 or image.dtype == np.float64:
                image_uint8 = (image * 255).astype(np.uint8)
            else:
                image_uint8 = image.astype(np.uint8)

            if image_uint8.shape[2] == 3:  # BGR to RGB conversion
                image_pil = Image.fromarray(cv2.cvtColor(image_uint8, cv2.COLOR_BGR2RGB))
            else:
                image_pil = Image.fromarray(image_uint8)

            # Calculate multiple perceptual hashes
            phash = imagehash.phash(image_pil, hash_size=8)
            dhash = imagehash.dhash(image_pil, hash_size=8)

            # Convert hashes to binary representations
            phash_bin = bin(int(str(phash), 16))[2:].zfill(64)
            dhash_bin = bin(int(str(dhash), 16))[2:].zfill(64)

            # 1. Advanced bit-run analysis
            def analyze_bit_runs(hash_bin):
                runs = []
                current_val = hash_bin[0]
                current_run = 1
                for bit in hash_bin[1:]:
                    if bit == current_val:
                        current_run += 1
                    else:
                        runs.append(current_run)
                        current_val = bit
                        current_run = 1
                runs.append(current_run)
                return {
                    'max_run': max(runs),
                    'avg_run': sum(runs)/len(runs),
                    'std_run': np.std(runs),
                    'long_runs': sum(1 for r in runs if r > 4)
                }

            phash_runs = analyze_bit_runs(phash_bin)
            dhash_runs = analyze_bit_runs(dhash_bin)

            # 2. Regularity pattern detection
            phash_reg = sum(phash_bin[i] == phash_bin[i+1] for i in range(63)) / 63
            dhash_reg = sum(dhash_bin[i] == dhash_bin[i+1] for i in range(63)) / 63
            regularity_score = (phash_reg + dhash_reg) / 2

            # 3. Mirror symmetry analysis
            phash_sym = sum(phash_bin[i] == phash_bin[63-i] for i in range(32)) / 32
            dhash_sym = sum(dhash_bin[i] == dhash_bin[63-i] for i in range(32)) / 32
            symmetry_score = (phash_sym + dhash_sym) / 2

            # 4. Cross-hash consistency analysis
            cross_sim = sum(p == d for p,d in zip(phash_bin, dhash_bin)) / 64

            # 5. Entropy calculation via bit density
            phash_density = phash_bin.count('1')/64
            dhash_density = dhash_bin.count('1')/64
            entropy_score = (abs(phash_density-0.5) + abs(dhash_density-0.5))/1.0

            # Composite AI probability score
            ai_score = (
                0.35 * regularity_score +
                0.35 * symmetry_score +
                0.20 * min(max(cross_sim-0.6, 0)/0.4, 1) +  # Scaled cross-sim
                0.10 * entropy_score
            )

            # Apply non-linear thresholding
            if ai_score > 0.85:
                ai_confidence = min((ai_score - 0.85) / 0.15, 1.0)
            elif ai_score > 0.7:
                ai_confidence = (ai_score - 0.7) / 0.15
            else:
                ai_confidence = 0

            # Generate radial gradient feature map
            if ai_confidence > 0:
                center_y, center_x = h//2, w//2
                y_coords, x_coords = np.indices((h, w))
                dist_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
                max_dist = np.sqrt(center_x**2 + center_y**2)

                # Sigmoid decay function for smooth gradient
                decay = 1 / (1 + np.exp(12*(dist_from_center/max_dist - 0.5)))
                feature_map = decay * ai_confidence

        except Exception as e:
            pass  # Return zero map on any failure

        return np.clip(feature_map, 0, 1).astype(np.float32)