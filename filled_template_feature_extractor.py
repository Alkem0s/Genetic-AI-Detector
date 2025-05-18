import numpy as np
import cv2
from skimage.transform import radon
from skimage.feature import graycoprops, graycomatrix
from skimage.metrics import structural_similarity as ssim
from skimage.feature import local_binary_pattern
from PIL import Image
import imagehash
from typing import Tuple, Dict, List, Any

from ai_detection_config import AIDetectionConfig

class FeatureExtractor:
    """
    Class for feature extraction in AI image detection system.
    Implements methods to extract various features that can identify AI-generated artifacts.
    
    The GeneticFeatureOptimizer expects this module to extract features
    from patches of images and return them in a consistent format.
    """
    
    def __init__(self, config=None):
        """
        Initialize the feature extractor with configuration parameters.
        
        Args:
            config: Configuration object or None to use defaults
        """
        if config is None:
            self.config = AIDetectionConfig()  # Default to AIDetectionConfig
        else:
            self.config = config
        
        # Each extracted feature should be normalized to range [0.0, 1.0]
        # where higher values typically indicate stronger AI artifacts presence
    
    def extract_all_features(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Extract all features from an image.
        
        This is the main method that the genetic algorithm will call.
        It should extract all features and return them in a standardized format.
        
        Args:
            image: Input image as numpy array (height, width, channels)
            
        Returns:
            Tuple containing:
                - visualization: Visualization image with highlighted features (can be None)
                - feature_stack: 3D array where each channel is a different feature map
                - metadata: Dictionary with additional metadata about extracted features
        """
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Initialize feature stack - MUST have the same spatial dimensions as the input image
        # with one channel for each feature type
        feature_stack = np.zeros((height, width, len(self.config.feature_weights.keys())), dtype=np.float32)

        # Extract individual features and populate the stack
        # Each feature should be normalized to [0, 1] range
        
        # Example for gradient feature extraction (implement your actual algorithm here)
        feature_stack[:, :, 0] = self._extract_gradient_feature(image)
        
        # Example for pattern detection
        feature_stack[:, :, 1] = self._extract_pattern_feature(image)
        
        # Noise analysis
        feature_stack[:, :, 2] = self._extract_noise_feature(image)
        
        # Edge coherence
        feature_stack[:, :, 3] = self._extract_edge_feature(image)
        
        # Symmetry detection
        feature_stack[:, :, 4] = self._extract_symmetry_feature(image)
        
        # Texture analysis
        feature_stack[:, :, 5] = self._extract_texture_feature(image)
        
        # Color distribution analysis
        feature_stack[:, :, 6] = self._extract_color_feature(image)
        
        # Perceptual hash comparison
        feature_stack[:, :, 7] = self._extract_hash_feature(image)
        
        # Create visualization image (optional)
        visualization = self._create_visualization(image, feature_stack)
        
        # Prepare metadata
        metadata = {
            "image_dimensions": (height, width),
            "feature_weights": self.config.feature_weights,
            "feature_statistics": self._compute_feature_statistics(feature_stack)
        }
        
        return visualization, feature_stack, metadata
    
    def _extract_gradient_feature(self, image: np.ndarray) -> np.ndarray:
        """
        Extract gradient perfection feature which identifies unnaturally perfect gradients.

        Args:
            image: Input image (assumed to be in BGR format if color)

        Returns:
            2D feature map same size as input image with values between 0 and 1
        """
        # Convert to grayscale if necessary
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Ensure the grayscale image is in uint8 (0-255)
        if np.issubdtype(gray.dtype, np.floating):
            gray = (gray * 255).astype(np.uint8)

        # Compute Scharr gradients
        scharrx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        scharry = cv2.Scharr(gray, cv2.CV_64F, 0, 1)

        # Calculate gradient magnitude and direction
        magnitude = np.sqrt(scharrx**2 + scharry**2)
        direction = np.arctan2(scharry, scharrx)

        # Process parameters
        patch_size = 16
        h, w = gray.shape
        pm_h = h // patch_size
        pm_w = w // patch_size
        perfection_map = np.zeros((pm_h, pm_w))
        direction_coherence = np.zeros((pm_h, pm_w))

        # Analyze each patch
        for i in range(0, h - patch_size, patch_size):
            for j in range(0, w - patch_size, patch_size):
                pm_i = i // patch_size
                pm_j = j // patch_size

                patch = magnitude[i:i+patch_size, j:j+patch_size]
                dir_patch = direction[i:i+patch_size, j:j+patch_size]

                if patch.std() > 0:
                    coef_var = patch.std() / patch.mean() if patch.mean() > 0 else float('inf')
                    coef_var_clipped = min(coef_var, 1.0)
                    perfection_score = 1 - coef_var_clipped

                    # FFT-based direction coherence analysis
                    complex_dir = np.exp(1j * dir_patch)
                    fft_dir = np.fft.fft2(complex_dir)
                    power_spectrum = np.abs(fft_dir)**2
                    total_power = np.sum(power_spectrum)
                    if total_power > 0:
                        if power_spectrum.size > 9:
                            center_power = np.sum(power_spectrum[:3, :3])
                        else:
                            center_power = total_power
                        spectral_concentration = center_power / total_power
                    else:
                        spectral_concentration = 0

                    direction_coherence[pm_i, pm_j] = spectral_concentration
                    perfection_map[pm_i, pm_j] = perfection_score * (0.5 + 0.5 * spectral_concentration)
                else:
                    perfection_map[pm_i, pm_j] = 0

        # Adjust for overall high perfection areas
        high_perfection_ratio = np.sum(perfection_map > 0.7) / perfection_map.size
        if high_perfection_ratio > 0.5:
            perfection_map *= (0.5 + high_perfection_ratio * 0.5)

        # Upsample to original image size using nearest-neighbor interpolation
        feature_map = cv2.resize(perfection_map, (w, h), interpolation=cv2.INTER_NEAREST)

        # Clip to ensure values are between 0 and 1
        feature_map = np.clip(feature_map, 0, 1)

        return feature_map
    
    def _extract_pattern_feature(self, image: np.ndarray) -> np.ndarray:
        """
        Detect repeating patterns that may indicate AI artifacts.
        
        Args:
            image: Input image (BGR format if color)
            
        Returns:
            2D feature map same size as input image with values between 0 and 1
        """
        # Convert to grayscale and handle input ranges
        if len(image.shape) == 3:
            if np.issubdtype(image.dtype, np.floating):
                gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            if np.issubdtype(gray.dtype, np.floating):
                gray = (gray * 255).astype(np.uint8)
    
        # FFT analysis
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-10)
        
        # Normalize spectrum
        mag_min = magnitude_spectrum.min()
        mag_max = magnitude_spectrum.max()
        magnitude_spectrum_norm = ((magnitude_spectrum - mag_min) / 
                                  (mag_max - mag_min)) if (mag_max - mag_min) > 0 else np.zeros_like(magnitude_spectrum)
    
        # Adaptive thresholding
        mean_spectrum = np.mean(magnitude_spectrum_norm)
        std_spectrum = np.std(magnitude_spectrum_norm)
        adaptive_threshold = mean_spectrum + 0.7 * std_spectrum  # Using original threshold parameter
        peaks = magnitude_spectrum_norm > adaptive_threshold
    
        # Mask DC component
        h, w = gray.shape
        Y, X = np.ogrid[:h, :w]
        center = (h//2, w//2)
        radius = min(h, w) // 20
        peaks[np.sqrt((X - center[1])**2 + (Y - center[0])**2) <= radius] = False
    
        # Pattern analysis
        peak_count = np.sum(peaks)
        pattern_score = 0.0
        
        if peak_count > 5:
            # Radial analysis
            py, px = np.where(peaks)
            distances = np.sqrt((px - center[1])**2 + (py - center[0])**2)
            radial_hist = np.histogram(distances, bins=20)[0]
            radial_hist = radial_hist / radial_hist.sum() if radial_hist.sum() > 0 else radial_hist
            radial_entropy = -np.sum(radial_hist * np.log2(radial_hist + 1e-10))
            max_entropy = np.log2(len(radial_hist))
            radial_score = 1 - (radial_entropy / max_entropy) if max_entropy > 0 else 0
    
            # Angular analysis
            angles = np.arctan2(py - center[0], px - center[1])
            angle_hist = np.histogram(angles, bins=16, range=(-np.pi, np.pi))[0]
            angle_hist = angle_hist / angle_hist.sum() if angle_hist.sum() > 0 else angle_hist
            angle_entropy = -np.sum(angle_hist * np.log2(angle_hist + 1e-10))
            angle_score = 1 - (angle_entropy / np.log2(16)) if angle_hist.size > 0 else 0
    
            pattern_score = radial_score * (1 - 0.5 * angle_score)
    
        # Create spatial feature map using inverse FFT of detected patterns
        f_shift_filtered = f_shift * peaks.astype(np.complex64)
        spatial_pattern = np.abs(np.fft.ifft2(np.fft.ifftshift(f_shift_filtered)))
        spatial_pattern_norm = (spatial_pattern - spatial_pattern.min()) / (spatial_pattern.ptp() + 1e-10)
        
        # Combine global score with local patterns
        feature_map = np.clip(spatial_pattern_norm * pattern_score, 0, 1)
        
        return feature_map.astype(np.float32)
    
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
    
    def _extract_edge_feature(self, image: np.ndarray) -> np.ndarray:
        """
        Examine edge coherence and artifacts typical in AI-generated images.

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

        # Multi-scale edge detection pyramid
        pyramid_levels = 3
        edge_pyramids = []
        current_img = gray.copy()

        for _ in range(pyramid_levels):
            edges = cv2.Canny(current_img, 100, 200)
            edge_pyramids.append(edges)
            current_img = cv2.pyrDown(current_img)

        # Combine edge maps
        combined_edges = edge_pyramids[0].copy()
        for edges in edge_pyramids[1:]:
            resized = cv2.resize(edges, (gray.shape[1], gray.shape[0]))
            combined_edges = cv2.bitwise_or(combined_edges, resized)

        # Global orientation analysis
        theta = np.linspace(0, 180, 36)
        global_radon = radon(combined_edges, theta=theta, circle=True)
        global_profile = np.sum(global_radon, axis=0)
        global_profile /= np.sum(global_profile) + 1e-10

        # Global texture analysis
        glcm = graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
        global_contrast = np.mean(graycoprops(glcm, 'contrast'))

        # Patch analysis parameters
        patch_size = 16
        h, w = gray.shape
        map_h, map_w = h // patch_size, w // patch_size
        edge_scores = np.zeros((map_h, map_w))

        # Process each patch
        for i in range(map_h):
            for j in range(map_w):
                y = i * patch_size
                x = j * patch_size
                edge_patch = combined_edges[y:y+patch_size, x:x+patch_size]
                gray_patch = gray[y:y+patch_size, x:x+patch_size]

                if np.sum(edge_patch) > patch_size:
                    # Local orientation analysis
                    patch_radon = radon(edge_patch, theta=theta, circle=True)
                    patch_profile = np.sum(patch_radon, axis=0)
                    patch_profile /= np.sum(patch_profile) + 1e-10

                    # Orientation entropy
                    entropy = -np.sum(patch_profile * np.log2(patch_profile + 1e-10))
                    max_entropy = np.log2(len(theta))
                    entropy_score = 1 - (entropy / max_entropy) if max_entropy > 0 else 0

                    # Profile correlation
                    correlation = np.corrcoef(patch_profile, global_profile)[0, 1]
                    correlation_score = 0.5 * (1 - abs(correlation))

                    # Combined score
                    score = entropy_score * 0.7 + correlation_score * 0.3

                    # Contrast adjustment
                    patch_glcm = graycomatrix(gray_patch, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
                    patch_contrast = np.mean(graycoprops(patch_glcm, 'contrast'))
                    contrast_ratio = patch_contrast / (global_contrast + 1e-10)

                    if 0.7 < contrast_ratio < 1.3:
                        score *= 0.7

                    edge_scores[i, j] = score

        # Create full-resolution feature map
        feature_map = cv2.resize(edge_scores, (w, h), interpolation=cv2.INTER_NEAREST)
        feature_map = np.clip(feature_map, 0, 1)

        return feature_map.astype(np.float32)
    
    def _extract_symmetry_feature(self, image: np.ndarray) -> np.ndarray:
        """
        Measure unnatural symmetry often present in AI-generated images.
        
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
        feature_map = np.zeros((h, w), dtype=np.float32)
    
        # Multi-scale symmetry analysis
        scales = [1.0, 0.5]
        scale_weights = [0.7, 0.3]
        symmetry_scores = []
        axis_preferences = []
    
        for scale, weight in zip(scales, scale_weights):
            if scale != 1.0:
                scaled_gray = cv2.resize(gray, (int(w*scale), int(h*scale)))
            else:
                scaled_gray = gray
    
            # Horizontal symmetry analysis
            half_w = scaled_gray.shape[1] // 2
            left = scaled_gray[:, :half_w]
            right = np.fliplr(scaled_gray[:, half_w:])
            min_width = min(left.shape[1], right.shape[1])
            h_ssim = ssim(left[:, :min_width], right[:, :min_width], data_range=255)
            
            # Vertical symmetry analysis
            half_h = scaled_gray.shape[0] // 2
            top = scaled_gray[:half_h, :]
            bottom = np.flipud(scaled_gray[half_h:, :])
            min_height = min(top.shape[0], bottom.shape[0])
            v_ssim = ssim(top[:min_height, :], bottom[:min_height, :], data_range=255)
    
            # Combine axis scores
            axis_score = max(h_ssim, v_ssim)
            symmetry_scores.append(axis_score * weight)
            axis_preferences.append(h_ssim > v_ssim)
    
        # Calculate combined symmetry score
        combined_score = np.sum(symmetry_scores) / np.sum(scale_weights)
        primary_horizontal = np.mean(axis_preferences) > 0.5
    
        # Edge-based symmetry verification
        edges = cv2.Canny(gray, 100, 200)
        if primary_horizontal:
            half_w = w // 2
            left_edges = edges[:, :half_w]
            right_edges = np.fliplr(edges[:, half_w:])
            edge_match = np.mean(left_edges == right_edges[:, :left_edges.shape[1]])
        else:
            half_h = h // 2
            top_edges = edges[:half_h, :]
            bottom_edges = np.flipud(edges[half_h:, :])
            edge_match = np.mean(top_edges == bottom_edges[:top_edges.shape[0], :])
    
        # Boost score for perfect edge alignment
        if edge_match > 0.9:
            combined_score = min(combined_score * 1.2, 1.0)
    
        # Create axis-aligned gradient pattern
        if primary_horizontal:
            center_x = w // 2
            x_dist = np.abs(np.arange(w) - center_x) / (w/2)
            gradient = 1 - np.minimum(x_dist, 1.0)
        else:
            center_y = h // 2
            y_dist = np.abs(np.arange(h) - center_y) / (h/2)
            gradient = 1 - np.minimum(y_dist, 1.0)
        
        # Apply gradient pattern scaled by symmetry score
        if primary_horizontal:
            feature_map = np.outer(np.ones(h), gradient) * combined_score
        else:
            feature_map = np.outer(gradient, np.ones(w)) * combined_score
    
        return np.clip(feature_map, 0, 1).astype(np.float32)
    
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
    
    def _compute_feature_statistics(self, feature_stack: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Compute statistics for each feature map for metadata.
        
        Args:
            feature_stack: 3D array containing all feature maps
            
        Returns:
            Dictionary with statistics for each feature
        """
        stats = {}
        
        for i, feature_name in enumerate(self.config.feature_weights.keys()):
            if i < feature_stack.shape[2]:
                feature_map = feature_stack[:, :, i]
                stats[feature_name] = {
                    "mean": float(np.mean(feature_map)),
                    "min": float(np.min(feature_map)),
                    "max": float(np.max(feature_map)),
                    "std": float(np.std(feature_map))
                }
        
        return stats
    
    def _create_visualization(self, image: np.ndarray, feature_stack: np.ndarray) -> np.ndarray:
        """
        Create a visualization of the extracted features overlaid on the image.
        
        Args:
            image: Original input image
            feature_stack: 3D array containing all feature maps
            
        Returns:
            Visualization image with features highlighted
        """
        # Simple visualization: Combine features into a heatmap
        if len(image.shape) == 3:
            visualization = image.copy()
        else:
            # Convert grayscale to RGB for visualization
            visualization = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Create a combined feature map (average of all features)
        combined_feature = np.mean(feature_stack, axis=2)
        
        # Normalize to 0-1 range
        if np.max(combined_feature) > 0:
            combined_feature = combined_feature / np.max(combined_feature)
        
        # Create a heatmap overlay (higher values are more red)
        heatmap = np.zeros_like(visualization)
        heatmap[:, :, 2] = (combined_feature * 255).astype(np.uint8)  # Red channel
        
        # Blend with original image
        alpha = 0.6  # Transparency of overlay
        blended = cv2.addWeighted(visualization, 1 - alpha, heatmap, alpha, 0)
        
        return blended
    
    def extract_patch_features(self, image: np.ndarray, patch_size: int = 16) -> np.ndarray:
        """
        Extract average feature values for each patch in the image.
        This function is particularly useful for the genetic algorithm's rule application.
        
        Args:
            image: Input image
            patch_size: Size of patches to analyze
            
        Returns:
            4D array with shape (n_patches_h, n_patches_w, n_features, 1) containing
            average feature values for each patch
        """
        # Extract all features
        _, feature_stack, _ = self.extract_all_features(image)
        
        height, width = image.shape[:2]
        n_patches_h = height // patch_size
        n_patches_w = width // patch_size
        n_features = len(self.config.feature_weights.keys())
        
        # Initialize patch features array
        patch_features = np.zeros((n_patches_h, n_patches_w, n_features), dtype=np.float32)
        
        # For each patch
        for h in range(n_patches_h):
            for w in range(n_patches_w):
                # Calculate patch boundaries
                y_start = h * patch_size
                y_end = min((h + 1) * patch_size, height)
                x_start = w * patch_size
                x_end = min((w + 1) * patch_size, width)
                
                # Extract average feature values for this patch
                for f in range(min(n_features, feature_stack.shape[2])):
                    patch_features[h, w, f] = np.mean(feature_stack[y_start:y_end, x_start:x_end, f])
        
        return patch_features