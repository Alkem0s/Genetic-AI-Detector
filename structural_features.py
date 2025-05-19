import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.transform import radon
from skimage.feature import graycoprops, graycomatrix

class StructuralFeatureExtractor:
    EPSILON = 1e-8
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
                    coef_var = patch.std() / (patch.mean() + self.EPSILON)
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
        feature_map = cv2.resize(perfection_map, (w, h), interpolation=cv2.INTER_LINEAR)

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
        global_profile /= (np.sum(global_profile) + self.EPSILON)

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
                    patch_profile /= (np.sum(patch_profile) + self.EPSILON)

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
                    contrast_ratio = patch_contrast / (global_contrast + self.EPSILON)

                    if 0.7 < contrast_ratio < 1.3:
                        score *= 0.7

                    edge_scores[i, j] = score

        # Create full-resolution feature map
        feature_map = cv2.resize(edge_scores, (w, h), interpolation=cv2.INTER_LINEAR)
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