import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.metrics import structural_similarity as ssim
from skimage.transform import radon

from utils import get_edge_detection, is_natural_scene

# Try to import numba for JIT compilation
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# Define JIT-compiled functions if numba is available
if NUMBA_AVAILABLE:
    @numba.jit(nopython=True)
    def analyze_patch_numba(patch, mean_global, std_global):
        """JIT-compiled patch analysis"""
        h, w = patch.shape
        std_dev = 0.0
        sum_val = 0.0
        sum_sq = 0.0
        
        for i in range(h):
            for j in range(w):
                val = patch[i, j]
                sum_val += val
                sum_sq += val * val
        
        mean = sum_val / (h * w)
        variance = (sum_sq / (h * w)) - (mean * mean)
        std_dev = variance ** 0.5
        
        return std_dev, mean, variance / max(std_global, 1e-5)

class StructuralFeaturesExtractor:
    @staticmethod
    def extract_gradient_perfection(image, threshold=0.8, patch_size=16):
        """
        Detect unnaturally perfect gradients in images
        AI-generated images often have unnaturally smooth gradients
        """
        # Convert to grayscale for gradient analysis if it's not already
        if len(image.shape) > 2 and image.shape[2] > 1:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        # Use Scharr operators for better gradient detection
        scharrx = get_edge_detection(gray, method='scharr', dx=1, dy=0)
        scharry = get_edge_detection(gray, method='scharr', dx=0, dy=1)
        
        # Calculate gradient magnitude and direction
        magnitude = np.sqrt(scharrx**2 + scharry**2)
        direction = np.arctan2(scharry, scharrx)
        
        # Find regions with near-constant gradient change (too perfect for natural images)
        # Split image into patches
        h, w = gray.shape
        perfection_map = np.zeros((h // patch_size, w // patch_size))
        
        # Add FFT-based analysis of gradient directions to detect natural vs artificial patterns
        direction_coherence = np.zeros((h // patch_size, w // patch_size))
        
        for i in range(0, h - patch_size, patch_size):
            for j in range(0, w - patch_size, patch_size):
                patch = magnitude[i:i+patch_size, j:j+patch_size]
                dir_patch = direction[i:i+patch_size, j:j+patch_size]
                
                if patch.std() > 0:  # Avoid division by zero
                    # Calculate coefficient of variation (lower means more uniform gradient)
                    coef_var = patch.std() / patch.mean() if patch.mean() > 0 else float('inf')
                    
                    # Simplified direction coherence using FFT
                    # Convert direction to complex numbers and compute FFT
                    complex_dir = np.exp(1j * dir_patch)
                    fft_dir = np.fft.fft2(complex_dir)
                    power_spectrum = np.abs(fft_dir)**2
                    
                    # Measure spectral concentration - high values indicate coherent directions
                    total_power = np.sum(power_spectrum)
                    center_power = np.sum(power_spectrum[:3, :3]) if power_spectrum.size > 9 else total_power
                    spectral_concentration = center_power / total_power if total_power > 0 else 0
                    
                    direction_coherence[i // patch_size, j // patch_size] = spectral_concentration
                    
                    # Combine magnitude consistency with direction analysis
                    perfection_score = 1 - min(coef_var, 1.0)
                    
                    # Weight perfection score by direction coherence
                    perfection_map[i // patch_size, j // patch_size] = perfection_score * (0.5 + 0.5 * direction_coherence[i // patch_size, j // patch_size])
        
        # Additional step: check if overall image has many high perfection areas
        # If too many patches have high scores, likely a false positive on a smooth natural image
        high_perfection_ratio = np.sum(perfection_map > 0.7) / perfection_map.size
        if high_perfection_ratio > 0.5:  # If more than half the image has perfect gradients
            # Likely a naturally smooth image (sky, water, etc.) - reduce all scores
            perfection_map = perfection_map * (0.5 + high_perfection_ratio * 0.5)
        
        # Highlight patches with perfection scores above threshold
        highlighted_patches = perfection_map > threshold
        
        # Create feature map same size as original image
        feature_map = np.zeros((h, w), dtype=np.uint8)
        for i in range(len(highlighted_patches)):
            for j in range(len(highlighted_patches[0])):
                if highlighted_patches[i, j]:
                    feature_map[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = 255
        
        return feature_map, perfection_map
    
    @staticmethod
    def detect_unnatural_patterns(image, threshold=0.7):
        """
        Detect patterns that are too regular/symmetrical
        AI-generated images often have unnaturally regular patterns in frequency domain
        """
        if len(image.shape) > 2 and image.shape[2] > 1:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        # Apply FFT to find frequency domain patterns
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        
        # Normalize to 0-1
        magnitude_spectrum_norm = (magnitude_spectrum - magnitude_spectrum.min()) / \
                                 (magnitude_spectrum.max() - magnitude_spectrum.min())
        
        # Look for strong periodic patterns (peaks in frequency domain)
        # Improved peak detection with adaptive thresholding
        # First get the average and std dev of spectrum
        mean_spectrum = np.mean(magnitude_spectrum_norm)
        std_spectrum = np.std(magnitude_spectrum_norm)
        
        # Adaptive threshold based on image statistics
        adaptive_threshold = mean_spectrum + (threshold * std_spectrum)
        peaks = magnitude_spectrum_norm > adaptive_threshold
        
        # Filter out DC component (center peak which appears in all images)
        h, w = magnitude_spectrum_norm.shape
        center_mask = np.ones_like(peaks)
        center_h, center_w = h // 2, w // 2
        center_radius = min(h, w) // 20  # Adjust radius as needed
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center_w)**2 + (Y - center_h)**2)
        center_mask[dist_from_center <= center_radius] = 0
        peaks = peaks & center_mask
        
        # Count peaks and their distribution
        peak_count = np.sum(peaks)
        
        # Simplified peak pattern analysis using radial density estimation
        peak_pattern_score = 0
        
        if peak_count > 5:  # Need minimum peaks for analysis
            # Calculate peak positions
            peak_positions = np.where(peaks)
            
            # Calculate radial density profile
            peak_distances = np.sqrt((peak_positions[0] - center_h)**2 + (peak_positions[1] - center_w)**2)
            
            # Create radial histogram
            max_dist = np.sqrt(center_h**2 + center_w**2)
            bins = 20  # Number of radial bins
            radial_hist, _ = np.histogram(peak_distances, bins=bins, range=(0, max_dist))
            radial_hist = radial_hist / np.sum(radial_hist) if np.sum(radial_hist) > 0 else radial_hist
            
            # Calculate entropy of radial distribution
            radial_entropy = -np.sum(radial_hist * np.log2(radial_hist + 1e-10))
            max_entropy = np.log2(bins)
            normalized_entropy = radial_entropy / max_entropy if max_entropy > 0 else 0
            
            # Low entropy indicates peaks concentrated at specific distances (artificial)
            peak_pattern_score = 1 - normalized_entropy
            
            # Calculate angular distribution of peaks
            angles = np.arctan2(peak_positions[0] - center_h, peak_positions[1] - center_w)
            angle_hist, _ = np.histogram(angles, bins=16, range=(-np.pi, np.pi))
            angle_hist = angle_hist / np.sum(angle_hist) if np.sum(angle_hist) > 0 else angle_hist
            
            # Calculate angular entropy
            angle_entropy = -np.sum(angle_hist * np.log2(angle_hist + 1e-10))
            max_angle_entropy = np.log2(16)
            normalized_angle_entropy = angle_entropy / max_angle_entropy if max_angle_entropy > 0 else 0
            
            # Combine with radial score - high angle entropy (varied angles) reduces score
            peak_pattern_score = peak_pattern_score * (1 - 0.5 * normalized_angle_entropy)
        
        # Create a simplified feature map highlighting regions with periodic patterns
        h, w = gray.shape
        feature_map = np.zeros((h, w), dtype=np.uint8)
        
        # If we have strong periodic patterns
        if peak_count > 10 and peak_pattern_score > 0.6:
            # Simplified feature map based on detected pattern strength
            feature_map = np.ones((h, w), dtype=np.uint8) * int(255 * peak_pattern_score)
        
        return feature_map, peak_pattern_score

    @staticmethod
    def analyze_edge_consistency(image, threshold=0.65, patch_size=16):
        """
        Analyze edge consistency - AI images often have too perfect or too regular edges
        """
        if len(image.shape) > 2 and image.shape[2] > 1:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        # Apply multi-scale edge detection using pyramid approach
        pyramid_levels = 3
        edge_pyramids = []
        
        # Generate image pyramid
        curr_img = gray.copy()
        for i in range(pyramid_levels):
            # Apply edge detection at current scale
            edges = get_edge_detection(curr_img, method='canny', low=100, high=200)
            edge_pyramids.append(edges)
            
            # Downscale for next level
            if i < pyramid_levels - 1:
                curr_img = cv2.pyrDown(curr_img)
        
        # Combine edge maps (resize and merge)
        multi_scale_edges = np.zeros_like(gray, dtype=np.uint8)
        for i, edges in enumerate(edge_pyramids):
            if i > 0:
                # Resize to original size
                resized_edges = cv2.resize(edges, (gray.shape[1], gray.shape[0]))
                multi_scale_edges = cv2.bitwise_or(multi_scale_edges, resized_edges)
            else:
                multi_scale_edges = edges
        
        # Analyze edge statistics in patches
        h, w = gray.shape
        edge_consistency_map = np.zeros((h // patch_size, w // patch_size))
        feature_map = np.zeros((h, w), dtype=np.uint8)
        
        # Extract edge orientation histogram for entire image (reference)
        # Use Radon transform for orientation analysis
        theta = np.linspace(0, 180, 36)
        radon_global = radon(multi_scale_edges, theta=theta, circle=True)
        global_orientation_profile = np.sum(radon_global, axis=0)
        global_orientation_profile = global_orientation_profile / np.sum(global_orientation_profile) if np.sum(global_orientation_profile) > 0 else global_orientation_profile
        
        # Calculate global edge statistics
        global_edge_count = np.sum(multi_scale_edges > 0)
        global_edge_density = global_edge_count / (h * w)
        
        # Use GLCM texture features to analyze edge patterns
        glcm = graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
        global_contrast = np.mean(graycoprops(glcm, 'contrast'))
        
        for i in range(0, h - patch_size, patch_size):
            for j in range(0, w - patch_size, patch_size):
                edge_patch = multi_scale_edges[i:i+patch_size, j:j+patch_size]
                gray_patch = gray[i:i+patch_size, j:j+patch_size]
                
                # Count edges in patch
                edge_count = np.sum(edge_patch > 0)
                edge_density = edge_count / (patch_size * patch_size)
                
                # Skip patches with very few edges - can't reliably analyze
                if edge_count > patch_size / 2:
                    # Use Radon transform for orientation analysis (more robust than gradient)
                    radon_patch = radon(edge_patch, theta=theta, circle=True)
                    patch_orientation_profile = np.sum(radon_patch, axis=0)
                    patch_orientation_profile = patch_orientation_profile / np.sum(patch_orientation_profile) if np.sum(patch_orientation_profile) > 0 else patch_orientation_profile
                    
                    # Calculate entropy of orientation distribution
                    orientation_entropy = -np.sum(patch_orientation_profile * np.log2(patch_orientation_profile + 1e-10))
                    max_entropy = np.log2(len(theta))
                    normalized_entropy = orientation_entropy / max_entropy if max_entropy > 0 else 0
                    
                    # Low entropy means edges have very similar orientations (unnatural)
                    orientation_score = 1 - normalized_entropy  # Higher score is more suspicious
                        
                    # Calculate the similarity between patch and global orientation profiles
                    # (Use correlation coefficient)
                    correlation = np.corrcoef(patch_orientation_profile, global_orientation_profile)[0, 1]
                    correlation_score = 0.5 * (1 - abs(correlation))  # Different from global pattern is suspicious
                    
                    # Final score combining orientation entropy and global comparison
                    final_score = orientation_score * 0.7 + correlation_score * 0.3
                    
                    # Content-aware adjustment
                    glcm_patch = graycomatrix(gray_patch, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
                    patch_contrast = np.mean(graycoprops(glcm_patch, 'contrast'))
                    
                    # If contrast similar to global, likely natural content
                    contrast_ratio = patch_contrast / global_contrast if global_contrast > 0 else 1
                    if 0.7 < contrast_ratio < 1.3:
                        final_score *= 0.7  # Reduce score - likely natural content
                        
                    if final_score > threshold:
                        edge_consistency_map[i // patch_size, j // patch_size] = final_score
                        feature_map[i:i+patch_size, j:j+patch_size] = int(255 * final_score)
        
        return feature_map, edge_consistency_map

    @staticmethod
    def detect_symmetry(image, threshold=0.8):
        """
        Detect unnatural symmetry in images
        AI-generated images sometimes have unnaturally high symmetry
        
        Optimized to:
        1. Use just 2 scales (1.0 and 0.5)
        2. Remove face detection
        3. Simplify scene classification
        """
        if len(image.shape) > 2 and image.shape[2] > 1:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            # Keep color for content analysis
            color_img = (image * 255).astype(np.uint8)
        else:
            gray = (image * 255).astype(np.uint8)
            color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        h, w = gray.shape
        feature_map = np.zeros((h, w), dtype=np.uint8)
        
        # Multi-scale symmetry analysis (check at 2 scales)
        symmetry_scores = []
        scale_weights = [0.7, 0.3]  # Weights for different scales
        scales = [1.0, 0.5]  # 100%, 50% scales
        
        for scale_idx, scale in enumerate(scales):
            # Resize image for multi-scale analysis
            if scale != 1.0:
                scaled_h, scaled_w = int(h * scale), int(w * scale)
                scaled_img = cv2.resize(gray, (scaled_w, scaled_h))
            else:
                scaled_img = gray
                scaled_h, scaled_w = h, w
            
            # Check horizontal symmetry
            left_half = scaled_img[:, :scaled_w//2]
            right_half = np.fliplr(scaled_img[:, scaled_w//2:])
            # Resize if needed
            if left_half.shape != right_half.shape:
                min_w = min(left_half.shape[1], right_half.shape[1])
                left_half = left_half[:, :min_w]
                right_half = right_half[:, :min_w]
            
            # Check vertical symmetry
            top_half = scaled_img[:scaled_h//2, :]
            bottom_half = np.flipud(scaled_img[scaled_h//2:, :])
            # Resize if needed
            if top_half.shape != bottom_half.shape:
                min_h = min(top_half.shape[0], bottom_half.shape[0])
                top_half = top_half[:min_h, :]
                bottom_half = bottom_half[:min_h, :]
            
            # Use SSIM for more perceptually meaningful symmetry comparison
            # This better matches how humans perceive symmetry
            
            # Calculate SSIM-based symmetry scores
            h_symmetry_ssim = ssim(left_half, right_half, data_range=255)
            v_symmetry_ssim = ssim(top_half, bottom_half, data_range=255)
            
            # Also calculate pixel-wise differences for fine details
            h_diff = np.abs(left_half.astype(float) - right_half.astype(float))
            v_diff = np.abs(top_half.astype(float) - bottom_half.astype(float))
            
            h_symmetry_pixel = 1 - np.mean(h_diff) / 255
            v_symmetry_pixel = 1 - np.mean(v_diff) / 255
            
            # Combine SSIM and pixel-based symmetry (SSIM is more important)
            h_symmetry = 0.7 * h_symmetry_ssim + 0.3 * h_symmetry_pixel
            v_symmetry = 0.7 * v_symmetry_ssim + 0.3 * v_symmetry_pixel
            
            # Add to multi-scale scores with appropriate weight
            symmetry_scores.append((max(h_symmetry, v_symmetry), h_symmetry > v_symmetry, scale_weights[scale_idx]))
        
        # Combine multi-scale symmetry scores
        combined_symmetry = sum(score * weight for score, _, weight in symmetry_scores) / sum(scale_weights)
        primary_axis = any(is_horizontal for _, is_horizontal, _ in symmetry_scores if is_horizontal)
        
        # Edge density analysis to detect natural vs artificial symmetry
        edges = get_edge_detection(gray, method='canny', low=100, high=200)
        edge_density = np.sum(edges > 0) / (h * w)
        
        # Calculate edge symmetry - natural objects have less perfect edge symmetry
        left_edges = edges[:, :w//2]
        right_edges = np.fliplr(edges[:, w//2:])
        top_edges = edges[:h//2, :]
        bottom_edges = np.flipud(edges[h//2:, :])
        
        # Adjust sizes if needed
        if left_edges.shape != right_edges.shape:
            min_w = min(left_edges.shape[1], right_edges.shape[1])
            left_edges = left_edges[:, :min_w]
            right_edges = right_edges[:, :min_w]
            
        if top_edges.shape != bottom_edges.shape:
            min_h = min(top_edges.shape[0], bottom_edges.shape[0])
            top_edges = top_edges[:min_h, :]
            bottom_edges = bottom_edges[:min_h, :]
        
        h_edge_symmetry = np.sum(left_edges == right_edges) / left_edges.size
        v_edge_symmetry = np.sum(top_edges == bottom_edges) / top_edges.size
        edge_symmetry = max(h_edge_symmetry, v_edge_symmetry)
        
        # Use is_natural_scene function from utils.py
        scene_info = is_natural_scene(image)
        is_natural_symmetrical = (scene_info['is_sky'] or scene_info['is_water'] or scene_info['is_architectural'])
        
        # Calculate final symmetry score, accounting for natural symmetry
        symmetry_score = combined_symmetry
        
        # Reduce score for naturally symmetrical content
        if is_natural_symmetrical:
            # Natural symmetry adjustment factor
            if scene_info['is_sky'] or scene_info['is_water']:
                # Natural scenes with sky/water tend to be somewhat symmetrical
                natural_reduction = 0.25
            elif scene_info['is_architectural'] and edge_symmetry > 0.7:
                # Architectural scenes often have symmetry
                natural_reduction = 0.3
            else:
                natural_reduction = 0.2
                
            symmetry_score = max(0, symmetry_score - natural_reduction)
        
        # Additional check: AI-generated images often have *perfect* symmetry
        # Natural objects rarely have perfect symmetry even when they're symmetrical
        if edge_symmetry > 0.9 and combined_symmetry > 0.9 and not is_natural_symmetrical:
            # This is unnaturally perfect symmetry - likely AI
            symmetry_score = max(symmetry_score, 0.85)
        
        # If symmetry is above threshold, highlight the image
        if symmetry_score > threshold:
            # Create gradient that highlights the axes of symmetry
            if primary_axis:  # Horizontal primary axis
                # Highlight horizontal axis
                mid_y = h // 2
                for y in range(h):
                    weight = 1 - min(abs(y - mid_y) / (h/4), 1.0)
                    intensity = int(255 * weight * symmetry_score)
                    feature_map[y, :] = np.maximum(feature_map[y, :], intensity)
            else:  # Vertical primary axis
                # Highlight vertical axis
                mid_x = w // 2
                for x in range(w):
                    weight = 1 - min(abs(x - mid_x) / (w/4), 1.0)
                    intensity = int(255 * weight * symmetry_score)
                    feature_map[:, x] = np.maximum(feature_map[:, x], intensity)
        
        return feature_map, symmetry_score