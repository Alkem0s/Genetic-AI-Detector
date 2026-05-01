import tensorflow as tf
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from PIL import Image
import imagehash

import global_config as config
import utils
from base_features import BaseFeatureExtractor

class TextureFeatureExtractor(BaseFeatureExtractor):
    EPSILON_NP = 1e-8

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[config.patch_size, config.patch_size, 3], dtype=tf.float32)
    ])
    def _extract_noise_feature(self, image: tf.Tensor) -> tf.Tensor:
        with tf.name_scope('noise_feature'):
            gray = self._rgb_to_grayscale(image)

            laplacian_kernel = tf.reshape(
                tf.constant([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=tf.float32),
                [3, 3, 1, 1]
            )
            gray_expanded = tf.expand_dims(tf.expand_dims(gray, 0), -1)
            noise = tf.nn.conv2d(gray_expanded, laplacian_kernel,
                                 strides=[1, 1, 1, 1], padding='SAME')
            noise = tf.squeeze(tf.abs(noise))

            flat_noise            = tf.reshape(noise, [-1])
            global_noise_variance = tf.math.reduce_variance(flat_noise)
            max_noise_variance    = tf.constant(5000.0, dtype=tf.float32)
            normalized_variance   = tf.clip_by_value(
                global_noise_variance / (max_noise_variance + self.EPSILON_TF), 0.0, 1.0
            )
            noise_uniformity_score = 1.0 - normalized_variance
            feature_map = tf.fill(tf.shape(gray), noise_uniformity_score)
            return self._apply_mask(feature_map, image)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[config.patch_size, config.patch_size, 3], dtype=tf.float32)
    ])
    def _extract_texture_feature(self, image: tf.Tensor) -> tf.Tensor:
        with tf.name_scope('texture_feature'):
            gray = self._rgb_to_grayscale(image)
            
            shifts = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
            lbp = tf.zeros_like(gray, dtype=tf.float32)
            
            for i, (dy, dx) in enumerate(shifts):
                shifted = tf.roll(gray, shift=[dy, dx], axis=[0, 1])
                bit = tf.cast(shifted >= gray, tf.float32)
                lbp += bit * (2 ** i)
                
            max_val = tf.reduce_max(lbp)
            texture_map = tf.cond(
                max_val > 0,
                lambda: lbp / max_val,
                lambda: tf.zeros_like(lbp)
            )
            return self._apply_mask(texture_map, image)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[config.patch_size, config.patch_size, 3], dtype=tf.float32)
    ])
    def _extract_color_feature(self, image: tf.Tensor) -> tf.Tensor:
        with tf.name_scope('color_feature'):
            if tf.shape(image)[-1] != 3:
                return tf.zeros(tf.shape(image)[:2], dtype=tf.float32)
            image_rgb  = image[..., ::-1]
            hsv_image  = tf.image.rgb_to_hsv(image_rgb)
            saturation = hsv_image[..., 1]
            sat_variance = tf.math.reduce_variance(tf.reshape(saturation, [-1]))
            max_sat_var  = tf.constant(0.1, dtype=tf.float32)
            normalized   = tf.clip_by_value(
                sat_variance / (max_sat_var + self.EPSILON_TF), 0.0, 1.0
            )
            color_perfection_score = 1.0 - normalized
            feature_map = tf.fill(tf.shape(image)[:2], color_perfection_score)
            return self._apply_mask(feature_map, image)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[config.patch_size, config.patch_size, 3], dtype=tf.float32)
    ])
    def _extract_hash_feature(self, image: tf.Tensor) -> tf.Tensor:
        with tf.name_scope('hash_feature'):
            resized = tf.image.resize(image, [32, 32], method=tf.image.ResizeMethod.BILINEAR)
            gray = self._rgb_to_grayscale(resized)
            
            dct_rows = tf.signal.dct(gray, type=2, norm='ortho')
            dct_cols = tf.signal.dct(tf.transpose(dct_rows), type=2, norm='ortho')
            dct_2d = tf.transpose(dct_cols)
            
            dct_8x8 = dct_2d[:8, :8]
            mean_val = tf.reduce_mean(dct_8x8)
            hash_bits = tf.cast(dct_8x8 > mean_val, tf.float32)
            hash_score = tf.reduce_mean(hash_bits)
            
            h, w = tf.shape(image)[0], tf.shape(image)[1]
            feature_map = tf.fill([h, w], hash_score)
            return self._apply_mask(feature_map, image)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[config.patch_size, config.patch_size, 3], dtype=tf.float32)
    ])
    def _extract_channel_correlation_feature(self, image: tf.Tensor) -> tf.Tensor:
        """
        Step 24: Color channel cross-correlation feature.
        Computes cross-correlation between Red and Blue channels via FFT.
        Real cameras exhibit chromatic aberration → soft, broad correlation peak.
        AI images have perfectly aligned channels → sharp, narrow peak.
        High score = sharp peak = more AI-like.
        """
        with tf.name_scope('channel_correlation'):
            # Extract R and B channels (image is BGR-ordered after data loading)
            r_channel = image[:, :, 2]  # Red
            b_channel = image[:, :, 0]  # Blue

            # Cast to complex for FFT
            r_fft = tf.signal.rfft2d(r_channel)   # [H, W//2+1] complex
            b_fft = tf.signal.rfft2d(b_channel)

            # Cross-correlation: multiply R's FFT by conjugate of B's FFT, then IFFT
            cross_power = r_fft * tf.math.conj(b_fft)
            # Normalize to get the phase-only correlation (suppress amplitude differences)
            magnitude = tf.abs(cross_power)  # float32
            denom = tf.cast(magnitude + self.EPSILON_TF, tf.complex64)
            cross_power_norm = cross_power / denom
            correlation = tf.signal.irfft2d(cross_power_norm)  # [H, W]

            # Measure peak sharpness: ratio of max to mean
            corr_max  = tf.reduce_max(tf.abs(correlation))
            corr_mean = tf.reduce_mean(tf.abs(correlation))
            sharpness = corr_max / (corr_mean + self.EPSILON_TF)
            # Normalize against expected max sharpness; clamp to [0, 1]
            # A sharpness of 1 means flat → real; high sharpness → AI
            score = tf.clip_by_value((sharpness - 1.0) / 50.0, 0.0, 1.0)

            h, w = tf.shape(image)[0], tf.shape(image)[1]
            feature_map = tf.fill([h, w], score)
            return self._apply_mask(feature_map, image)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[config.patch_size, config.patch_size, 3], dtype=tf.float32)
    ])
    def _extract_glcm_feature(self, image: tf.Tensor) -> tf.Tensor:
        """
        Step 25: GLCM-based texture feature (TF-native approximation).
        Quantizes the grayscale patch to 16 levels, builds a co-occurrence matrix
        using unsorted_segment_sum on horizontal pixel-pair indices, then extracts
        contrast and homogeneity as a combined scalar score.
        AI images tend to have lower contrast and higher homogeneity (smoother textures).
        """
        with tf.name_scope('glcm_feature'):
            gray = self._rgb_to_grayscale(image)   # [H, W]

            # Quantize to 16 gray levels
            n_levels = 16
            gray_q = tf.cast(
                tf.clip_by_value(tf.floor(gray * n_levels), 0, n_levels - 1),
                tf.int32
            )  # [H, W]

            # Horizontal pixel-pair co-occurrences: (pixel[i,j], pixel[i,j+1])
            i_vals = tf.reshape(gray_q[:, :-1], [-1])   # left pixels
            j_vals = tf.reshape(gray_q[:, 1:], [-1])    # right pixels

            # Flatten 2D indices into 1D: idx = i * n_levels + j
            flat_idx = i_vals * n_levels + j_vals        # [N_pairs]
            ones     = tf.ones(tf.shape(flat_idx), dtype=tf.float32)

            # Build the GLCM by accumulating counts at each (i, j) index
            glcm_flat = tf.math.unsorted_segment_sum(
                ones, flat_idx, num_segments=n_levels * n_levels
            )  # [n_levels * n_levels]
            glcm = tf.reshape(glcm_flat, [n_levels, n_levels])

            # Normalize to a probability matrix
            glcm_sum = tf.reduce_sum(glcm) + self.EPSILON_TF
            glcm_p   = glcm / glcm_sum

            # Build difference index matrix |i - j|
            idx_range = tf.cast(tf.range(n_levels), tf.float32)
            ii, jj    = tf.meshgrid(idx_range, idx_range, indexing='ij')
            diff      = tf.abs(ii - jj)

            # Contrast: high values mean large gray-level differences → natural texture
            contrast    = tf.reduce_sum(glcm_p * diff ** 2)
            # Homogeneity: high values mean concentrated near diagonal → smooth texture
            homogeneity = tf.reduce_sum(glcm_p / (1.0 + diff))

            # AI images → low contrast, high homogeneity.
            # Score: weight homogeneity high, contrast low.
            # Normalize contrast (expected max ~100 for 16-level GLCM) and clamp.
            contrast_norm    = tf.clip_by_value(contrast / 100.0, 0.0, 1.0)
            homogeneity_norm = tf.clip_by_value(homogeneity, 0.0, 1.0)
            # High score = more AI-like = high homogeneity + low contrast
            glcm_score = (homogeneity_norm * 0.6) + ((1.0 - contrast_norm) * 0.4)

            h, w = tf.shape(image)[0], tf.shape(image)[1]
            feature_map = tf.fill([h, w], glcm_score)
            return self._apply_mask(feature_map, image)