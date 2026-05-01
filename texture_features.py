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
        tf.TensorSpec(shape=[None, config.patch_size, config.patch_size, 3], dtype=tf.float32)
    ])
    def _extract_noise_feature(self, image: tf.Tensor) -> tf.Tensor:
        with tf.name_scope('noise_feature'):
            gray = self._rgb_to_grayscale(image)

            laplacian_kernel = tf.reshape(
                tf.constant([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=tf.float32),
                [3, 3, 1, 1]
            )
            rank = tf.rank(gray)
            gray_expanded = tf.cond(
                tf.equal(rank, 2),
                lambda: tf.expand_dims(tf.expand_dims(gray, 0), -1),
                lambda: tf.expand_dims(gray, -1)
            )
            noise = tf.nn.conv2d(gray_expanded, laplacian_kernel,
                                 strides=[1, 1, 1, 1], padding='SAME')
            noise = tf.squeeze(tf.abs(noise), [3])
            noise = tf.cond(tf.equal(rank, 2), lambda: tf.squeeze(noise, [0]), lambda: noise)

            global_noise_variance = tf.cond(
                tf.equal(rank, 2),
                lambda: tf.math.reduce_variance(tf.reshape(noise, [-1])),
                lambda: tf.math.reduce_variance(tf.reshape(noise, [tf.shape(noise)[0], -1]), axis=1)
            )

            max_noise_variance    = tf.constant(5000.0, dtype=tf.float32)
            normalized_variance   = tf.clip_by_value(
                global_noise_variance / (max_noise_variance + self.EPSILON_TF), 0.0, 1.0
            )
            noise_uniformity_score = 1.0 - normalized_variance
            
            score = tf.reshape(noise_uniformity_score, [-1, 1, 1])
            feature_map = tf.ones_like(gray) * score
                
            return self._apply_mask(feature_map, image)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, config.patch_size, config.patch_size, 3], dtype=tf.float32)
    ])
    def _extract_texture_feature(self, image: tf.Tensor) -> tf.Tensor:
        with tf.name_scope('texture_feature'):
            gray = self._rgb_to_grayscale(image)
            
            shifts = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
            lbp = tf.zeros_like(gray, dtype=tf.float32)
            
            for i, (dy, dx) in enumerate(shifts):
                shifted = tf.roll(gray, shift=[dy, dx], axis=[-2, -1])
                bit = tf.cast(shifted >= gray, tf.float32)
                lbp += bit * (2 ** i)
                
            max_val = tf.reduce_max(lbp, axis=[-2, -1], keepdims=True)
            texture_map = tf.where(max_val > 0, lbp / max_val, tf.zeros_like(lbp))
            return self._apply_mask(texture_map, image)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, config.patch_size, config.patch_size, 3], dtype=tf.float32)
    ])
    def _extract_color_feature(self, image: tf.Tensor) -> tf.Tensor:
        with tf.name_scope('color_feature'):
            if tf.shape(image)[-1] != 3:
                return tf.zeros(tf.shape(image)[:-1], dtype=tf.float32)
            image_rgb  = image[..., ::-1]
            hsv_image  = tf.image.rgb_to_hsv(image_rgb)
            saturation = hsv_image[..., 1]
            
            sat_variance = tf.math.reduce_variance(saturation, axis=[-2, -1])
            max_sat_var  = tf.constant(0.1, dtype=tf.float32)
            normalized   = tf.clip_by_value(
                sat_variance / (max_sat_var + self.EPSILON_TF), 0.0, 1.0
            )
            color_perfection_score = 1.0 - normalized
            
            score = tf.reshape(color_perfection_score, [-1, 1, 1])
            feature_map = tf.ones_like(saturation) * score
            return self._apply_mask(feature_map, image)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, config.patch_size, config.patch_size, 3], dtype=tf.float32)
    ])
    def _extract_hash_feature(self, image: tf.Tensor) -> tf.Tensor:
        with tf.name_scope('hash_feature'):
            resized = tf.image.resize(image, [32, 32], method=tf.image.ResizeMethod.BILINEAR)
            gray = self._rgb_to_grayscale(resized)
            
            dct_rows = tf.signal.dct(gray, type=2, norm='ortho')
            
            rank = tf.rank(gray)
            trans_perm = tf.cond(tf.equal(rank, 2), lambda: tf.constant([1, 0]), lambda: tf.constant([0, 2, 1]))
            
            dct_cols = tf.signal.dct(tf.transpose(dct_rows, trans_perm), type=2, norm='ortho')
            dct_2d = tf.transpose(dct_cols, trans_perm)
            
            dct_8x8 = dct_2d[..., :8, :8]
            mean_val = tf.reduce_mean(dct_8x8, axis=[-2, -1], keepdims=True)
            hash_bits = tf.cast(dct_8x8 > mean_val, tf.float32)
            hash_score = tf.reduce_mean(hash_bits, axis=[-2, -1])
            
            score = tf.reshape(hash_score, [-1, 1, 1])
            feature_map = tf.ones_like(self._rgb_to_grayscale(image)) * score

            return self._apply_mask(feature_map, image)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, config.patch_size, config.patch_size, 3], dtype=tf.float32)
    ])
    def _extract_channel_correlation_feature(self, image: tf.Tensor) -> tf.Tensor:
        """
        Step 24: Color channel cross-correlation feature.
        """
        with tf.name_scope('channel_correlation'):
            r_channel = image[..., 2]  # Red
            b_channel = image[..., 0]  # Blue

            r_fft = tf.signal.rfft2d(r_channel)
            b_fft = tf.signal.rfft2d(b_channel)

            cross_power = r_fft * tf.math.conj(b_fft)
            magnitude = tf.abs(cross_power)
            denom = tf.cast(magnitude + self.EPSILON_TF, tf.complex64)
            cross_power_norm = cross_power / denom
            correlation = tf.signal.irfft2d(cross_power_norm)

            corr_max  = tf.reduce_max(tf.abs(correlation), axis=[-2, -1])
            corr_mean = tf.reduce_mean(tf.abs(correlation), axis=[-2, -1])
            sharpness = corr_max / (corr_mean + self.EPSILON_TF)
            
            score = tf.clip_by_value((sharpness - 1.0) / 50.0, 0.0, 1.0)

            score = tf.reshape(score, [-1, 1, 1])
            feature_map = tf.ones_like(r_channel) * score
            return self._apply_mask(feature_map, image)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, config.patch_size, config.patch_size, 3], dtype=tf.float32)
    ])
    def _extract_glcm_feature(self, image: tf.Tensor) -> tf.Tensor:
        """
        Step 25: GLCM-based texture feature (TF-native approximation).
        """
        with tf.name_scope('glcm_feature'):
            gray = self._rgb_to_grayscale(image)

            n_levels = 16
            gray_q = tf.cast(
                tf.clip_by_value(tf.floor(gray * n_levels), 0, n_levels - 1),
                tf.int32
            )

            rank = tf.rank(gray)
            n_batch = tf.cond(tf.equal(rank, 2), lambda: 1, lambda: tf.shape(gray)[0])
            
            gray_q_3d = tf.cond(tf.equal(rank, 2), lambda: tf.expand_dims(gray_q, 0), lambda: gray_q)
            
            i_vals = tf.reshape(gray_q_3d[:, :, :-1], [n_batch, -1])
            j_vals = tf.reshape(gray_q_3d[:, :, 1:], [n_batch, -1])

            idx = i_vals * n_levels + j_vals
            batch_offsets = tf.expand_dims(tf.range(n_batch) * (n_levels * n_levels), 1)
            flat_idx = tf.reshape(idx + batch_offsets, [-1])
            
            num_segments = n_batch * n_levels * n_levels
            ones = tf.ones(tf.shape(flat_idx), dtype=tf.float32)
            
            glcm_flat = tf.math.unsorted_segment_sum(ones, flat_idx, num_segments)
            glcm = tf.reshape(glcm_flat, [n_batch, n_levels, n_levels])
            
            glcm_sum = tf.reduce_sum(glcm, axis=[1, 2], keepdims=True) + self.EPSILON_TF
            glcm_p = glcm / glcm_sum

            idx_range = tf.cast(tf.range(n_levels), tf.float32)
            ii, jj = tf.meshgrid(idx_range, idx_range, indexing='ij')
            diff = tf.abs(ii - jj)

            contrast = tf.reduce_sum(glcm_p * diff ** 2, axis=[1, 2])
            homogeneity = tf.reduce_sum(glcm_p / (1.0 + diff), axis=[1, 2])

            contrast_norm = tf.clip_by_value(contrast / 100.0, 0.0, 1.0)
            homogeneity_norm = tf.clip_by_value(homogeneity, 0.0, 1.0)
            glcm_score = (homogeneity_norm * 0.6) + ((1.0 - contrast_norm) * 0.4)

            glcm_score = tf.cond(tf.equal(rank, 2), lambda: tf.squeeze(glcm_score, [0]), lambda: glcm_score)

            score = tf.reshape(glcm_score, [-1, 1, 1])
            feature_map = tf.ones_like(gray) * score
                
            return self._apply_mask(feature_map, image)