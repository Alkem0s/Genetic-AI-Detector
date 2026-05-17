import tensorflow as tf
import numpy as np
from typing import Tuple
import global_config as config
import utils
from base_features import BaseFeatureExtractor

class StructuralFeatureExtractor(BaseFeatureExtractor):

    SOBEL_X_KERNEL = tf.constant(
        [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
        dtype=tf.float32, shape=[3, 3, 1, 1]
    )
    SOBEL_Y_KERNEL = tf.constant(
        [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
        dtype=tf.float32, shape=[3, 3, 1, 1]
    )
    GRADIENT_MAX_VAL = tf.constant(4.0 * 255.0, dtype=tf.float32)
    FEATURE_MAX_VAL  = tf.constant(100.0, dtype=tf.float32)
    GAUSSIAN_KERNEL  = tf.reshape(
        tf.constant([
            [1., 4., 7., 4., 1.],
            [4., 16., 26., 16., 4.],
            [7., 26., 41., 26., 7.],
            [4., 16., 26., 16., 4.],
            [1., 4., 7., 4., 1.]
        ], dtype=tf.float32) / 273.0,
        [5, 5, 1, 1]
    )
    # Comb filter for periodic peak detection in FFT magnitude spectrum.
    # A 5x5 kernel with peaks at (0,0) and (0,2) detects grid-aligned periodicity.
    COMB_KERNEL = tf.reshape(
        tf.constant([
            [1., 0., 1., 0., 1.],
            [0., 0., 0., 0., 0.],
            [1., 0., 0., 0., 1.],
            [0., 0., 0., 0., 0.],
            [1., 0., 1., 0., 1.],
        ], dtype=tf.float32) / 8.0,
        [5, 5, 1, 1]
    )

    @staticmethod
    @tf.function
    def _apply_sobel_filters(gray: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        rank = tf.rank(gray)
        gray_expanded = tf.cond(
            tf.equal(rank, 2),
            lambda: tf.expand_dims(tf.expand_dims(gray, 0), -1),
            lambda: tf.expand_dims(gray, -1)
        )
        grad_x = tf.nn.conv2d(gray_expanded, StructuralFeatureExtractor.SOBEL_X_KERNEL,
                              strides=[1, 1, 1, 1], padding='SAME')
        grad_y = tf.nn.conv2d(gray_expanded, StructuralFeatureExtractor.SOBEL_Y_KERNEL,
                              strides=[1, 1, 1, 1], padding='SAME')
        grad_x = tf.cond(tf.equal(rank, 2), lambda: tf.squeeze(grad_x, [0, 3]), lambda: tf.squeeze(grad_x, [3]))
        grad_y = tf.cond(tf.equal(rank, 2), lambda: tf.squeeze(grad_y, [0, 3]), lambda: tf.squeeze(grad_y, [3]))
        return grad_x, grad_y

    @staticmethod
    @tf.function
    def _tf_native_canny(gray, low_threshold=50.0, high_threshold=150.0):
        rank = tf.rank(gray)
        gray_expanded = tf.cond(
            tf.equal(rank, 2),
            lambda: tf.expand_dims(tf.expand_dims(gray, 0), -1),
            lambda: tf.expand_dims(gray, -1)
        )
        blurred = tf.nn.conv2d(gray_expanded, StructuralFeatureExtractor.GAUSSIAN_KERNEL,
                               strides=[1, 1, 1, 1], padding='SAME')
        blurred = tf.cond(tf.equal(rank, 2), lambda: tf.squeeze(blurred, [0, 3]), lambda: tf.squeeze(blurred, [3]))
        grad_x, grad_y = StructuralFeatureExtractor._apply_sobel_filters(blurred)
        magnitude = tf.sqrt(grad_x**2 + grad_y**2 + BaseFeatureExtractor.EPSILON_TF)
        
        magnitude_padded = tf.cond(
            tf.equal(rank, 2),
            lambda: tf.pad(magnitude, [[1, 1], [1, 1]], mode='REFLECT'),
            lambda: tf.pad(magnitude, [[0, 0], [1, 1], [1, 1]], mode='REFLECT')
        )
        
        max_pooled = tf.cond(
            tf.equal(rank, 2),
            lambda: tf.squeeze(tf.nn.max_pool2d(
                tf.expand_dims(tf.expand_dims(magnitude_padded, 0), -1),
                ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID'
            ), [0, 3]),
            lambda: tf.squeeze(tf.nn.max_pool2d(
                tf.expand_dims(magnitude_padded, -1),
                ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID'
            ), [3])
        )
            
        suppressed   = tf.where(tf.equal(magnitude, max_pooled), magnitude, 0.0)
        strong_edges = tf.cast(suppressed > high_threshold, tf.float32)
        weak_edges   = tf.cast(
            tf.logical_and(suppressed > low_threshold, suppressed <= high_threshold),
            tf.float32
        ) * 0.5
        return tf.clip_by_value(strong_edges + weak_edges, 0.0, 1.0)

    @staticmethod
    @tf.function
    def _create_radial_gradient_map(h, w, score):
        center_y = tf.cast(h, tf.float32) / 2.0
        center_x = tf.cast(w, tf.float32) / 2.0
        yy, xx = tf.meshgrid(
            tf.range(tf.cast(h, tf.int32), dtype=tf.float32),
            tf.range(tf.cast(w, tf.int32), dtype=tf.float32),
            indexing='ij'
        )
        distance    = tf.sqrt((yy - center_y)**2 + (xx - center_x)**2)
        max_distance = tf.sqrt(center_y**2 + center_x**2) + BaseFeatureExtractor.EPSILON_TF
        gradient    = 1.0 - tf.minimum(distance / max_distance, 1.0)
        
        # Broadcast score if batched
        rank = tf.rank(score)
        score = tf.cond(
            tf.equal(rank, 0),
            lambda: score,
            lambda: tf.reshape(score, [-1, 1, 1])
        )
            
        return gradient * score

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, config.patch_size, config.patch_size, 3], dtype=tf.float32)
    ])
    def _extract_gradient_feature(self, image: tf.Tensor) -> tf.Tensor:
        with tf.name_scope('gradient_feature'):
            gray = tf.cast(self._rgb_to_grayscale(image), tf.float32)
            grad_x, grad_y = self._apply_sobel_filters(gray)
            magnitude = tf.sqrt(grad_x**2 + grad_y**2 + self.EPSILON_TF)
            magnitude_normalized = tf.clip_by_value(
                magnitude / (self.GRADIENT_MAX_VAL + self.EPSILON_TF), 0.0, 1.0
            )
            mean_mag  = tf.reduce_mean(magnitude_normalized, axis=[-2, -1], keepdims=True)
            variance  = tf.reduce_mean(tf.square(magnitude_normalized - mean_mag), axis=[-2, -1])
            perfection_score = 1.0 / (variance + self.EPSILON_TF)
            
            h, w = tf.shape(image)[-3], tf.shape(image)[-2]
            feature_map = self._create_radial_gradient_map(
                tf.cast(h, tf.float32), tf.cast(w, tf.float32), perfection_score
            )
            feature_map = tf.clip_by_value(feature_map / self.FEATURE_MAX_VAL, 0.0, 1.0)
            return self._apply_mask(feature_map, image)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, config.patch_size, config.patch_size, 3], dtype=tf.float32)
    ])
    def _extract_pattern_feature(self, image: tf.Tensor) -> tf.Tensor:
        """
        Step 22: Enhanced Fourier analysis with:
        - Original high/low freq energy ratio (magnitude score)
        - Phase consistency score: low variance in phase → AI image
        - Periodic peak detection via comb filter convolution on magnitude spectrum
        """
        with tf.name_scope('pattern_feature'):
            gray = tf.cast(self._rgb_to_grayscale(image), tf.float32)
            f_transform = tf.signal.fft2d(tf.cast(gray, tf.complex64))
            f_shift     = tf.signal.fftshift(f_transform)

            # --- Original magnitude ratio score ---
            magnitude_spectrum = tf.math.log(tf.abs(f_shift) + self.EPSILON_TF)
            h, w = tf.shape(magnitude_spectrum)[-2], tf.shape(magnitude_spectrum)[-1]
            center_h, center_w = h // 2, w // 2
            yy, xx = tf.meshgrid(
                tf.range(h, dtype=tf.float32),
                tf.range(w, dtype=tf.float32),
                indexing='ij'
            )
            dist_from_center = tf.sqrt(
                (yy - tf.cast(center_h, tf.float32))**2 +
                (xx - tf.cast(center_w, tf.float32))**2
            )
            max_dist        = tf.sqrt(tf.cast(center_h**2 + center_w**2, tf.float32))
            high_freq_mask  = tf.cast(dist_from_center > max_dist * 0.3, tf.float32)
            low_freq_mask   = 1.0 - high_freq_mask
            high_freq_energy = tf.reduce_sum(magnitude_spectrum * high_freq_mask, axis=[-2, -1])
            low_freq_energy  = tf.reduce_sum(magnitude_spectrum * low_freq_mask, axis=[-2, -1])
            magnitude_score  = high_freq_energy / (low_freq_energy + self.EPSILON_TF)

            # --- Phase consistency score ---
            phase = tf.math.angle(f_shift)  # shape [H, W], values in [-pi, pi]
            phase_variance = tf.math.reduce_variance(phase, axis=[-2, -1])
            phase_score = 1.0 / (phase_variance + self.EPSILON_TF)
            phase_score = tf.clip_by_value(phase_score / self.FEATURE_MAX_VAL, 0.0, 1.0)

            # --- Periodic peak detector via comb filter ---
            rank = tf.rank(magnitude_spectrum)
            mag_exp = tf.cond(
                tf.equal(rank, 2),
                lambda: tf.expand_dims(tf.expand_dims(magnitude_spectrum, 0), -1),
                lambda: tf.expand_dims(magnitude_spectrum, -1)
            )
            comb_response = tf.nn.conv2d(
                mag_exp, self.COMB_KERNEL, strides=[1, 1, 1, 1], padding='SAME'
            )
            comb_response = tf.cond(tf.equal(rank, 2), lambda: tf.squeeze(comb_response, [0, 3]), lambda: tf.squeeze(comb_response, [3]))
            
            comb_mean   = tf.reduce_mean(comb_response, axis=[-2, -1])
            comb_max    = tf.reduce_max(comb_response, axis=[-2, -1])
            comb_score  = (comb_max - comb_mean) / (comb_mean + self.EPSILON_TF)
            comb_score  = tf.clip_by_value(comb_score / self.FEATURE_MAX_VAL, 0.0, 1.0)

            pattern_score = (magnitude_score / self.FEATURE_MAX_VAL) * 0.4 \
                            + phase_score * 0.3 \
                            + comb_score * 0.3
            pattern_score = tf.clip_by_value(pattern_score, 0.0, 1.0)

            ih, iw = tf.shape(image)[-3], tf.shape(image)[-2]
            feature_map = self._create_radial_gradient_map(
                tf.cast(ih, tf.float32), tf.cast(iw, tf.float32), pattern_score
            )
            feature_map = tf.clip_by_value(feature_map, 0.0, 1.0)
            return self._apply_mask(feature_map, image)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, config.patch_size, config.patch_size, 3], dtype=tf.float32)
    ])
    def _extract_noise_spectrum_feature(self, image: tf.Tensor) -> tf.Tensor:
        """
        Spectral Peak Detection: Detects periodic upsampling artifacts (checkerboard patterns)
        that AI generators leave in the Fourier domain due to transposed convolution operations.

        Method:
            1. Compute the log-magnitude of the FFT and suppress the DC region.
            2. Identify local spectral peaks via max-pooling and compare to the local mean.
            3. A high peak-to-average ratio is a strong indicator of artificial periodicity.
        """
        with tf.name_scope('noise_spectrum_feature'):
            gray = tf.cast(self._rgb_to_grayscale(image), tf.float32)
            f_transform = tf.signal.fft2d(tf.cast(gray, tf.complex64))
            f_shift     = tf.signal.fftshift(f_transform)

            magnitude_spectrum = tf.math.log(tf.abs(f_shift) + self.EPSILON_TF)

            # --- Suppress the DC component (center region) ---
            h, w = tf.shape(magnitude_spectrum)[-2], tf.shape(magnitude_spectrum)[-1]
            center_h, center_w = h // 2, w // 2
            yy, xx = tf.meshgrid(
                tf.range(h, dtype=tf.float32),
                tf.range(w, dtype=tf.float32),
                indexing='ij'
            )
            dc_radius  = tf.cast(tf.minimum(h, w), tf.float32) * 0.1
            dc_mask    = tf.cast(
                tf.sqrt((yy - tf.cast(center_h, tf.float32))**2 +
                        (xx - tf.cast(center_w, tf.float32))**2) > dc_radius,
                tf.float32
            )
            # dc_mask shape: [H, W]; broadcast over batch
            mag_no_dc = magnitude_spectrum * dc_mask

            # --- Detect local spectral peaks ---
            # Expand to 4D for max_pool2d: [batch, H, W, 1]
            rank = tf.rank(mag_no_dc)
            mag_4d = tf.cond(
                tf.equal(rank, 2),
                lambda: tf.expand_dims(tf.expand_dims(mag_no_dc, 0), -1),
                lambda: tf.expand_dims(mag_no_dc, -1)
            )
            # Local max over 3x3 neighbourhood
            local_max  = tf.nn.max_pool2d(mag_4d,  ksize=3, strides=1, padding='SAME')
            # Local mean via Conv2D (more robust than AvgPool2D on small tensors in some cuDNN versions)
            mean_kernel = tf.ones((3, 3, 1, 1), dtype=tf.float32) / 9.0
            local_mean  = tf.nn.conv2d(mag_4d, mean_kernel, strides=[1, 1, 1, 1], padding='SAME')

            # A pixel is a peak if it equals the local max and is above the local mean
            is_peak = tf.logical_and(
                tf.abs(mag_4d - local_max) < self.EPSILON_TF,
                mag_4d > local_mean + self.EPSILON_TF
            )
            peak_heights  = tf.where(is_peak, mag_4d - local_mean, tf.zeros_like(mag_4d))

            # peak-to-average ratio: how much the peaks stand out
            sum_peak_height = tf.reduce_sum(peak_heights,     axis=[1, 2, 3])
            mean_spectrum   = tf.reduce_mean(tf.abs(mag_4d),  axis=[1, 2, 3])
            par_score       = sum_peak_height / (mean_spectrum * tf.cast(h * w, tf.float32) + self.EPSILON_TF)

            # Normalise: empirically PAR up to ~0.05 covers real images; clip to [0,1]
            score = tf.clip_by_value(par_score / 0.05, 0.0, 1.0)
            score = tf.reshape(score, [-1, 1, 1])

            feature_map = tf.ones_like(gray) * score
            return self._apply_mask(feature_map, image)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, config.patch_size, config.patch_size, 3], dtype=tf.float32)
    ])
    def _extract_laplacian_peak_ratio_feature(self, image: tf.Tensor) -> tf.Tensor:
        """
        Laplacian Peak Ratio: Measures the concentration of high-frequency energy
        relative to background mid-frequency energy in the Laplacian domain.

        AI generators produce unnatural sharpness spikes from their upsampling layers
        (transposed convolutions, pixel shuffle, etc.), creating a high local
        peak-to-mean ratio in the Laplacian response. Physical camera lenses distribute
        blur and sharpness continuously and smoothly, creating a lower, more uniform ratio.

        This signal is robust to JPEG compression because it measures relative ratios
        between frequency bands rather than absolute pixel values.

        Method:
            1. Apply Laplacian kernel to get the high-frequency response.
            2. Compute local max (3x3 max-pool) vs. local mean (3x3 avg-pool).
            3. Aggregate the mean peak-to-mean ratio across the patch.
            4. Normalise: natural ~2-4x ratio maps to low score; AI spikes ~8-12x map to high.
        """
        with tf.name_scope('laplacian_peak_ratio'):
            gray = self._rgb_to_grayscale(image)  # [batch, H, W]

            laplacian_kernel = tf.reshape(
                tf.constant([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=tf.float32),
                [3, 3, 1, 1]
            )

            rank = tf.rank(gray)
            gray_exp = tf.cond(
                tf.equal(rank, 2),
                lambda: tf.expand_dims(tf.expand_dims(gray, 0), -1),
                lambda: tf.expand_dims(gray, -1)
            )  # [batch, H, W, 1]

            lap     = tf.nn.conv2d(gray_exp, laplacian_kernel, strides=[1, 1, 1, 1], padding='SAME')
            lap_abs = tf.abs(lap)  # [batch, H, W, 1]

            # Local max via 3x3 max-pool
            local_max = tf.nn.max_pool2d(lap_abs, ksize=3, strides=1, padding='SAME')

            # Local mean via 3x3 average-pool
            mean_kernel = tf.ones([3, 3, 1, 1], dtype=tf.float32) / 9.0
            local_mean  = tf.nn.conv2d(lap_abs, mean_kernel, strides=[1, 1, 1, 1], padding='SAME')

            # Peak-to-mean ratio at each pixel
            peak_ratio = local_max / (local_mean + self.EPSILON_TF)  # [batch, H, W, 1]

            # Aggregate: mean peak ratio across spatial dims -> [batch]
            mean_ratio = tf.reduce_mean(peak_ratio, axis=[1, 2, 3])

            # Normalise: ratio in [1, 10+]; natural images ~2-4, AI spikes ~8-12
            score = tf.clip_by_value((mean_ratio - 1.0) / 9.0, 0.0, 1.0)

            score = tf.cond(tf.equal(rank, 2), lambda: tf.squeeze(score, [0]), lambda: score)
            score = tf.reshape(score, [-1, 1, 1])

            feature_map = tf.ones_like(gray) * score
            return self._apply_mask(feature_map, image)


    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, config.patch_size, config.patch_size, 3], dtype=tf.float32)
    ])
    def _extract_symmetry_feature(self, image: tf.Tensor) -> tf.Tensor:
        with tf.name_scope('symmetry_feature'):
            gray = tf.cast(self._rgb_to_grayscale(image), tf.float32)
            h, w = tf.shape(gray)[-2], tf.shape(gray)[-1]
            left_half  = gray[..., :, :w // 2]
            right_half = tf.reverse(gray[..., :, w // 2:], axis=[-1])
            min_w      = tf.minimum(tf.shape(left_half)[-1], tf.shape(right_half)[-1])
            
            diff_h = tf.abs(left_half[..., :, :min_w] - right_half[..., :, :min_w])
            h_sym = 1.0 - tf.reduce_mean(diff_h, axis=[-2, -1]) / \
                    (tf.reduce_max(gray, axis=[-2, -1]) + self.EPSILON_TF)
                    
            top_half    = gray[..., :h // 2, :]
            bottom_half = tf.reverse(gray[..., h // 2:, :], axis=[-2])
            min_h       = tf.minimum(tf.shape(top_half)[-2], tf.shape(bottom_half)[-2])
            
            diff_v = tf.abs(top_half[..., :min_h, :] - bottom_half[..., :min_h, :])
            v_sym = 1.0 - tf.reduce_mean(diff_v, axis=[-2, -1]) / \
                    (tf.reduce_max(gray, axis=[-2, -1]) + self.EPSILON_TF)
                    
            symmetry_score = tf.clip_by_value((h_sym + v_sym) / 2.0, 0.0, 1.0)
            
            ih, iw = tf.shape(image)[-3], tf.shape(image)[-2]
            feature_map = self._create_radial_gradient_map(
                tf.cast(ih, tf.float32), tf.cast(iw, tf.float32), symmetry_score
            )
            return self._apply_mask(feature_map, image)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, config.patch_size, config.patch_size, 3], dtype=tf.float32)
    ])
    def _extract_dct_feature(self, image: tf.Tensor) -> tf.Tensor:
        """
        Frequency Decay Analysis: Natural images follow a 1/f energy decay in DCT space.
        AI generators violate this law by leaving excess energy in mid-to-high frequency
        coefficient bands relative to the low-frequency band.

        Method:
            1. Apply 2D DCT to 8x8 blocks.
            2. Measure energy in Low (top-left 3x3), Mid, and High (bottom-right 3x3) bands.
            3. Score = deviation from the expected natural-image decay: Low >> Mid > High.
               A high score means the mid/high bands have *more* relative energy than expected.
        """
        with tf.name_scope('dct_feature'):
            gray = tf.cast(self._rgb_to_grayscale(image), tf.float32)

            block_size = 8
            rank = tf.rank(gray)
            gray_exp = tf.cond(
                tf.equal(rank, 2),
                lambda: tf.expand_dims(tf.expand_dims(gray, 0), -1),
                lambda: tf.expand_dims(gray, -1)
            )
            blocks = tf.image.extract_patches(
                images=gray_exp,
                sizes=[1, block_size, block_size, 1],
                strides=[1, block_size, block_size, 1],
                rates=[1, 1, 1, 1],
                padding='VALID'
            )
            n_batch = tf.shape(blocks)[0]
            n_bh    = tf.shape(blocks)[1]
            n_bw    = tf.shape(blocks)[2]
            blocks  = tf.reshape(blocks, [n_batch * n_bh * n_bw, block_size, block_size])

            dct_rows = tf.signal.dct(blocks, type=2, norm='ortho')
            dct_2d   = tf.signal.dct(tf.transpose(dct_rows, [0, 2, 1]), type=2, norm='ortho')
            dct_2d   = tf.transpose(dct_2d, [0, 2, 1])  # [n_blocks, 8, 8]

            coef_sq  = tf.square(dct_2d)  # squared DCT coefficients

            # Band masks over the 8x8 DCT coefficient grid:
            #   Low  : 3x3 top-left  (low frequencies, dominant in natural images)
            #   High : 3x3 bottom-right (high frequencies, prone to AI artifacts)
            #   Mid  : everything in between
            rows = tf.cast(tf.range(block_size), tf.float32)
            cols = tf.cast(tf.range(block_size), tf.float32)
            rr, cc = tf.meshgrid(rows, cols, indexing='ij')  # [8, 8]

            low_mask  = tf.cast(tf.logical_and(rr < 3.0, cc < 3.0), tf.float32)  # 9 coeffs
            high_mask = tf.cast(tf.logical_and(rr > 4.0, cc > 4.0), tf.float32)  # 9 coeffs
            mid_mask  = 1.0 - low_mask - high_mask

            low_e  = tf.reduce_sum(coef_sq * low_mask,  axis=[1, 2])
            mid_e  = tf.reduce_sum(coef_sq * mid_mask,  axis=[1, 2])
            high_e = tf.reduce_sum(coef_sq * high_mask, axis=[1, 2])
            total_e = low_e + mid_e + high_e + self.EPSILON_TF

            mid_frac  = mid_e  / total_e
            high_frac = high_e / total_e

            # Natural images: mid_frac ~ 0.35, high_frac ~ 0.05.
            # AI-generated:   excess mid/high → higher score.
            # Clip & normalise so that the expected natural baseline maps to ~0
            # and strong AI artifacts map to ~1.
            expected_mid  = tf.constant(0.35, dtype=tf.float32)
            expected_high = tf.constant(0.05, dtype=tf.float32)

            decay_violation = (
                tf.nn.relu(mid_frac  - expected_mid)  * 1.5 +
                tf.nn.relu(high_frac - expected_high) * 3.0
            )
            dct_score = tf.clip_by_value(decay_violation, 0.0, 1.0)

            dct_score   = tf.reshape(
                tf.reduce_mean(tf.reshape(dct_score, [n_batch, n_bh * n_bw]), axis=1),
                [-1, 1, 1]
            )
            feature_map = tf.ones_like(gray) * dct_score
            return self._apply_mask(feature_map, image)