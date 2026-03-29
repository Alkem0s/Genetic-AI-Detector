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

    @staticmethod
    @tf.function
    def _apply_sobel_filters(gray: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        gray_expanded = tf.expand_dims(tf.expand_dims(gray, 0), -1)
        grad_x = tf.nn.conv2d(gray_expanded, StructuralFeatureExtractor.SOBEL_X_KERNEL,
                              strides=[1, 1, 1, 1], padding='SAME')
        grad_y = tf.nn.conv2d(gray_expanded, StructuralFeatureExtractor.SOBEL_Y_KERNEL,
                              strides=[1, 1, 1, 1], padding='SAME')
        return tf.squeeze(grad_x, [0, 3]), tf.squeeze(grad_y, [0, 3])

    @staticmethod
    @tf.function
    def _tf_native_canny(gray, low_threshold=50.0, high_threshold=150.0):
        gray_expanded = tf.expand_dims(tf.expand_dims(gray, 0), -1)
        blurred = tf.nn.conv2d(gray_expanded, StructuralFeatureExtractor.GAUSSIAN_KERNEL,
                               strides=[1, 1, 1, 1], padding='SAME')
        blurred = tf.squeeze(blurred, [0, 3])
        grad_x, grad_y = StructuralFeatureExtractor._apply_sobel_filters(blurred)
        magnitude = tf.sqrt(grad_x**2 + grad_y**2 + BaseFeatureExtractor.EPSILON_TF)
        magnitude_padded = tf.pad(magnitude, [[1, 1], [1, 1]], mode='REFLECT')
        max_pooled = tf.nn.max_pool2d(
            tf.expand_dims(tf.expand_dims(magnitude_padded, 0), -1),
            ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID'
        )
        max_pooled = tf.squeeze(max_pooled, [0, 3])
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
        return gradient * score

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[config.patch_size, config.patch_size, 3], dtype=tf.float32)
    ])
    def _extract_gradient_feature(self, image: tf.Tensor) -> tf.Tensor:
        with tf.name_scope('gradient_feature'):
            gray = tf.cast(self._rgb_to_grayscale(image), tf.float32)   # ← inherited
            grad_x, grad_y = self._apply_sobel_filters(gray)
            magnitude = tf.sqrt(grad_x**2 + grad_y**2 + self.EPSILON_TF)
            magnitude_normalized = tf.clip_by_value(
                magnitude / (self.GRADIENT_MAX_VAL + self.EPSILON_TF), 0.0, 1.0
            )
            mean_mag  = tf.reduce_mean(magnitude_normalized)
            variance  = tf.reduce_mean(tf.square(magnitude_normalized - mean_mag))
            perfection_score = 1.0 / (variance + self.EPSILON_TF)
            h, w = tf.shape(image)[0], tf.shape(image)[1]
            feature_map = self._create_radial_gradient_map(
                tf.cast(h, tf.float32), tf.cast(w, tf.float32), perfection_score
            )
            feature_map = tf.clip_by_value(feature_map / self.FEATURE_MAX_VAL, 0.0, 1.0)
            return self._apply_mask(feature_map, image)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[config.patch_size, config.patch_size, 3], dtype=tf.float32)
    ])
    def _extract_pattern_feature(self, image: tf.Tensor) -> tf.Tensor:
        with tf.name_scope('pattern_feature'):
            gray = tf.cast(self._rgb_to_grayscale(image), tf.float32)
            f_transform = tf.signal.fft2d(tf.cast(gray, tf.complex64))
            f_shift     = tf.signal.fftshift(f_transform)
            magnitude_spectrum = tf.math.log(tf.abs(f_shift) + self.EPSILON_TF)
            h, w = tf.shape(magnitude_spectrum)[0], tf.shape(magnitude_spectrum)[1]
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
            high_freq_energy = tf.reduce_sum(magnitude_spectrum * high_freq_mask)
            low_freq_energy  = tf.reduce_sum(magnitude_spectrum * low_freq_mask)
            pattern_score   = high_freq_energy / (low_freq_energy + self.EPSILON_TF)
            ih, iw = tf.shape(image)[0], tf.shape(image)[1]
            feature_map = self._create_radial_gradient_map(
                tf.cast(ih, tf.float32), tf.cast(iw, tf.float32), pattern_score
            )
            feature_map = tf.clip_by_value(feature_map / self.FEATURE_MAX_VAL, 0.0, 1.0)
            return self._apply_mask(feature_map, image)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[config.patch_size, config.patch_size, 3], dtype=tf.float32)
    ])
    def _extract_edge_feature(self, image: tf.Tensor) -> tf.Tensor:
        with tf.name_scope('edge_feature'):
            gray = tf.cast(self._rgb_to_grayscale(image), tf.float32) * 255.0
            edge_map = self._tf_native_canny(gray, low_threshold=50.0, high_threshold=150.0)
            return self._apply_mask(edge_map, image)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[config.patch_size, config.patch_size, 3], dtype=tf.float32)
    ])
    def _extract_symmetry_feature(self, image: tf.Tensor) -> tf.Tensor:
        with tf.name_scope('symmetry_feature'):
            gray = tf.cast(self._rgb_to_grayscale(image), tf.float32)
            h, w = tf.shape(gray)[0], tf.shape(gray)[1]
            left_half  = gray[:, :w // 2]
            right_half = tf.reverse(gray[:, w // 2:], axis=[1])
            min_w      = tf.minimum(tf.shape(left_half)[1], tf.shape(right_half)[1])
            h_sym = 1.0 - tf.reduce_mean(tf.abs(left_half[:, :min_w] - right_half[:, :min_w])) / \
                    (tf.reduce_max(gray) + self.EPSILON_TF)
            top_half    = gray[:h // 2, :]
            bottom_half = tf.reverse(gray[h // 2:, :], axis=[0])
            min_h       = tf.minimum(tf.shape(top_half)[0], tf.shape(bottom_half)[0])
            v_sym = 1.0 - tf.reduce_mean(tf.abs(top_half[:min_h, :] - bottom_half[:min_h, :])) / \
                    (tf.reduce_max(gray) + self.EPSILON_TF)
            symmetry_score = tf.clip_by_value((h_sym + v_sym) / 2.0, 0.0, 1.0)
            ih, iw = tf.shape(image)[0], tf.shape(image)[1]
            feature_map = self._create_radial_gradient_map(
                tf.cast(ih, tf.float32), tf.cast(iw, tf.float32), symmetry_score
            )
            return self._apply_mask(feature_map, image)