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
            gray      = self._rgb_to_grayscale(image)
            gray_uint8 = tf.cast(gray, tf.uint8)

            def py_lbp(image_np):
                image_np = image_np.numpy()
                radius, n_points = 1, 8
                lbp_image = local_binary_pattern(image_np, n_points, radius, method='uniform')
                max_val   = np.max(lbp_image)
                if max_val == 0:
                    return np.zeros_like(lbp_image, dtype=np.float32)
                return (lbp_image / max_val).astype(np.float32)

            texture_map = tf.py_function(py_lbp, [gray_uint8], tf.float32)
            texture_map.set_shape([config.patch_size, config.patch_size])
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
            def py_hash_analysis(image_tf_input, epsilon_np_val):
                try:
                    if not hasattr(Image, 'ANTIALIAS'):
                        Image.ANTIALIAS = Image.LANCZOS
                    image_np  = image_tf_input.numpy()
                    image_pil = Image.fromarray(
                        cv2.cvtColor(image_np.astype(np.uint8), cv2.COLOR_BGR2RGB)
                        if image_np.shape[-1] == 3
                        else image_np.astype(np.uint8)
                    )
                    phash = imagehash.phash(image_pil, hash_size=8)
                    dhash = imagehash.dhash(image_pil, hash_size=8)
                    return np.float32((phash.hash.sum() + dhash.hash.sum()) / (2 * 64))
                except Exception as e:
                    tf.print(f"Error in py_hash_analysis: {e}")
                    return np.float32(0.0)

            hash_score = tf.py_function(
                func=py_hash_analysis,
                inp=[image, self.EPSILON_NP],
                Tout=tf.float32
            )
            hash_score.set_shape([])
            h, w        = tf.shape(image)[0], tf.shape(image)[1]
            feature_map = tf.fill([h, w], hash_score)
            return self._apply_mask(feature_map, image)