# feature_extractor.py
import os
import numpy as np

import tensorflow as tf
from typing import Tuple, Dict, List, Any

import global_config as config
import utils
from structural_features import StructuralFeatureExtractor
from texture_features import TextureFeatureExtractor

class FeatureExtractor:
    """
    Class for feature extraction in AI image detection system.
    Implements methods to extract various features that can identify AI-generated artifacts.
    
    The GeneticFeatureOptimizer expects this module to extract features
    from patches of images and return them in a consistent format.
    """
    def __init__(self, config):
        self.structural_extractor = StructuralFeatureExtractor()
        self.texture_extractor = TextureFeatureExtractor()
        self.feature_weights = config.feature_weights

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, config.image_size, config.image_size, 3], dtype=tf.float32)
    ])
    def extract_batch_patch_features(self, images: tf.Tensor) -> tf.Tensor:
        """
        Vectorized batch feature extraction.
        Extracts all patches from the entire image batch in one shot, then runs
        a single flat tf.map_fn over all (batch * n_patches_h * n_patches_w) patches
        simultaneously.  This eliminates the previously nested map_fn loop and
        lets the GPU work on the full patch workload at once.

        Returns:
            Float32 tensor of shape (batch_size, n_patches_h, n_patches_w, n_features).
        """
        num_features    = len(self.feature_weights)
        static_patch_size = config.patch_size

        if tf.size(images) == 0:
            n_patches_h = config.image_size // static_patch_size
            n_patches_w = config.image_size // static_patch_size
            return tf.zeros([0, n_patches_h, n_patches_w, num_features], dtype=tf.float32)

        batch_size = tf.shape(images)[0]

        # Extract all patches from all images: [batch, n_ph, n_pw, patch_size*patch_size*3]
        all_patches = tf.image.extract_patches(
            images=images,
            sizes=[1, static_patch_size, static_patch_size, 1],
            strides=[1, static_patch_size, static_patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        n_patches_h = tf.shape(all_patches)[1]
        n_patches_w = tf.shape(all_patches)[2]
        total_patches = batch_size * n_patches_h * n_patches_w

        # Flatten to [total_patches, patch_size, patch_size, 3]
        flat_patches = tf.reshape(
            all_patches, [total_patches, static_patch_size, static_patch_size, 3]
        )

        # Process all patches directly using fully vectorized tensor operations
        flat_features = self._extract_single_patch_features(flat_patches)
        # Reduce spatial dims of each patch to a scalar per feature: [total_patches, num_features]
        flat_features = tf.reduce_mean(flat_features, axis=[1, 2])

        # Reshape back to [batch, n_patches_h, n_patches_w, num_features]
        return tf.reshape(flat_features, [batch_size, n_patches_h, n_patches_w, num_features])

    def extract_patch_features(self, image: tf.Tensor) -> tf.Tensor:
        """Single-image wrapper that delegates to extract_batch_patch_features."""
        return self.extract_batch_patch_features(tf.expand_dims(image, 0))[0]


    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, config.patch_size, config.patch_size, 3], dtype=tf.float32)
    ])
    def _extract_single_patch_features(self, patch: tf.Tensor) -> tf.Tensor:
        """
        Extracts all 11 individual features from a single patch.
        Returns a 3D tensor: [patch_size, patch_size, num_features]

        Feature order must match global_config / optuna_config.feature_weight_ranges key order:
            gradient, pattern, noise, symmetry, texture,
            color, hash, dct, glcm,
            noise_spectrum, local_entropy
        """
        gradient_feature   = self.structural_extractor._extract_gradient_feature(patch)
        pattern_feature    = self.structural_extractor._extract_pattern_feature(patch)
        noise_feature      = self.texture_extractor._extract_noise_feature(patch)
        symmetry_feature   = self.structural_extractor._extract_symmetry_feature(patch)
        texture_feature    = self.texture_extractor._extract_texture_feature(patch)
        color_feature      = self.texture_extractor._extract_color_feature(patch)
        hash_feature       = self.texture_extractor._extract_hash_feature(patch)
        dct_feature        = self.structural_extractor._extract_dct_feature(patch)
        glcm_feature       = self.texture_extractor._extract_glcm_feature(patch)
        noise_spectrum_feature = self.structural_extractor._extract_noise_spectrum_feature(patch)
        local_entropy_feature = self.texture_extractor._extract_local_entropy_feature(patch)

        # Stack in the same order as feature_weights in global_config.py / optuna_config.py
        feature_stack = tf.stack([
            gradient_feature,
            pattern_feature,
            noise_feature,
            symmetry_feature,
            texture_feature,
            color_feature,
            hash_feature,
            dct_feature,
            glcm_feature,
            noise_spectrum_feature,
            local_entropy_feature,
        ], axis=-1)  # [patch_size, patch_size, 11]

        return feature_stack
    
class FeaturePipeline:
    """
    Manages feature extraction, caching, and dataset preparation for ModelWrapper.

    """

    def __init__(
        self,
        genetic_rules: tf.Tensor,
        n_patches_h: tf.Tensor,
        n_patches_w: tf.Tensor,
        feature_cache_dir: str,
        feature_channels: int = 8,
    ):
        """
        Args:
            genetic_rules:      Evolved rule tensor from the genetic algorithm.
            n_patches_h:        Number of patches along the height axis (tf.int32 scalar).
            n_patches_w:        Number of patches along the width axis  (tf.int32 scalar).
            feature_cache_dir:  Directory used to persist pre-computed feature .npy files.
            feature_channels:   Number of feature channels produced by FeatureExtractor.
        """
        self.genetic_rules     = genetic_rules
        self.n_patches_h       = n_patches_h
        self.n_patches_w       = n_patches_w
        self.feature_cache_dir = feature_cache_dir
        self.feature_channels  = feature_channels

        # Lazily initialised – created on first use.
        self._feature_extractor: FeatureExtractor | None = None

        os.makedirs(self.feature_cache_dir, exist_ok=True)

    def _ensure_feature_extractor(self) -> None:
        """Initialise the FeatureExtractor the first time it is needed."""
        if self._feature_extractor is None:
            self._feature_extractor = FeatureExtractor(config)

    def _generate_mask(self, patch_features: tf.Tensor) -> tf.Tensor:
        """Thin wrapper so callers don't need to import fitness_evaluation."""
        from fitness_evaluation import generate_dynamic_mask
        mask, _ = generate_dynamic_mask(
            patch_features,
            self.genetic_rules,
        )
        return mask

    def set_genetic_rules(self, genetic_rules: tf.Tensor) -> None:
        """Replace the genetic rules (e.g. after a new GA run)."""
        self.genetic_rules = tf.convert_to_tensor(genetic_rules, dtype=tf.float32)

    def extract_batch_features(self, images: tf.Tensor) -> tf.Tensor:
        """
        Extract patch features for a batch of images, apply genetic masks, and
        return pixel-level feature maps ready for the CNN.

        Args:
            images: Float32 tensor of shape (batch, H, W, 3).

        Returns:
            Float32 tensor of shape (batch, H, W, feature_channels).
        """

        self._ensure_feature_extractor()

        if not isinstance(images, tf.Tensor):
            images = tf.convert_to_tensor(images, dtype=tf.float32)

        # (batch, n_patches_h, n_patches_w, n_features)
        patch_features = self._feature_extractor.extract_batch_patch_features(images)

        # Build masks for the batch using vectorized operations directly
        batch_masks = self._generate_mask(patch_features)

        # (batch, n_patches_h, n_patches_w) → pixel masks
        pixel_mask = utils.convert_patch_mask_to_pixel_mask(batch_masks)

        if pixel_mask.shape.rank == 3:       # (batch, H, W)
            pixel_mask = tf.expand_dims(pixel_mask, axis=-1)
        elif pixel_mask.shape.rank == 2:     # (H, W) — single image
            pixel_mask = tf.expand_dims(tf.expand_dims(pixel_mask, 0), axis=-1)

        # Resize patch features to full image resolution.
        resized_feature_maps = tf.image.resize(
            patch_features,
            [config.image_size, config.image_size],
            method=tf.image.ResizeMethod.BILINEAR,
        )

        return resized_feature_maps * pixel_mask

    def extract_single_image_features(self, image: tf.Tensor) -> tf.Tensor:
        """
        Convenience wrapper: extract features for a single image tensor.

        Args:
            image: Float32 tensor of shape (H, W, 3) or (1, H, W, 3).

        Returns:
            Float32 tensor of shape (H, W, feature_channels).
        """
        if len(image.shape) == 3:
            image = tf.expand_dims(image, axis=0)
        return self.extract_batch_features(image)[0]

    def precompute_features(
        self, dataset: tf.data.Dataset, dataset_name: str
    ) -> None:
        """
        Extract features for every image in *dataset* and persist them as
        individual patch-level .npy files under ``feature_cache_dir/<dataset_name>/``.
        This stores raw patch features (shape [32, 32, 11]), which are 64x smaller
        than full-resolution pixel feature maps, saving huge amounts of disk space.

        Args:
            dataset:      A tf.data.Dataset that yields (images, labels) or images.
            dataset_name: Subdirectory name used for this dataset's cache.
        """
        import logging
        from tqdm import tqdm
        logger = logging.getLogger(__name__)
        logger.info(f"Precomputing patch features for {dataset_name}")

        save_dir = os.path.join(self.feature_cache_dir, dataset_name)
        os.makedirs(save_dir, exist_ok=True)

        self._ensure_feature_extractor()

        # Try to determine cardinality for progress bar total
        cardinality = dataset.cardinality().numpy()
        total_batches = int(cardinality) if cardinality > 0 else None

        index = 0
        pbar = tqdm(dataset, total=total_batches, desc=f"Precomputing {dataset_name} features")
        for batch in pbar:
            images   = batch[0] if isinstance(batch, tuple) else batch
            # Extract raw patch-level features: [batch, n_patches_h, n_patches_w, feature_channels]
            patch_features = self._feature_extractor.extract_batch_patch_features(images)

            for i in range(patch_features.shape[0]):
                np.save(
                    os.path.join(save_dir, f"feature_{index}.npy"),
                    patch_features[i].numpy().astype(np.float16), # Save as float16 to save space
                )
                index += 1

        logger.info(f"Saved {index} patch features for {dataset_name}")

    def load_features(self, dataset_name: str) -> tf.data.Dataset:
        """
        Return a tf.data.Dataset that streams pre-computed patch-level feature tensors
        from ``feature_cache_dir/<dataset_name>/``.

        Args:
            dataset_name: Subdirectory whose .npy files should be loaded.

        Returns:
            A Dataset of float32 tensors, each shaped
            (n_patches_h, n_patches_w, feature_channels).
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Loading precomputed patch features for {dataset_name}")

        save_dir = os.path.join(self.feature_cache_dir, dataset_name)
        feature_paths = sorted(
            [
                os.path.join(save_dir, f)
                for f in os.listdir(save_dir)
                if f.endswith(".npy")
            ],
            key=lambda x: int(x.split("_")[-1].split(".")[0]),
        )

        def load_feature(path: bytes) -> np.ndarray:
            return np.load(path.decode("utf-8")).astype(np.float32)

        feature_dataset = tf.data.Dataset.from_tensor_slices(feature_paths)
        feature_dataset = feature_dataset.map(
            lambda path: tf.numpy_function(load_feature, [path], tf.float32),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        feature_dataset = feature_dataset.map(
            lambda x: tf.ensure_shape(
                x, [self.n_patches_h, self.n_patches_w, self.feature_channels]
            )
        )
        return feature_dataset

    def prepare_dataset(
        self,
        dataset: tf.data.Dataset,
        dataset_name: str | None = None,
    ) -> tf.data.Dataset:
        """
        Combine an image dataset with feature maps, either from the cache
        (when *dataset_name* is given) or computed on-the-fly.

        Args:
            dataset:      Source tf.data.Dataset yielding (images, labels).
            dataset_name: If provided, load features from the pre-computed cache.

        Returns:
            A Dataset yielding ({'image_input': …, 'feature_input': …}, labels).
        """
        if dataset_name:
            feature_dataset = self.load_features(dataset_name)
            # Batch the feature dataset to match the image dataset batching structure!
            # We use config.cnn_batch_size which is the global batch size.
            feature_dataset = feature_dataset.batch(config.cnn_batch_size)

            def combine(inputs, feature_batch):
                # inputs: (images, labels) or images
                # feature_batch: [batch_size, n_patches_h, n_patches_w, feature_channels]
                
                # Apply mask to patch features
                batch_masks = self._generate_mask(feature_batch) # [batch, n_ph, n_pw]
                pixel_mask = utils.convert_patch_mask_to_pixel_mask(batch_masks) # [batch, H, W]
                pixel_mask = tf.expand_dims(pixel_mask, axis=-1) # [batch, H, W, 1]
                
                # Resize patch features to full image resolution
                resized_feature_maps = tf.image.resize(
                    feature_batch,
                    [config.image_size, config.image_size],
                    method=tf.image.ResizeMethod.BILINEAR,
                ) # [batch, H, W, feature_channels]
                
                # Apply mask
                masked_features = resized_feature_maps * pixel_mask
                
                if isinstance(inputs, tuple):
                    image, label = inputs
                    return {"image_input": image, "feature_input": masked_features}, label
                return {"image_input": inputs, "feature_input": masked_features}

            return (
                tf.data.Dataset.zip((dataset, feature_dataset))
                .map(combine, num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE)
            )

        return self._prepare_dataset_on_the_fly(dataset)

    def _prepare_dataset_on_the_fly(
        self, dataset: tf.data.Dataset
    ) -> tf.data.Dataset:
        """
        Fallback: compute features for each batch as it is consumed.

        Args:
            dataset: Source tf.data.Dataset yielding (images, labels).

        Returns:
            A Dataset yielding ({'image_input': …, 'feature_input': …}, labels).
        """
        import logging
        logging.getLogger(__name__).info(
            "Preparing dataset with on-the-fly feature computation"
        )

        pipeline = self  # capture for the closure

        def add_features_on_the_fly(
            images: tf.Tensor, labels: tf.Tensor
        ) -> tuple:
            images   = tf.cast(images, tf.float32)
            features = pipeline.extract_batch_features(images)
            return {"image_input": images, "feature_input": features}, labels

        return dataset.map(
            add_features_on_the_fly,
            num_parallel_calls=tf.data.AUTOTUNE,
        ).prefetch(tf.data.AUTOTUNE)