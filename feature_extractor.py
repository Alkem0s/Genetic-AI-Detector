# feature_extractor.py
import os
import numpy as np

import tensorflow as tf
from typing import Tuple, Dict, List, Any

import global_config as config
import utils
from structural_features import StructuralFeatureExtractor
from texture_features import TextureFeatureExtractor

# Global in-memory cache to speed up HPO trials and epoch times by avoiding disk reads
_IN_MEMORY_FEATURE_CACHE = {}

def _augment_image(image):
    # Random flips
    image = tf.image.random_flip_left_right(image)
    # Random brightness and contrast adjustments
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    # Clip to [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


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
        Extracts all 8 individual features from a single patch.
        Returns a 3D tensor: [patch_size, patch_size, num_features]

        Feature order must match global_config / optuna_config.feature_weight_ranges key order:
            gradient, noise, symmetry, texture, color, hash, glcm, local_entropy
        """
        gradient_feature   = self.structural_extractor._extract_gradient_feature(patch)
        noise_feature      = self.texture_extractor._extract_noise_feature(patch)
        symmetry_feature   = self.structural_extractor._extract_symmetry_feature(patch)
        texture_feature    = self.texture_extractor._extract_texture_feature(patch)
        color_feature      = self.texture_extractor._extract_color_feature(patch)
        hash_feature       = self.texture_extractor._extract_hash_feature(patch)
        glcm_feature       = self.texture_extractor._extract_glcm_feature(patch)
        local_entropy_feature = self.texture_extractor._extract_local_entropy_feature(patch)

        # Stack in the same order as feature_weights in global_config.py / optuna_config.py
        feature_stack = tf.stack([
            gradient_feature,
            noise_feature,
            symmetry_feature,
            texture_feature,
            color_feature,
            hash_feature,
            glcm_feature,
            local_entropy_feature,
        ], axis=-1)  # [patch_size, patch_size, 8]

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
        feature_scales: tf.Tensor = None,
    ):
        """
        Args:
            genetic_rules:      Evolved rule tensor from the genetic algorithm.
            n_patches_h:        Number of patches along the height axis (tf.int32 scalar).
            n_patches_w:        Number of patches along the width axis  (tf.int32 scalar).
            feature_cache_dir:  Directory used to persist pre-computed feature .npy files.
            feature_channels:   Number of feature channels produced by FeatureExtractor.
            feature_scales:     Optional feature normalisation scales.
        """
        self.genetic_rules     = genetic_rules
        self.n_patches_h       = n_patches_h
        self.n_patches_w       = n_patches_w
        self.feature_cache_dir = feature_cache_dir
        self.feature_channels  = feature_channels
        self.feature_scales    = feature_scales

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

        # Scale features if feature_scales is available
        if hasattr(self, 'feature_scales') and self.feature_scales is not None:
            patch_features = patch_features / self.feature_scales

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
        a single consolidated features.npy file under ``feature_cache_dir/<dataset_name>/``.
        This stores raw patch features (shape [N, 32, 32, 11]), which are 64x smaller
        than full-resolution pixel feature maps, saving huge amounts of disk space
        and avoiding the heavy I/O overhead of writing individual files.

        Args:
            dataset:      A tf.data.Dataset that yields (images, labels) or images.
            dataset_name: Subdirectory name used for this dataset's cache.
        """
        import logging
        from tqdm import tqdm
        logger = logging.getLogger(__name__)

        save_dir = os.path.join(self.feature_cache_dir, dataset_name)
        features_file = os.path.join(save_dir, "features.npy")
        
        # Check if features are already precomputed and reuse them
        if os.path.exists(features_file):
            logger.info(f"Precomputed features for {dataset_name} already exist in {features_file}. Reusing cached features.")
            return
            
        # Legacy fallback check (if individual files already exist, do not recompute)
        if os.path.exists(save_dir):
            existing_files = [f for f in os.listdir(save_dir) if f.endswith('.npy') and f != "features.npy"]
            if len(existing_files) > 0:
                logger.info(f"Legacy precomputed features for {dataset_name} already exist in {save_dir} ({len(existing_files)} files). Reusing cached features.")
                return

        # Invalidate in-memory cache for this directory before writing new features
        keys_to_remove = [k for k in _IN_MEMORY_FEATURE_CACHE if k[0] == save_dir]
        for k in keys_to_remove:
            _IN_MEMORY_FEATURE_CACHE.pop(k, None)

        logger.info(f"Precomputing patch features for {dataset_name}")
        os.makedirs(save_dir, exist_ok=True)

        self._ensure_feature_extractor()

        # Batch the dataset for fast feature extraction
        batched_dataset = dataset.batch(config.cnn_batch_size).prefetch(tf.data.AUTOTUNE)

        # Try to determine cardinality for progress bar total
        import math
        cardinality = dataset.cardinality().numpy()
        total_batches = math.ceil(cardinality / config.cnn_batch_size) if cardinality > 0 else None

        features_list = []
        pbar = tqdm(batched_dataset, total=total_batches, desc=f"Precomputing {dataset_name} features")
        for batch in pbar:
            images   = batch[0] if isinstance(batch, tuple) else batch
            # Extract raw patch-level features: [batch, n_patches_h, n_patches_w, feature_channels]
            patch_features = self._feature_extractor.extract_batch_patch_features(images)
            features_list.append(patch_features.numpy().astype(np.float16))

        if features_list:
            all_features = np.concatenate(features_list, axis=0)
            np.save(features_file, all_features)
            logger.info(f"Saved {all_features.shape[0]} patch features to {features_file}")

    def load_features(self, dataset_name: str) -> tf.data.Dataset:
        """
        Return a tf.data.Dataset that streams pre-computed patch-level feature tensors.
        Loads from disk into RAM once and caches them to speed up subsequent epochs and HPO trials.

        Args:
            dataset_name: Subdirectory whose features should be loaded.

        Returns:
            A Dataset of float32 tensors, each shaped
            (n_patches_h, n_patches_w, feature_channels).
        """
        import logging
        logger = logging.getLogger(__name__)

        save_dir = os.path.join(self.feature_cache_dir, dataset_name)
        features_file = os.path.join(save_dir, "features.npy")
        h_val = self.n_patches_h.numpy() if hasattr(self.n_patches_h, 'numpy') else self.n_patches_h
        w_val = self.n_patches_w.numpy() if hasattr(self.n_patches_w, 'numpy') else self.n_patches_w
        cache_key = (save_dir, h_val, w_val, self.feature_channels)

        if cache_key in _IN_MEMORY_FEATURE_CACHE:
            logger.info(f"Reusing in-memory precomputed patch features for {dataset_name}")
            features_array = _IN_MEMORY_FEATURE_CACHE[cache_key]
        else:
            logger.info(f"Loading precomputed patch features for {dataset_name} into RAM...")
            if os.path.exists(features_file):
                features_array = np.load(features_file).astype(np.float32)
            else:
                # Fallback to load legacy individual .npy files if features.npy doesn't exist
                logger.warning(f"{features_file} not found. Searching for legacy individual .npy files...")
                feature_paths = sorted(
                    [
                        os.path.join(save_dir, f)
                        for f in os.listdir(save_dir)
                        if f.endswith(".npy")
                    ],
                    key=lambda x: int(x.split("_")[-1].split(".")[0]),
                )

                # Load all .npy files into a list and stack into a single numpy array
                features_list = []
                for path in feature_paths:
                    features_list.append(np.load(path).astype(np.float32))

                if features_list:
                    features_array = np.stack(features_list, axis=0)
                else:
                    raise FileNotFoundError(f"No precomputed features found in {save_dir}")
                
            _IN_MEMORY_FEATURE_CACHE[cache_key] = features_array
            logger.info(f"Loaded features array into RAM. Shape: {features_array.shape}")

        # Store the sample count so prepare_dataset can compute steps_per_epoch.
        self._last_loaded_n_samples = len(features_array)

        # Use from_generator (factory pattern) instead of from_tensor_slices:
        # from_tensor_slices converts the entire numpy array into a tf.constant,
        # which TF allocates on GPU during graph compilation — causing OOM for
        # large feature arrays (~937 MB for 30k train samples).
        # The generator factory (a callable that returns a fresh generator each
        # call) streams slices from CPU RAM on demand, so only one batch at a
        # time ever lives on the GPU.
        def gen():
            for slice_arr in features_array:
                yield slice_arr

        feature_dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature=tf.TensorSpec(
                shape=(h_val, w_val, self.feature_channels),
                dtype=tf.float32
            )
        )
        return feature_dataset

    def prepare_dataset(
        self,
        dataset: tf.data.Dataset,
        dataset_name: str | None = None,
        is_training: bool = False,
    ) -> tuple:
        """
        Combine an image dataset with feature maps, either from the cache
        (when *dataset_name* is given) or computed on-the-fly.

        Args:
            dataset:      Source tf.data.Dataset yielding (images, labels).
            dataset_name: If provided, load features from the pre-computed cache.
            is_training:  Whether to apply shuffling, batching, and data augmentation.

        Returns:
            Tuple of (prepared_dataset, steps_per_epoch). steps_per_epoch is an
            int when the dataset uses the feature cache (so model.fit can bound
            each epoch correctly), or None for on-the-fly / eval paths where
            Keras infers the step count from the dataset cardinality.
        """
        if dataset_name:
            feature_dataset = self.load_features(dataset_name) # Unbatched
            n_samples = getattr(self, '_last_loaded_n_samples', None)

            # Zip them unbatched so they are perfectly aligned in the original order!
            zipped = tf.data.Dataset.zip((dataset, feature_dataset))

            if is_training:
                # Make the zipped dataset infinite so Keras never sees it exhaust
                # mid-run. Without .repeat(), zip(image_ds, generator_ds) has
                # UNKNOWN cardinality and Keras 3 treats the whole multi-epoch
                # run as one long stream; when the generator exhausts after
                # epoch 1 it warns and resets inconsistently. With .repeat() we
                # supply steps_per_epoch so Keras knows exactly when each epoch
                # ends without relying on dataset exhaustion.
                zipped = zipped.repeat()

                # Apply data augmentation to images
                if config.use_augmentation:
                    def augment(inputs, features):
                        if isinstance(inputs, tuple):
                            img, lbl = inputs
                            return (_augment_image(img), lbl), features
                        return _augment_image(inputs), features
                    zipped = zipped.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

                # Shuffle the zipped dataset
                buffer_size = min(10000, config.cnn_batch_size * 100)
                zipped = zipped.shuffle(buffer_size=buffer_size, seed=config.random_seed)

            # Batch the zipped dataset
            zipped = zipped.batch(config.cnn_batch_size)

            def combine(inputs, feature_batch):
                # inputs: (images, labels) or images
                # feature_batch: [batch_size, n_patches_h, n_patches_w, feature_channels]
                
                # Scale features if feature_scales is available
                if hasattr(self, 'feature_scales') and self.feature_scales is not None:
                    feature_batch = feature_batch / self.feature_scales

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

            import math
            # Always return steps regardless of is_training so the caller can
            # use it to bound both training (steps_per_epoch) and validation
            # (validation_steps) when wrapping the dataset with .repeat().
            steps = math.ceil(n_samples / config.cnn_batch_size) if n_samples else None
            prepared = (
                zipped
                .map(combine, num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE)
            )
            return prepared, steps

        ds, _ = self._prepare_dataset_on_the_fly(dataset, is_training=is_training)
        return ds, None

    def _prepare_dataset_on_the_fly(
        self, dataset: tf.data.Dataset, is_training: bool = False
    ) -> tuple:
        """
        Fallback: compute features for each batch as it is consumed.
        Returns (dataset, None) — steps_per_epoch is None because the
        dataset's cardinality is known to Keras from the image dataset itself.
        """
        import logging
        logging.getLogger(__name__).info(
            "Preparing dataset with on-the-fly feature computation"
        )

        # Shuffle and batch the image dataset first
        if is_training:
            if config.use_augmentation:
                dataset = dataset.map(
                    lambda x, y: (_augment_image(x), y),
                    num_parallel_calls=tf.data.AUTOTUNE
                )
            buffer_size = min(10000, config.cnn_batch_size * 100)
            dataset = dataset.shuffle(buffer_size=buffer_size, seed=config.random_seed)
        
        dataset = dataset.batch(config.cnn_batch_size)

        pipeline = self  # capture for the closure

        def add_features_on_the_fly(
            images: tf.Tensor, labels: tf.Tensor
        ) -> tuple:
            images   = tf.cast(images, tf.float32)
            features = pipeline.extract_batch_features(images)
            return {"image_input": images, "feature_input": features}, labels

        return (
            dataset.map(
                add_features_on_the_fly,
                num_parallel_calls=tf.data.AUTOTUNE,
            ).prefetch(tf.data.AUTOTUNE),
            None
        )