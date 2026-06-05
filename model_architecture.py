# model_architecture.py
import json
import tensorflow as tf
import builtins
builtins.tf = tf
from tensorflow.keras import layers, models, Model, callbacks # type: ignore
import numpy as np
import os

import global_config as config
from feature_extractor import FeaturePipeline

import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Random Mask Generator
# ---------------------------------------------------------------------------

class RandomMaskGenerator:
    """
    Generates patch-level binary masks with the same sparsity distribution
    as the GA-optimised masks — the critical experimental control.

    Args:
        target_sparsity: Mean fraction of patches to select (e.g. 0.30 for 30 %).
                         Must be retrieved from an actual GA run
                         (``run_results['mask_sparsity_mean']``) and passed here;
                         it is **not** hardcoded.
        random_seed:     Optional integer seed for reproducible masks.
    """

    def __init__(self, target_sparsity: float, random_seed: int | None = None):
        if not 0.0 <= target_sparsity <= 1.0:
            raise ValueError(
                f"target_sparsity must be in [0, 1], got {target_sparsity}"
            )
        self.target_sparsity = tf.constant(target_sparsity, dtype=tf.float32)
        self.random_seed = random_seed

    def generate(self, n_patches_h, n_patches_w) -> tf.Tensor:
        """
        Produce a single random binary mask of shape (n_patches_h, n_patches_w).

        The expected fraction of 1-entries equals ``target_sparsity``.
        Each call is independently random (different per image).

        Returns:
            int8 tensor of shape (n_patches_h, n_patches_w).
        """
        shape = tf.stack([n_patches_h, n_patches_w])
        flat = tf.random.uniform(shape, seed=self.random_seed, dtype=tf.float32)
        binary = tf.cast(flat < self.target_sparsity, tf.int8)
        return binary

    @tf.function
    def generate_stateless(self, batch_size, n_patches_h, n_patches_w, seed) -> tf.Tensor:
        """
        Produce a batch of random binary masks statelessly (avoiding RNG lock contention).
        """
        shape = tf.stack([batch_size, n_patches_h, n_patches_w])
        flat = tf.random.stateless_uniform(shape, seed=seed, dtype=tf.float32)
        binary = tf.cast(flat < self.target_sparsity, tf.int8)
        return binary


# ---------------------------------------------------------------------------
# Random Mask Pipeline  (FeaturePipeline subclass for mask_mode='random')
# ---------------------------------------------------------------------------

class _RandomMaskPipeline(FeaturePipeline):
    """
    A FeaturePipeline variant that replaces GA dynamic masks with per-image
    random masks produced by a RandomMaskGenerator.  All other FeaturePipeline
    infrastructure (feature extraction, caching, dataset preparation) is reused
    unchanged so that the three model modes produce identical downstream output.
    """

    def __init__(
        self,
        random_generator: RandomMaskGenerator,
        n_patches_h,
        n_patches_w,
        feature_cache_dir: str,
        feature_channels: int = 8,
    ):
        # Pass genetic_rules=None; it will never be used because we override _generate_mask.
        super().__init__(
            genetic_rules=None,
            n_patches_h=n_patches_h,
            n_patches_w=n_patches_w,
            feature_cache_dir=feature_cache_dir,
            feature_channels=feature_channels,
        )
        self._random_generator = random_generator

    def _generate_mask(self, patch_features: tf.Tensor) -> tf.Tensor:
        """Override: produce a batch of random binary masks statelessly."""
        batch_size = tf.shape(patch_features)[0]
        # Generate a seed from the patch features to make it stateless but pseudo-random per batch
        feature_sum = tf.reduce_sum(patch_features)
        seed_val = tf.cast(tf.abs(feature_sum * 1000.0), tf.int32)
        base_seed = config.random_seed if config.random_seed is not None else 42
        seed = tf.stack([seed_val + base_seed, seed_val + base_seed + 13])
        return self._random_generator.generate_stateless(batch_size, self.n_patches_h, self.n_patches_w, seed)


# Check if mixed precision is available and supported
try:
    # Test if mixed precision works on this hardware
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    logger.info("Mixed precision (mixed_float16) enabled successfully")
except Exception as e:
    logger.warning(f"Mixed precision not available or supported: {e}")
    logger.info("Falling back to float32 precision")
    policy = tf.keras.mixed_precision.Policy('float32')
    tf.keras.mixed_precision.set_global_policy(policy)

class AIDetectorModel:
    """
    Class containing model architecture and training functionality for detecting AI-generated images.
    Can be configured to use image inputs only or combined with feature maps from feature extraction.
    """
    def __init__(self, feature_channels=None):
        """
        Initialize the model architecture.
        
        Args:
            feature_channels (int): Number of feature channels from feature extraction
        """
        if feature_channels is None:
            feature_channels = len(config.feature_weights)
        logger.info("Initializing AIDetectorModel...")
        self.input_shape = (config.image_size, config.image_size, 3)
        self.feature_channels = feature_channels
        self.use_features = config.use_feature_extraction
        self.model = self._build_model()
        logger.info("AIDetectorModel initialized.")
        
    def _build_model(self):
        """
        Build the model architecture, either with or without feature extraction.

        Returns:
            tf.keras.Model: The compiled model
        """
        logger.info(f"Building model architecture. Use features: {self.use_features}")

        # Determine compute dtype based on global policy
        compute_dtype = tf.keras.mixed_precision.global_policy().compute_dtype
        variable_dtype = tf.keras.mixed_precision.global_policy().variable_dtype

        logger.info(f"Using compute dtype: {compute_dtype}, variable dtype: {variable_dtype}")

        # Setup L2 regularizer
        reg = tf.keras.regularizers.l2(config.cnn_l2_reg) if config.cnn_l2_reg > 0 else None

        # Main input for the image - always use float32 for inputs
        image_input = layers.Input(shape=self.input_shape, name='image_input', dtype='float32')

        # Image processing branch
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(image_input)
        x = layers.MaxPooling2D((2, 2), dtype='float32')(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(x)
        x = layers.MaxPooling2D((2, 2), dtype='float32')(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(x)
        x = layers.MaxPooling2D((2, 2), dtype='float32')(x)
        x = layers.BatchNormalization()(x)
        
        spatial_size = self.input_shape[0] // 8
        
        if self.use_features:
            logger.info("Building model with feature extraction branch.")
            # Feature map input - always use float32 for inputs
            feature_input = layers.Input(shape=(self.input_shape[0], self.input_shape[1], self.feature_channels), 
                                         name='feature_input', dtype='float32')
            
             # Process feature maps
            f = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(feature_input)
            f = layers.MaxPooling2D((2, 2), dtype='float32')(f)
            f = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(f)
            f = layers.MaxPooling2D((2, 2), dtype='float32')(f)
            f = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(f)
            f = layers.MaxPooling2D((2, 2), dtype='float32')(f)

            # Resize feature maps to fixed spatial size
            f = layers.Resizing(spatial_size, spatial_size, interpolation='bilinear')(f)
            
            # Cast feature maps to compute dtype using Identity layer
            f = layers.Identity(dtype=compute_dtype)(f)
            
            # Global attention weights
            attn_weights = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same', kernel_regularizer=reg)(x)
            # Apply attention to feature maps
            f_weighted = layers.Multiply()([f, attn_weights])
            # Cross-attention
            attention = layers.Multiply()([x, f_weighted])
            # Combine original with attention-weighted
            enhanced = layers.Add()([x, attention])
            logger.debug("Attention mechanism applied.")
            
            # Continue with the CNN
            x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(enhanced)
        else:
            logger.info("Building model without feature extraction branch.")
            # Without feature extraction, add an extra convolutional layer to maintain model capacity
            x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(x)

        # Common downstream layers
        x = layers.MaxPooling2D((2, 2), dtype='float32')(x)
        x = layers.BatchNormalization()(x)
        
        # Global Average Pooling instead of Flatten to ensure fully defined shape
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(config.cnn_dropout_1)(x)
        x = layers.Dense(config.cnn_dense_units, activation='relu', kernel_regularizer=reg)(x)
        x = layers.Dropout(config.cnn_dropout_2)(x)

        # Output layer must use float32 for numerical stability
        output = layers.Dense(1, activation='sigmoid', dtype='float32', kernel_regularizer=reg)(x)

        # Create model with appropriate inputs
        if self.use_features:
            model = Model(inputs=[image_input, feature_input], outputs=output)
        else:
            model = Model(inputs=image_input, outputs=output)

        # Dynamic optimizer setup based on HPO config
        opt_name = config.cnn_optimizer.lower()
        lr = config.cnn_learning_rate
        if opt_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        elif opt_name == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
        elif opt_name == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # Only use LossScaleOptimizer if we're actually using mixed precision
        if tf.keras.mixed_precision.global_policy().compute_dtype == 'float16':
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
            logger.info("Using LossScaleOptimizer for mixed precision training")

        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        logger.info("Model compilation complete.")

        return model

    def train(self, train_dataset, validation_dataset=None, output_model_path='human_ai_detector_model.h5'):
        """
        Train the model with callbacks for early stopping and learning rate reduction.
        
        Args:
            train_dataset (tf.data.Dataset): Training dataset
            validation_dataset (tf.data.Dataset): Validation dataset
            output_model_path (str): Path to save the best model

        Returns:
            dict: Training history
        """
        logger.info("Starting model training...")
        # Set up callbacks monitoring validation accuracy dynamically
        patience = getattr(config, 'cnn_early_stopping_patience', 10)
        reduce_lr_patience = max(2, patience // 2)
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                mode='max',
                patience=patience,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                mode='max',
                factor=0.5,
                patience=reduce_lr_patience
            ),
            callbacks.ModelCheckpoint(
                output_model_path,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True
            )
        ]
        
        if config.profile:
            tb_callback = callbacks.TensorBoard(
                log_dir=os.path.join(config.profile_log_dir, 'training'),
                profile_batch=(2, 12),  # Profile from batch 2 to 12
                update_freq='epoch'
            )
            callbacks_list.append(tb_callback)
            logger.info("TensorBoard profiling callback added for training.")

        logger.debug("Callbacks set up for training.")
        
        # Train the model
        history = self.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=config.epochs,
            callbacks=callbacks_list
        )
        logger.info("Model training complete.")
        
        return history
    
    def evaluate(self, test_dataset):
        """
        Evaluate the model on test data.
        
        Args:
            test_dataset (tf.data.Dataset): Test dataset
            
        Returns:
            tuple: (test_loss, test_accuracy)
        """
        logger.info("Evaluating model on test dataset...")
        results = self.model.evaluate(test_dataset)
        logger.info(f"Model evaluation complete. Loss: {results[0]:.4f}, Accuracy: {results[1]:.4f}")
        return results
    
    def predict(self, inputs):
        """
        Make predictions on new inputs.
        
        Args:
            inputs: Could be any of:
                - np.ndarray of images
                - dict with 'image_input' and 'feature_input' keys
                - TensorFlow dataset
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        logger.info("Generating predictions...")
        predictions = self.model.predict(inputs)
        logger.info("Predictions generated.")
        return predictions
    
    def save(self, filepath):
        """
        Save the model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        logger.info(f"Saving model to {filepath}...")
        self.model.save(filepath)
        logger.info("Model saved successfully.")
    
    def load(self, filepath):
        """
        Load a saved model from disk.
        
        Args:
            filepath (str): Path to the saved model
        """
        logger.info(f"Loading model from {filepath}...")
        try:
            tf.keras.config.enable_unsafe_deserialization()
        except AttributeError:
            pass
        self.model = tf.keras.models.load_model(filepath)
        # Determine if this is a feature-using model or not by checking the input shape
        self.use_features = len(self.model.inputs) > 1
        logger.info(f"Model loaded successfully. Use features: {self.use_features}")


def _augment_image(image):
    # Random flips
    image = tf.image.random_flip_left_right(image)
    # Random brightness and contrast adjustments
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    # Clip to [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    image.set_shape([config.image_size, config.image_size, 3])
    return image


class ModelWrapper:
    """
    Wrapper class that integrates genetic algorithm dynamic masks with the CNN model.
    Feature extraction, caching, and dataset preparation are delegated to FeaturePipeline.

    Three mask modes are supported:
        ``"none"``   – Baseline CNN (no feature extraction / masking).
        ``"ga"``     – GA-generated dynamic masks (existing implementation).
        ``"random"`` – Random masks matching the GA sparsity distribution.
    """

    def __init__(
        self,
        feature_channels: int = None,
        genetic_rules: tf.Tensor = None,
        mask_mode: str | None = None,
        random_mask_sparsity: float | None = None,
    ):
        """
        Args:
            feature_channels:     Number of feature channels produced by the feature extractor.
            genetic_rules:        Evolved rules from the genetic algorithm (only used when
                                  ``mask_mode == 'ga'``).
            mask_mode:            One of ``"none"``, ``"ga"``, or ``"random"``.
                                  Defaults to ``config.mask_mode`` (or ``"ga"`` if unset).
            random_mask_sparsity: Fraction of patches to select when ``mask_mode == 'random'``.
                                  Must be provided (or be available on the config object) when
                                  ``mask_mode == 'random'``.
        """
        logger.info("Initializing ModelWrapper...")

        # Resolve feature_channels lazily so we don't hit AttributeError at import time
        if feature_channels is None:
            feature_channels = len(config.feature_weights)

        # Resolve mask mode from argument > config > default
        if mask_mode is None:
            mask_mode = getattr(config, 'mask_mode', 'ga')
        self.mask_mode = mask_mode

        self.input_shape      = (config.image_size, config.image_size, 3)
        self.feature_channels = feature_channels
        self.patch_size       = config.patch_size

        # use_features is True for both "ga" and "random" modes
        self.use_features = self.mask_mode in ('ga', 'random')
        # Keep a copy on config so AIDetectorModel picks it up correctly
        config.use_feature_extraction = self.use_features

        self.genetic_rules = (
            tf.convert_to_tensor(genetic_rules, dtype=tf.float32)
            if genetic_rules is not None else None
        )

        self.n_patches_h = tf.constant(self.input_shape[0] // self.patch_size, dtype=tf.int32)
        self.n_patches_w = tf.constant(self.input_shape[1] // self.patch_size, dtype=tf.int32)

        self.feature_cache_dir = os.path.join(config.output_dir, config.feature_cache_dir)

        # --- Build the appropriate feature pipeline ---
        if self.mask_mode == 'random':
            # Resolve sparsity: argument takes precedence, then config attribute
            if random_mask_sparsity is None:
                random_mask_sparsity = getattr(config, '_random_mask_sparsity', None)
            if random_mask_sparsity is None:
                raise ValueError(
                    "mask_mode='random' requires random_mask_sparsity to be provided. "
                    "Pass it explicitly or store it in config._random_mask_sparsity."
                )
            self._random_generator = RandomMaskGenerator(
                target_sparsity=random_mask_sparsity,
                random_seed=config.random_seed,
            )
            self.pipeline = _RandomMaskPipeline(
                random_generator=self._random_generator,
                n_patches_h=self.n_patches_h,
                n_patches_w=self.n_patches_w,
                feature_cache_dir=self.feature_cache_dir,
                feature_channels=feature_channels,
            )
        else:
            self._random_generator = None
            self.pipeline = FeaturePipeline(
                genetic_rules=self.genetic_rules,
                n_patches_h=self.n_patches_h,
                n_patches_w=self.n_patches_w,
                feature_cache_dir=self.feature_cache_dir,
                feature_channels=feature_channels,
            )

        self.model = AIDetectorModel(feature_channels=feature_channels)
        logger.info(f"ModelWrapper initialized. mask_mode='{self.mask_mode}'")


    def set_genetic_rules(self, genetic_rules: tf.Tensor) -> None:
        """Update genetic rules in both the wrapper and the pipeline."""
        self.genetic_rules = tf.convert_to_tensor(genetic_rules, dtype=tf.float32)
        self.pipeline.set_genetic_rules(self.genetic_rules)
        logger.info(f"Updated genetic rules. Shape: {tf.shape(self.genetic_rules)}")

    def _ensure_feature_extractor(self) -> None:
        self.pipeline._ensure_feature_extractor()

    def extract_batch_features(self, images: tf.Tensor) -> tf.Tensor:
        """Extract masked pixel-level feature maps for a batch of images."""
        logger.info("Extracting batch features using genetic dynamic masks.")
        return self.pipeline.extract_batch_features(images)

    def extract_single_image_features(self, image: tf.Tensor) -> tf.Tensor:
        """Extract features for a single image tensor."""
        if len(image.shape) == 3:
            image = tf.expand_dims(image, axis=0)
        return self.pipeline.extract_single_image_features(image[0])

    def precompute_features(
        self, dataset: tf.data.Dataset, dataset_name: str
    ) -> None:
        """Precompute and cache features for an entire dataset."""
        self.pipeline.precompute_features(dataset, dataset_name)

    def load_features(self, dataset_name: str) -> tf.data.Dataset:
        """Load pre-computed features from the cache as a tf.data.Dataset."""
        return self.pipeline.load_features(dataset_name)

    def prepare_dataset(
        self,
        dataset: tf.data.Dataset,
        dataset_name: str | None = None,
        is_training: bool = False,
    ) -> tf.data.Dataset:
        """Prepare dataset for training/evaluation (adding features, shuffling, batching)."""
        # If not using features, just shuffle (if training) and batch
        if not self.use_features:
            if is_training:
                if config.use_augmentation:
                    dataset = dataset.map(
                        lambda x, y: (_augment_image(x), y),
                        num_parallel_calls=tf.data.AUTOTUNE
                    )
                buffer_size = min(10000, config.cnn_batch_size * 100)
                dataset = dataset.shuffle(buffer_size=buffer_size, seed=config.random_seed)
            dataset = dataset.batch(config.cnn_batch_size)
            return dataset.prefetch(tf.data.AUTOTUNE)

        # If using features, delegate to pipeline
        return self.pipeline.prepare_dataset(dataset, dataset_name, is_training=is_training)

    def train(self, train_dataset, validation_dataset=None, model_path='ai_detector_model.keras', precompute_features=None):
        """Training method with index-based feature caching"""
        logger.info("Starting model wrapper training.")
        
        # Resolve precompute_features from argument > config > default
        if precompute_features is None:
            precompute_features = getattr(config, 'use_feature_cache', True)
        
        if self.use_features and precompute_features:
            # Precompute features
            self.precompute_features(train_dataset, "train")
            if validation_dataset:
                self.precompute_features(validation_dataset, "val")
            
            # Prepare datasets
            prepared_train = self.prepare_dataset(train_dataset, "train", is_training=True)
            prepared_val = self.prepare_dataset(validation_dataset, "val", is_training=False) if validation_dataset else None
        else:
            # Fallback to on-the-fly
            logger.info("Using on-the-fly feature extraction (no disk cache).")
            prepared_train = self.prepare_dataset(train_dataset, is_training=True)
            prepared_val = self.prepare_dataset(validation_dataset, is_training=False) if validation_dataset else None
        
        # Train the model
        history = self.model.train(
            prepared_train,
            validation_dataset=prepared_val,
            output_model_path=model_path
        )
        return history

    def evaluate(self, test_dataset):
        logger.info("Evaluating model wrapper on test dataset.")
        use_cache = getattr(config, 'use_feature_cache', True)
        
        if self.use_features:
            if use_cache:
                logger.info("Precomputing and preparing test dataset with genetic features.")
                self.precompute_features(test_dataset, "test")
                prepared_test_ds = self.prepare_dataset(test_dataset, "test")
            else:
                logger.info("Preparing test dataset with on-the-fly genetic features.")
                prepared_test_ds = self.prepare_dataset(test_dataset)
        else:
            logger.info("Evaluating without genetic features.")
            prepared_test_ds = self.prepare_dataset(test_dataset, is_training=False)
        results = self.model.evaluate(prepared_test_ds)
        logger.info(f"Model wrapper evaluation complete. Loss: {results[0]:.4f}, Accuracy: {results[1]:.4f}")
        return results
        
    def predict_image(self, image):
        """Predict single image with on-the-fly features"""
        logger.info("Making prediction for a single image.")
        if not isinstance(image, tf.Tensor):
            image = tf.convert_to_tensor(image, dtype=tf.float32)
        
        if len(image.shape) == 3:
            image = tf.expand_dims(image, axis=0)
        
        if self.use_features:
            features = tf.expand_dims(self.extract_single_image_features(image[0]), 0)
            inputs = {'image_input': image, 'feature_input': features}
        else:
            inputs = image
        
        prediction = self.model.predict(inputs)[0][0]
        is_ai = prediction > 0.5
        confidence = prediction if is_ai else 1.0 - prediction
        
        return {
            'is_ai': bool(is_ai),
            'label': "AI-generated" if is_ai else "Human-generated",
            'confidence': float(confidence),
            'raw_prediction': float(prediction)
        }
    
    def save(self, base_path):
        """
        Save the complete model state including neural network and genetic rules.
        
        Args:
            base_path (str): Base path for saving (without extension)
        """
        logger.info(f"Saving complete model state to {base_path}.*")
        
        # Save the neural network model
        model_path = f"{base_path}.keras"
        self.model.save(model_path)
        
        # Save genetic rules and configuration
        config_data = {
            'use_features': self.use_features,
            'feature_channels': self.feature_channels,
            'mask_mode': self.mask_mode,
            'genetic_rules': self.genetic_rules.numpy().tolist() if self.genetic_rules is not None else None,
            'random_mask_sparsity': (
                float(self._random_generator.target_sparsity.numpy())
                if self._random_generator is not None else None
            ),
        }
        
        config_path = f"{base_path}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        logger.info(f"Model state saved: {model_path}, {config_path}")


    @classmethod
    def load(cls, base_path):
        """
        Load the complete model state and create a new ModelWrapper instance.
        
        Args:
            base_path (str): Base path for loading (without extension)
            
        Returns:
            ModelWrapper: New ModelWrapper instance with loaded state
        """
        logger.info(f"Loading complete model state from {base_path}.*")
        
        # Load configuration
        config_path = f"{base_path}_config.json"
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Create genetic rules tensor
        genetic_rules = None
        if config_data.get('genetic_rules') is not None:
            genetic_rules = tf.convert_to_tensor(config_data['genetic_rules'], dtype=tf.float32)
        
        # Create new ModelWrapper instance via the class (respects subclassing)
        model_wrapper = cls(
            feature_channels=config_data['feature_channels'],
            genetic_rules=genetic_rules,
            mask_mode=config_data.get('mask_mode', 'ga'),
            random_mask_sparsity=config_data.get('random_mask_sparsity'),
        )
        
        # Load the neural network model
        model_path = f"{base_path}_model.keras"
        if not os.path.exists(model_path) and os.path.exists(f"{base_path}.keras"):
            model_path = f"{base_path}.keras"
        model_wrapper.model.load(model_path)

        model_wrapper.use_features = config_data.get('use_features', model_wrapper.model.use_features)
        
        logger.info(f"Model state loaded from: {model_path}, {config_path}")
        return model_wrapper
