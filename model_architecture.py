
# model_architecture.py
import json
import tensorflow as tf
from tensorflow.keras import layers, models, Model, callbacks # type: ignore
import numpy as np
import os
import hashlib

import global_config as config
import utils

import logging
logger = logging.getLogger(__name__)

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
    def __init__(self, feature_channels=8):
        """
        Initialize the model architecture.
        
        Args:
            feature_channels (int): Number of feature channels from feature extraction
        """
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

        # Main input for the image - always use float32 for inputs
        image_input = layers.Input(shape=self.input_shape, name='image_input', dtype='float32')

        # Image processing branch
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
        x = layers.MaxPooling2D((2, 2), dtype='float32')(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), dtype='float32')(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), dtype='float32')(x)
        x = layers.BatchNormalization()(x)
        
        spatial_size = self.input_shape[0] // 8
        
        if self.use_features:
            logger.info("Building model with feature extraction branch.")
            # Feature map input - always use float32 for inputs
            feature_input = layers.Input(shape=(self.input_shape[0], self.input_shape[1], self.feature_channels), 
                                        name='feature_input', dtype='float32')
            
             # Process feature maps
            f = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(feature_input)
            f = layers.MaxPooling2D((2, 2), dtype='float32')(f)
            f = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(f)
            f = layers.MaxPooling2D((2, 2), dtype='float32')(f)
            f = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(f)
            f = layers.MaxPooling2D((2, 2), dtype='float32')(f)

            # Resize feature maps to fixed spatial size
            f = layers.Resizing(spatial_size, spatial_size, interpolation='bilinear')(f)
            
            # Create custom layer for dtype conversion
            class CastToComputeDtype(layers.Layer):
                def __init__(self, dtype, **kwargs):
                    super().__init__(**kwargs)
                    self.dtype = dtype
                def call(self, inputs):
                    return tf.cast(inputs, self.dtype)
                def compute_output_shape(self, input_shape):
                    return input_shape
            
            # Cast feature maps to compute dtype
            f = layers.Lambda(lambda x: tf.cast(x, compute_dtype))(f)
            
            # Global attention weights
            attn_weights = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)
            # Apply attention to feature maps
            f_weighted = layers.Multiply()([f, attn_weights])
            # Cross-attention
            attention = layers.Multiply()([x, f_weighted])
            # Combine original with attention-weighted
            enhanced = layers.Add()([x, attention])
            logger.debug("Attention mechanism applied.")
            
            # Continue with the CNN
            x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(enhanced)
        else:
            logger.info("Building model without feature extraction branch.")
            # Without feature extraction, add an extra convolutional layer to maintain model capacity
            x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)

        # Common downstream layers
        x = layers.MaxPooling2D((2, 2), dtype='float32')(x)
        x = layers.BatchNormalization()(x)
        
        # Global Average Pooling instead of Flatten to ensure fully defined shape
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(config.image_size, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

        # Output layer must use float32 for numerical stability
        output = layers.Dense(1, activation='sigmoid', dtype='float32')(x)

        # Create model with appropriate inputs
        if self.use_features:
            model = Model(inputs=[image_input, feature_input], outputs=output)
        else:
            model = Model(inputs=image_input, outputs=output)

        # Simplified optimizer setup for mixed precision
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

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
        # Set up callbacks
        callbacks_list = [
            callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            callbacks.ModelCheckpoint(output_model_path, save_best_only=True)
        ]
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
        self.model = tf.keras.models.load_model(filepath)
        # Determine if this is a feature-using model or not by checking the input shape
        self.use_features = len(self.model.inputs) > 1
        logger.info(f"Model loaded successfully. Use features: {self.use_features}")


class ModelWrapper:
    """
    Wrapper class that integrates genetic algorithm dynamic masks with the CNN model.
    Handles feature extraction using the evolved genetic rules and manages model training.
    """
    def __init__(self, feature_channels=8, genetic_rules: tf.Tensor= None):
        """
        Initialize the model wrapper with genetic algorithm integration.
        
        Args:
            feature_channels (int): Number of feature channels from feature extraction
            genetic_rules (tf.Tensor, optional): Evolved rules from genetic algorithm as a TensorFlow tensor.
        """
        logger.info("Initializing ModelWrapper...")
        self.input_shape = (config.image_size, config.image_size, 3)
        self.use_features = config.use_feature_extraction
        self.feature_channels = feature_channels
        self.patch_size = config.patch_size
        self.genetic_rules = tf.convert_to_tensor(genetic_rules, dtype=tf.float32) if genetic_rules is not None else None
        
        # Calculate patch grid dimensions using TensorFlow operations
        self.n_patches_h = tf.constant(self.input_shape[0] // self.patch_size, dtype=tf.int32)
        self.n_patches_w = tf.constant(self.input_shape[1] // self.patch_size, dtype=tf.int32)

        # Initialize the model
        self.model = AIDetectorModel(feature_channels=feature_channels)

        # Feature extraction components will be initialized when needed
        self.feature_extractor = None
        
        # Precomputed feature maps cache
        self.feature_cache_dir = os.path.join(config.output_dir, config.feature_cache_dir)
        os.makedirs(self.feature_cache_dir, exist_ok=True)
        
        logger.info("ModelWrapper initialized.")
        
    def set_genetic_rules(self, genetic_rules: tf.Tensor):
        """
        Set or update the genetic algorithm rules for dynamic mask generation.
        
        Args:
            genetic_rules (tf.Tensor): Evolved rules from genetic algorithm as a TensorFlow tensor.
        """
        self.genetic_rules = tf.convert_to_tensor(genetic_rules, dtype=tf.float32)
        logger.info(f"Updated genetic rules. Shape: {tf.shape(self.genetic_rules)}")

        # Clear precomputed feature maps when rules change since they depend on genetic rules
        self.precomputed_feature_maps = {}
        logger.debug("Precomputed feature maps cleared due to rule change.")
            
    def _ensure_feature_extractor(self):
        """Initialize the feature extractor if not already done"""
        if self.feature_extractor is None:
            from feature_extractor import FeatureExtractor
            self.feature_extractor = FeatureExtractor()
            logger.info("Feature extractor initialized.")
    
    def extract_batch_features(self, images):
        """Extract features without caching (on-the-fly)"""
        logger.info("Extracting batch features using genetic dynamic masks.")
        self._ensure_feature_extractor()
        
        # Convert to tensor if needed
        if not isinstance(images, tf.Tensor):
            images = tf.convert_to_tensor(images, dtype=tf.float32)
        
        # Extract patch features
        patch_features = self.feature_extractor.extract_batch_patch_features(images)
        
        # Generate masks for all images in batch
        batch_masks = []
        for i in range(images.shape[0]):
            mask = utils.generate_dynamic_mask(
                patch_features[i],
                self.n_patches_h,
                self.n_patches_w,
                self.genetic_rules
            )
            batch_masks.append(mask)
        
        # Convert to pixel masks with batch support
        pixel_mask = utils.convert_patch_mask_to_pixel_mask(tf.stack(batch_masks))
        
        # Ensure proper dimensions for broadcasting
        if pixel_mask.shape.rank == 3:  # Batch of 2D masks
            pixel_mask = tf.expand_dims(pixel_mask, axis=-1)
        elif pixel_mask.shape.rank == 2:  # Single 2D mask
            pixel_mask = tf.expand_dims(tf.expand_dims(pixel_mask, 0), axis=-1)
        
        # Resize feature maps and apply mask
        resized_feature_maps = tf.image.resize(
            patch_features,
            [config.image_size, config.image_size],
            method=tf.image.ResizeMethod.BILINEAR
        )
        
        return resized_feature_maps * pixel_mask
        
    def extract_single_image_features(self, image):
        """Public wrapper for single image feature extraction"""
        if len(image.shape) == 3:
            image = tf.expand_dims(image, axis=0)
        return self._extract_single_image_features(image)[0]

    def precompute_features(self, dataset, dataset_name):
        """Precompute and save features for a dataset"""
        logger.info(f"Precomputing features for {dataset_name}")
        save_dir = os.path.join(self.feature_cache_dir, dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        
        index = 0
        for batch in dataset:
            images = batch[0] if isinstance(batch, tuple) else batch
            features = self.extract_batch_features(images)
            
            for i in range(features.shape[0]):
                feature = features[i].numpy()
                np.save(os.path.join(save_dir, f"feature_{index}.npy"), feature)
                index += 1
        
        logger.info(f"Saved {index} features for {dataset_name}")

    def load_features(self, dataset_name):
        """Load precomputed features into a dataset"""
        logger.info(f"Loading precomputed features for {dataset_name}")
        save_dir = os.path.join(self.feature_cache_dir, dataset_name)
        feature_paths = sorted(
            [os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.endswith('.npy')],
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )

        def load_feature(path):
            return np.load(path.decode('utf-8'))
        
        feature_dataset = tf.data.Dataset.from_tensor_slices(feature_paths)
        feature_dataset = feature_dataset.map(
            lambda path: tf.numpy_function(load_feature, [path], tf.float32),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        # Ensure correct shape
        feature_dataset = feature_dataset.map(
            lambda x: tf.ensure_shape(x, [config.image_size, config.image_size, self.feature_channels])
        )
        return feature_dataset

    def prepare_dataset(self, dataset, dataset_name=None):
        """Prepare dataset with precomputed features"""
        if not self.use_features:
            return dataset
        
        # For training/validation
        if dataset_name:
            feature_dataset = self.load_features(dataset_name)
            
            def combine(inputs, feature):
                # Add batch dimension to feature tensor
                feature = tf.expand_dims(feature, axis=0)
                
                if isinstance(inputs, tuple):
                    image, label = inputs
                    # Ensure feature matches batch size
                    feature = tf.repeat(feature, tf.shape(image)[0], axis=0)
                    return {'image_input': image, 'feature_input': feature}, label
                return {'image_input': inputs, 'feature_input': feature}
            
            # Create combined dataset
            combined = tf.data.Dataset.zip((dataset, feature_dataset))
            return combined.map(
                combine, 
                num_parallel_calls=tf.data.AUTOTUNE
            ).prefetch(tf.data.AUTOTUNE)
        
        # For prediction without precomputation
        return self._prepare_dataset_on_the_fly(dataset)
    
    def _prepare_dataset_on_the_fly(self, dataset):
        """
        Fallback method to prepare dataset with on-the-fly feature computation
        """
        if not self.use_features:
            return dataset
        
        logger.info("Preparing dataset with on-the-fly feature computation")
        
        def add_features_on_the_fly(images, labels):
            """Compute features on-the-fly for each batch"""
            images = tf.cast(images, tf.float32)
            
            # Extract features using the existing method
            features = self.extract_batch_features(images)
            
            return {'image_input': images, 'feature_input': features}, labels
        
        return dataset.map(
            add_features_on_the_fly,
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)

    def train(self, train_dataset, validation_dataset=None, model_path='ai_detector_model.h5', precompute_features=True):
        """Training method with index-based feature caching"""
        logger.info("Starting model wrapper training.")
        
        if self.use_features and precompute_features:
            # Precompute features
            self.precompute_features(train_dataset, "train")
            if validation_dataset:
                self.precompute_features(validation_dataset, "val")
            
            # Prepare datasets
            prepared_train = self.prepare_dataset(train_dataset, "train")
            prepared_val = self.prepare_dataset(validation_dataset, "val") if validation_dataset else None
        else:
            # Fallback to on-the-fly
            prepared_train = self.prepare_dataset(train_dataset)
            prepared_val = self.prepare_dataset(validation_dataset) if validation_dataset else None
        
        # Train the model
        history = self.model.train(
            prepared_train,
            validation_dataset=prepared_val,
            output_model_path=model_path
        )
        return history

    def evaluate(self, test_dataset):
        """
        Evaluate the model on test data.
        
        Args:
            test_dataset (tf.data.Dataset): Test dataset
            
        Returns:
            tuple: (test_loss, test_accuracy)
        """
        logger.info("Evaluating model wrapper on test dataset.")
        # Prepare test dataset with features if needed
        if self.use_features:
            logger.info("Preparing test dataset with genetic features.")
            prepared_test_ds = self.prepare_dataset(test_dataset)
        else:
            logger.info("Evaluating without genetic features. Using original test dataset.")
            prepared_test_ds = test_dataset
            
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
    
    @staticmethod
    def save_model_state(model_wrapper, base_path):
        """
        Save the complete model state including neural network and genetic rules.
        
        Args:
            model_wrapper (ModelWrapper): The model wrapper instance to save
            base_path (str): Base path for saving (without extension)
        """
        logger.info(f"Saving complete model state to {base_path}.*")
        
        # Save the neural network model
        model_path = f"{base_path}_model.h5"
        model_wrapper.model.save(model_path)
        
        # Save genetic rules and configuration
        config_data = {
            'use_features': model_wrapper.use_features,
            'feature_channels': model_wrapper.feature_channels,
            'genetic_rules': model_wrapper.genetic_rules.numpy().tolist() if model_wrapper.genetic_rules is not None else None
        }
        
        config_path = f"{base_path}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        logger.info(f"Model state saved: {model_path}, {config_path}")

    @staticmethod
    def load_model_state(base_path):
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
        if config_data['genetic_rules'] is not None:
            genetic_rules = tf.convert_to_tensor(config_data['genetic_rules'], dtype=tf.float32)
        
        # Create new ModelWrapper instance
        model_wrapper = ModelWrapper(
            feature_channels=config_data['feature_channels'],
            genetic_rules=genetic_rules
        )
        
        # Load the neural network model
        model_path = f"{base_path}_model.h5"
        model_wrapper.model.load(model_path)
        
        logger.info(f"Model state loaded from: {model_path}, {config_path}")
        return model_wrapper