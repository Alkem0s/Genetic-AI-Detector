# model_architecture.py
import json
import tensorflow as tf
from tensorflow.keras import layers, models, Model, callbacks # type: ignore
import numpy as np
import os

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
        
        # CNN for image processing - let layers use default policy dtypes
        x = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
        x = layers.MaxPooling2D((2, 2), dtype='float32')(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2), dtype='float32')(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2), dtype='float32')(x)
        x = layers.BatchNormalization()(x)
        
        if self.use_features:
            logger.info("Building model with feature extraction branch.")
            # Feature map input - always use float32 for inputs
            feature_input = layers.Input(shape=(self.input_shape[0], self.input_shape[1], self.feature_channels), 
                                        name='feature_input', dtype='float32')
            
            # Process feature maps
            f = layers.Conv2D(32, (3, 3), activation='relu')(feature_input)
            f = layers.MaxPooling2D((2, 2), dtype='float32')(f)
            f = layers.Conv2D(64, (3, 3), activation='relu')(f)
            f = layers.MaxPooling2D((2, 2), dtype='float32')(f)
            f = layers.Conv2D(128, (3, 3), activation='relu')(f)
            f = layers.MaxPooling2D((2, 2), dtype='float32')(f)

            class ResizeAndAlign(layers.Layer):
                def __init__(self, **kwargs):
                    super(ResizeAndAlign, self).__init__(**kwargs)
                    self._compute_dtype = tf.keras.mixed_precision.global_policy().compute_dtype
                
                def call(self, inputs):
                    tensor, target_tensor = inputs
                    target_shape = tf.shape(target_tensor)[1:3]
                    tensor = tf.cast(tensor, self._compute_dtype)
                    resized = tf.image.resize(tensor, target_shape)
                    return tf.cast(resized, self._compute_dtype)
                
                def compute_output_shape(self, input_shape):
                    tensor_shape, target_shape = input_shape
                    return (tensor_shape[0], None, None, tensor_shape[3])
            
            # Ensure both tensors have same shape and dtype
            f = ResizeAndAlign()([f, x])
            
            logger.debug("Feature branch resized for concatenation.")
            
            # Global attention weights
            attn_weights = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
            # Apply attention to feature maps
            f_weighted = layers.Multiply()([f, attn_weights])
            # Cross-attention
            attention = layers.Multiply()([x, f_weighted])
            # Combine original with attention-weighted
            enhanced = layers.Add()([x, attention])
            logger.debug("Attention mechanism applied.")
            
            # Continue with the CNN
            x = layers.Conv2D(128, (3, 3), activation='relu')(enhanced)
        else:
            logger.info("Building model without feature extraction branch.")
            # Without feature extraction, add an extra convolutional layer to maintain model capacity
            x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        
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
        
        # Get the full path to feature cache directory
        feature_cache_dir = os.path.join(config.output_dir, config.feature_cache_dir)
        self.feature_cache_dir = feature_cache_dir

        # Create feature cache directory if specified
        if self.feature_cache_dir and not os.path.exists(self.feature_cache_dir):
            os.makedirs(self.feature_cache_dir)
            logger.info(f"Created feature cache directory: {self.feature_cache_dir}")
        
        # Storage for precomputed features - these eliminate redundant computation
        self.precomputed_patch_features = {}
        self.precomputed_feature_maps = {}
        
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
    
    def _hash_images_batch(self, images):
        """
        Generate hash keys for a batch of images to use as cache keys.
        
        Args:
            images (tf.Tensor): Batch of images
            
        Returns:
            list: List of hash keys
        """
        # Convert to numpy for hashing
        images_np = images.numpy()
        return [hash(img.tobytes()) for img in images_np]
            
    def _ensure_feature_extractor(self):
        """Initialize the feature extractor if not already done"""
        if self.feature_extractor is None:
            from feature_extractor import FeatureExtractor
            self.feature_extractor = FeatureExtractor()
            logger.info("Feature extractor initialized.")
    
    def precompute_all_patch_features(self, dataset):
        """
        Precompute patch features for all images in the dataset.
        This should be called once before training.
        
        Args:
            dataset (tf.data.Dataset): Dataset containing images
        """
        if not self.use_features:
            logger.info("Feature extraction not enabled. Skipping precomputation.")
            return
            
        self._ensure_feature_extractor()
        logger.info("Starting precomputation of all patch features...")
        
        total_batches = 0
        processed_images = 0
        
        for batch_data in dataset:
            # Handle different dataset formats
            if isinstance(batch_data, tuple):
                images, _ = batch_data
            else:
                images = batch_data
                
            images = tf.cast(images, tf.float32)
            batch_size = tf.shape(images)[0]
            
            # Get hash keys for this batch
            hash_keys = self._hash_images_batch(images)
            
            # Check which images we haven't computed features for yet
            new_images = []
            new_hash_keys = []
            
            for i, hash_key in enumerate(hash_keys):
                if hash_key not in self.precomputed_patch_features:
                    new_images.append(images[i])
                    new_hash_keys.append(hash_key)
            
            if new_images:
                # Convert list back to tensor
                new_images_tensor = tf.stack(new_images)
                
                # Compute patch features for new images
                batch_patch_features = self.feature_extractor.extract_batch_patch_features(new_images_tensor)
                
                # Store patch features
                for i, hash_key in enumerate(new_hash_keys):
                    self.precomputed_patch_features[hash_key] = batch_patch_features[i]
                
                logger.debug(f"Precomputed patch features for {len(new_images)} new images in batch {total_batches}")
            
            total_batches += 1
            processed_images += batch_size.numpy()
            
            if total_batches % 10 == 0:
                logger.info(f"Processed {total_batches} batches, {processed_images} images. "
                           f"Cached features for {len(self.precomputed_patch_features)} unique images.")
        
        logger.info(f"Precomputation complete. Cached patch features for {len(self.precomputed_patch_features)} unique images.")
    
    def precompute_all_feature_maps(self, dataset):
        """
        Precompute feature maps for all images in the dataset using genetic rules.
        This should be called once before training, after precompute_all_patch_features.
        
        Args:
            dataset (tf.data.Dataset): Dataset containing images
        """
        if not self.use_features or self.genetic_rules is None:
            logger.info("Feature extraction not enabled or genetic rules not set. Skipping feature map precomputation.")
            return
            
        logger.info("Starting precomputation of all feature maps with genetic masks...")
        
        total_batches = 0
        processed_images = 0
        
        for batch_data in dataset:
            # Handle different dataset formats
            if isinstance(batch_data, tuple):
                images, _ = batch_data
            else:
                images = batch_data
                
            images = tf.cast(images, tf.float32)
            batch_size = tf.shape(images)[0]
            
            # Get hash keys for this batch
            hash_keys = self._hash_images_batch(images)
            
            # Check which images we haven't computed feature maps for yet
            new_images = []
            new_hash_keys = []
            
            for i, hash_key in enumerate(hash_keys):
                if hash_key not in self.precomputed_feature_maps:
                    new_images.append(images[i])
                    new_hash_keys.append(hash_key)
            
            if new_images:
                # Convert list back to tensor
                new_images_tensor = tf.stack(new_images)
                
                # Compute feature maps for new images using precomputed patch features
                batch_feature_maps = self._compute_feature_maps_from_precomputed_patches(new_images_tensor, new_hash_keys)
                
                # Store feature maps
                for i, hash_key in enumerate(new_hash_keys):
                    self.precomputed_feature_maps[hash_key] = batch_feature_maps[i]
                
                logger.debug(f"Precomputed feature maps for {len(new_images)} new images in batch {total_batches}")
            
            total_batches += 1
            processed_images += batch_size.numpy()
            
            if total_batches % 10 == 0:
                logger.info(f"Processed {total_batches} batches, {processed_images} images. "
                           f"Cached feature maps for {len(self.precomputed_feature_maps)} unique images.")
        
        logger.info(f"Feature map precomputation complete. Cached feature maps for {len(self.precomputed_feature_maps)} unique images.")
    
    def _compute_feature_maps_from_precomputed_patches(self, images, hash_keys):
        """
        Compute feature maps using precomputed patch features and genetic rules.
        This eliminates redundant feature extraction by reusing precomputed features.
        
        Args:
            images (tf.Tensor): Batch of images
            hash_keys (list): Hash keys for the images
            
        Returns:
            tf.Tensor: Batch of feature maps
        """
        self._ensure_feature_extractor()
        
        batch_feature_maps = []
        
        for i, (img, hash_key) in enumerate(zip(images, hash_keys)):
            # Get precomputed patch features
            if hash_key in self.precomputed_patch_features:
                patch_features = self.precomputed_patch_features[hash_key]
            else:
                # Fallback: compute patch features if not precomputed
                logger.warning(f"Patch features not found for hash {hash_key}, computing on-the-fly")
                patch_features = self.feature_extractor.extract_patch_features(img)
                self.precomputed_patch_features[hash_key] = patch_features
            
            patch_features = tf.cast(patch_features, tf.float32)
            
            # Generate dynamic mask using genetic rules
            patch_mask = utils.generate_dynamic_mask(patch_features, self.n_patches_h, self.n_patches_w, self.genetic_rules)
            patch_mask.set_shape([self.n_patches_h, self.n_patches_w])
            patch_mask = tf.cast(patch_mask, tf.float32)
            
            # Convert patch mask to pixel mask
            pixel_mask = utils.convert_patch_mask_to_pixel_mask(patch_mask)
            pixel_mask = tf.cast(pixel_mask, tf.float32)
            
            # Use the precomputed patch features directly instead of re-extracting
            feature_maps = patch_features
            feature_maps = tf.cast(feature_maps, tf.float32)
            
            # Resize feature maps to target size
            target_h = self.input_shape[0]
            target_w = self.input_shape[1]

            resized_feature_maps = tf.image.resize(
                feature_maps,
                [target_h, target_w],
                method=tf.image.ResizeMethod.BILINEAR
            )
            resized_feature_maps = tf.cast(resized_feature_maps, tf.float32)
            resized_feature_maps.set_shape([target_h, target_w, self.feature_channels])
            
            # Apply mask to the resized feature maps
            pixel_mask_expanded = tf.expand_dims(pixel_mask, axis=-1)
            pixel_mask_expanded = tf.cast(pixel_mask, tf.float32)
            
            masked_features = resized_feature_maps * pixel_mask_expanded
            masked_features = tf.cast(masked_features, tf.float32)
            
            batch_feature_maps.append(masked_features)
        
        return tf.stack(batch_feature_maps)
    
    def extract_batch_features(self, images):
        """
        Extract features from a batch of images using genetic dynamic masks.
        Uses precomputed features when available.
        
        Args:
            images (tf.Tensor or np.ndarray): Batch of input images
            
        Returns:
            tf.Tensor: Batch of masked feature maps
        """
        logger.info("Extracting batch features using genetic dynamic masks.")
        
        if not isinstance(images, tf.Tensor):
            images = tf.convert_to_tensor(images, dtype=tf.float32)
        
        # Get hash keys for this batch
        hash_keys = self._hash_images_batch(images)
        
        # Check if we have precomputed feature maps for all images
        missing_hashes = [h for h in hash_keys if h not in self.precomputed_feature_maps]
        
        if missing_hashes:
            logger.warning(f"Missing precomputed feature maps for {len(missing_hashes)} images. Consider running precomputation first.")
            # Fallback to on-the-fly computation
            return self._compute_feature_maps_from_precomputed_patches(images, hash_keys)
        else:
            # Use precomputed feature maps
            feature_maps = []
            for hash_key in hash_keys:
                feature_maps.append(self.precomputed_feature_maps[hash_key])
            return tf.stack(feature_maps)
        
    def extract_single_image_features(self, image):
        """
        Extract features from a single image using genetic dynamic masks.
        Uses precomputed features when available.
        
        Args:
            image (tf.Tensor or np.ndarray): Single input image
            
        Returns:
            tf.Tensor: Masked feature maps
        """
        logger.info("Extracting features for a single image using genetic dynamic mask.")
        
        if not isinstance(image, tf.Tensor):
            image = tf.convert_to_tensor(image, dtype=tf.float32)
        
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = tf.expand_dims(image, axis=0)
        
        # Extract features for the batch (size 1)
        features = self.extract_batch_features(image)
        
        # Remove batch dimension
        return features[0]
    
    def prepare_dataset_with_precomputed_features(self, dataset):
        """
        Prepare a TensorFlow dataset for training using precomputed features.
        This is much more efficient than computing features on-the-fly.
        
        Args:
            dataset (tf.data.Dataset): Input dataset with (images, labels)
            
        Returns:
            tf.data.Dataset: Dataset with precomputed features added
        """
        if not self.use_features:
            logger.info("Feature extraction not enabled. Returning original dataset.")
            return dataset
            
        logger.info("Preparing dataset with precomputed features.")
        
        def add_precomputed_features_to_batch(images, labels):
            """Add precomputed features to each batch"""
            images = tf.cast(images, tf.float32)
            
            # Use tf.py_function to access precomputed features
            def get_features_for_batch(images_np):
                hash_keys = [hash(img.tobytes()) for img in images_np]
                features_list = []
                for hash_key in hash_keys:
                    if hash_key in self.precomputed_feature_maps:
                        features_list.append(self.precomputed_feature_maps[hash_key].numpy())
                    else:
                        # This should not happen if precomputation was done correctly
                        logger.error(f"Feature map not found in cache for hash {hash_key}")
                        raise ValueError(f"Missing precomputed feature map for image hash {hash_key}")
                
                return np.stack(features_list)
            
            features = tf.py_function(
                get_features_for_batch,
                [images],
                tf.float32
            )
            
            # Set the shape explicitly
            features.set_shape([None, self.input_shape[0], self.input_shape[1], self.feature_channels])
            
            return {'image_input': images, 'feature_input': features}, labels
            
        # Apply our mapping function to the dataset
        return dataset.map(
            add_precomputed_features_to_batch,
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)
        
    def prepare_dataset(self, dataset):
        """
        Prepare a TensorFlow dataset for training by adding feature extraction.
        This method now prioritizes precomputed features.
        
        Args:
            dataset (tf.data.Dataset): Input dataset with (images, labels)
            
        Returns:
            tf.data.Dataset: Dataset with features added
        """
        if not self.use_features:
            logger.info("Feature extraction not enabled. Returning original dataset.")
            return dataset
        
        # Check if we have precomputed features
        if len(self.precomputed_feature_maps) > 0:
            logger.info("Using precomputed features for dataset preparation.")
            return self.prepare_dataset_with_precomputed_features(dataset)
        else:
            logger.error("No precomputed features found. Please run precompute_all_feature_maps() first.")
            raise ValueError("Precomputed features are required but not found. Run precomputation first.")
    
    def train(self, train_dataset, validation_dataset=None, model_path='ai_detector_model.h5', precompute_features=True):
        """
        Train the model with the prepared dataset.
        
        Args:
            train_dataset (tf.data.Dataset): Training dataset
            validation_dataset (tf.data.Dataset): Optional validation dataset
            model_path (str): Path to save the model
            precompute_features (bool): Whether to precompute features before training
            
        Returns:
            dict: Training history
        """
        logger.info("Starting model wrapper training.")
        
        # Precompute features if requested and using feature extraction
        if self.use_features and precompute_features:
            logger.info("Precomputing features before training...")
            
            # Precompute patch features first
            self.precompute_all_patch_features(train_dataset)
            if validation_dataset:
                self.precompute_all_patch_features(validation_dataset)
            
            # Then precompute feature maps using genetic rules
            if self.genetic_rules is not None:
                self.precompute_all_feature_maps(train_dataset)
                if validation_dataset:
                    self.precompute_all_feature_maps(validation_dataset)
            else:
                logger.warning("Genetic rules not set. Cannot precompute feature maps.")
        
        # Prepare datasets with features
        if self.use_features:
            logger.info("Preparing training and validation datasets with features.")
            prepared_train_ds = self.prepare_dataset(train_dataset)
            prepared_val_ds = self.prepare_dataset(validation_dataset) if validation_dataset else None
        else:
            logger.info("Training without features. Using original datasets.")
            prepared_train_ds = train_dataset
            prepared_val_ds = validation_dataset
            
        # Train the model
        history = self.model.train(
            prepared_train_ds,
            validation_dataset=prepared_val_ds,
            output_model_path=model_path
        )
        logger.info("Model wrapper training complete.")

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
        """
        Make a prediction on a single image.
        
        Args:
            image (tf.Tensor or np.ndarray): Input image
            
        Returns:
            dict: Prediction results
        """
        logger.info("Making prediction for a single image.")
        # Convert to tensor if needed
        if not isinstance(image, tf.Tensor):
            image = tf.convert_to_tensor(image, dtype=tf.float32)
        
        # Ensure single image has batch dimension
        image_shape = tf.shape(image)
        image = tf.cond(
            tf.equal(tf.rank(image), 3),
            lambda: tf.expand_dims(image, axis=0),
            lambda: image
        )
        logger.debug(f"Image shape after batch dimension check: {tf.shape(image)}")
        
        # If using features, we need to extract them using genetic masks
        if self.use_features:
            logger.info("Extracting genetic features for prediction.")
            features = self.extract_single_image_features(image[0])  # Remove batch dim for extraction
            features = tf.expand_dims(features, axis=0)  # Add batch dim back
            inputs = {'image_input': image, 'feature_input': features}
        else:
            logger.info("Making prediction without genetic features.")
            inputs = image
        
        # Get raw prediction
        prediction = self.model.predict(inputs)
        
        # Extract scalar value from prediction tensor
        if isinstance(prediction, tf.Tensor):
            prediction_value = tf.squeeze(prediction)
        else:
            prediction_value = tf.constant(prediction[0][0] if len(prediction.shape) > 1 else prediction[0])
        
        # Convert to Python float
        prediction_value = tf.cast(prediction_value, tf.float32)
        
        # Determine result using TensorFlow operations
        is_ai = tf.greater(prediction_value, 0.5)
        confidence = tf.cond(
            is_ai,
            lambda: prediction_value,
            lambda: 1.0 - prediction_value
        )
        
        result = {
            'is_ai': bool(is_ai.numpy()),
            'label': "AI-generated" if is_ai.numpy() else "Human-generated",
            'confidence': float(confidence.numpy()),
            'raw_prediction': float(prediction_value.numpy())
        }
        logger.info(f"Prediction complete: {result['label']} with confidence {result['confidence']:.4f}")
        return result