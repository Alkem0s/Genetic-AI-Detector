# model_architecture.py
import tensorflow as tf
from tensorflow.keras import layers, models, Model, callbacks # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

import global_config
import utils # Assuming utils contains generate_dynamic_mask and other helpers

import logging
logger = logging.getLogger(__name__)

class AIDetectorModel:
    """
    Class containing model architecture and training functionality for detecting AI-generated images.
    Can be configured to use image inputs only or combined with feature maps from feature extraction.
    """
    def __init__(self, config, feature_channels=8):
        """
        Initialize the model architecture.
        
        Args:
            config (object): Configuration object with model parameters
            feature_channels (int): Number of feature channels from feature extraction
        """
        logger.info("Initializing AIDetectorModel...")
        self.config = config
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
        # Main input for the image
        image_input = layers.Input(shape=self.input_shape, name='image_input')
        
        # CNN for image processing
        x = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization()(x)
        
        if self.use_features:
            logger.info("Building model with feature extraction branch.")
            # Feature map input (output from genetic feature extraction)
            feature_input = layers.Input(shape=(self.input_shape[0], self.input_shape[1], self.feature_channels), 
                                        name='feature_input')
            
            # Process feature maps
            f = layers.Conv2D(32, (3, 3), activation='relu')(feature_input)
            f = layers.MaxPooling2D((2, 2))(f)
            f = layers.Conv2D(64, (3, 3), activation='relu')(f)
            f = layers.MaxPooling2D((2, 2))(f)
            f = layers.Conv2D(128, (3, 3), activation='relu')(f)
            f = layers.MaxPooling2D((2, 2))(f)
            
            # Combine feature processing with main CNN using attention mechanism
            # First get them to the same dimensions
            # Use Lambda layers for TensorFlow operations on KerasTensors
            x_shape = layers.Lambda(lambda t: tf.shape(t)[1:3])(x)
            f_shape = layers.Lambda(lambda t: tf.shape(t)[1:3])(f)
            
            # Use tf.cond for conditional resizing, wrapped in a Lambda layer
            f = layers.Lambda(lambda args: tf.cond(
                tf.reduce_any(tf.not_equal(args[0], args[1])),
                lambda: layers.Resizing(args[0][0], args[0][1])(args[2]),
                lambda: args[2]
            ))([x_shape, f_shape, f])
            logger.debug("Feature branch resized for concatenation.")
            
            # Enhanced attention mechanism
            # Global attention weights
            attn_weights = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
            # Apply attention to feature maps
            f_weighted = layers.Multiply()([f, attn_weights])
            # Cross-attention
            attention = layers.Multiply()([x, f_weighted])  # Element-wise multiplication as attention
            enhanced = layers.Add()([x, attention])  # Combine original with attention-weighted
            logger.debug("Attention mechanism applied.")
            
            # Continue with the CNN
            x = layers.Conv2D(128, (3, 3), activation='relu')(enhanced)
        else:
            logger.info("Building model without feature extraction branch.")
            # Without feature extraction, add an extra convolutional layer to maintain model capacity
            x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        
        # Common downstream layers
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization()(x)
        
        # Global Average Pooling instead of Flatten to ensure fully defined shape
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(self.config.image_size, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        output = layers.Dense(1, activation='sigmoid')(x)
        
        # Create model with appropriate inputs
        if self.use_features:
            model = Model(inputs=[image_input, feature_input], outputs=output)
        else:
            model = Model(inputs=image_input, outputs=output)
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        logger.info("Model compilation complete.")
        
        return model
    
    def get_data_augmentation(self):
        """
        Create data augmentation pipeline.
        
        Returns:
            tf.keras.Sequential: Data augmentation pipeline
        """
        logger.debug("Creating data augmentation pipeline.")
        return tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
            layers.RandomBrightness(0.05),
        ])

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
            epochs=self.config.epochs,
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
    def __init__(self, config, feature_channels=8, genetic_rules=None):
        """
        Initialize the model wrapper with genetic algorithm integration.
        
        Args:
            config (object): Configuration object with model parameters
            feature_channels (int): Number of feature channels from feature extraction
            genetic_rules (list, optional): List of evolved rules from genetic algorithm
            patch_size (int): Size of patches for feature extraction
            feature_cache_dir (str, optional): Directory to cache extracted features
        """
        logger.info("Initializing ModelWrapper...")
        self.input_shape = (config.image_size, config.image_size, 3)
        self.use_features = config.use_feature_extraction
        self.feature_channels = feature_channels
        self.patch_size = config.patch_size
        self.genetic_rules = genetic_rules
        
        # Calculate patch grid dimensions using TensorFlow operations
        self.n_patches_h = tf.constant(self.input_shape[0] // self.patch_size, dtype=tf.int32)
        self.n_patches_w = tf.constant(self.input_shape[1] // self.patch_size, dtype=tf.int32)

        # Initialize the model
        self.model = AIDetectorModel(config, feature_channels=feature_channels)

        # Feature extraction components will be initialized when needed
        self.feature_extractor = None
        
        # Get the full path to feature cache directory
        feature_cache_dir = os.path.join(config.output_dir, config.feature_cache_dir)
        self.feature_cache_dir = feature_cache_dir

        # Create feature cache directory if specified
        if self.feature_cache_dir and not os.path.exists(self.feature_cache_dir):
            os.makedirs(self.feature_cache_dir)
            logger.info(f"Created feature cache directory: {self.feature_cache_dir}")
            
        # Cache for precomputed patch features during training
        self.patch_features_cache = {}
        logger.info("ModelWrapper initialized.")
        
    def set_genetic_rules(self, genetic_rules):
        """
        Set or update the genetic algorithm rules for dynamic mask generation.
        
        Args:
            genetic_rules (list): List of evolved rules from genetic algorithm
        """
        self.genetic_rules = genetic_rules
        logger.info(f"Updated genetic rules: {len(genetic_rules)} rules set")
        
        # Clear cache when rules change
        self.patch_features_cache = {}
        logger.debug("Patch features cache cleared due to rule change.")
        
    def load_genetic_rules(self, rules_path):
        """
        Load genetic algorithm rules from a file.
        
        Args:
            rules_path (str): Path to the saved genetic rules file
        """
        logger.info(f"Attempting to load genetic rules from {rules_path}...")
        try:
            with open(rules_path, 'rb') as f:
                self.genetic_rules = pickle.load(f)
            logger.info(f"Loaded {len(self.genetic_rules)} genetic rules from {rules_path}")
        except Exception as e:
            logger.error(f"Failed to load genetic rules from {rules_path}: {e}")
            raise
            
    def save_genetic_rules(self, rules_path):
        """
        Save the current genetic rules to a file.
        
        Args:
            rules_path (str): Path to save the genetic rules
        """
        if self.genetic_rules:
            logger.info(f"Attempting to save genetic rules to {rules_path}...")
            try:
                with open(rules_path, 'wb') as f:
                    pickle.dump(self.genetic_rules, f)
                logger.info(f"Saved {len(self.genetic_rules)} genetic rules to {rules_path}")
            except Exception as e:
                logger.error(f"Failed to save genetic rules to {rules_path}: {e}")
                raise
        else:
            logger.warning("No genetic rules to save. Skipping save operation.")
            
    def _ensure_feature_extractor(self):
        """Initialize the feature extractor if not already done"""
        if self.feature_extractor is None:
            from feature_extractor import FeatureExtractor
            self.feature_extractor = FeatureExtractor()
            logger.info("Feature extractor initialized.")
            
    # Removed @tf.function decorator and tf.py_function as hashing Python objects isn't directly compatible
    # with TF graph execution for caching purposes in this manner.
    def _hash_image_for_cache(self, image):
        """
        Create a hash key for an image using numpy (CPU) for caching.
        This function is intended to be called outside of @tf.function
        contexts if caching is done at the Python level.
        
        Args:
            image (tf.Tensor): Input image
            
        Returns:
            int: Hash value
        """
        logger.debug("Hashing image for caching.")
        # Convert image tensor to numpy array and then to bytes for hashing
        # This will run on CPU, but is for cache key generation only.
        return hash(image.numpy().tobytes())
            
    @tf.function
    def _precompute_patch_features(self, images, force_recompute=False):
        """
        Precompute patch features for a batch of images using TensorFlow operations.
        
        Args:
            images (tf.Tensor): Batch of images
            force_recompute (bool): Whether to force recomputation even if cached
            
        Returns:
            tf.Tensor: Tensor of patch features for each image
        """
        self._ensure_feature_extractor()
        logger.debug(f"Precomputing patch features for batch of {tf.shape(images)[0]} images. Force recompute: {force_recompute}")
        
        # Convert to tensor if needed
        if not isinstance(images, tf.Tensor):
            images = tf.convert_to_tensor(images, dtype=tf.float32)
        
        # Process images in a batch for efficiency using tf.map_fn
        def extract_single_patch_features_tf(img):
            # Caching logic would typically be outside this @tf.function context
            # if using a Python dict. For now, features are always computed.
            patch_features = self.feature_extractor.extract_patch_features(img)
            return patch_features
        
        # Use tf.map_fn to process all images in the batch
        batch_features = tf.map_fn(
            extract_single_patch_features_tf, # Renamed to avoid confusion with the old py_function lambda
            images,
            fn_output_signature=tf.TensorSpec(
                shape=[self.n_patches_h, self.n_patches_w, len(global_config.feature_weights)],
                dtype=tf.float32
            ),
            parallel_iterations=10
        )
        logger.debug("Batch patch features precomputed.")
        
        return batch_features
        
    @tf.function
    def convert_patch_mask_to_pixel_mask(patch_mask):
        """
        Convert a 2D patch mask to pixel-level mask using GPU-optimized TensorFlow operations.
        Designed for fixed patch sizes and image shapes with minimal function tracing.
        
        Args:
            patch_mask (tf.Tensor): 2D binary mask of shape (n_patches_h, n_patches_w)
            patch_size (int or tuple): Fixed patch size - int for square patches or (h, w) tuple
            image_shape (tuple): Fixed image shape (height, width)
        
        Returns:
            tf.Tensor: Pixel-level binary mask of shape (height, width)
        """

        patch_size = global_config.default_patch_size  # Use default patch size from config
        image_shape = (global_config.image_size, global_config.image_size)  # Use default image size from config

        # Handle patch size input
        if isinstance(patch_size, int):
            patch_h = patch_w = patch_size
        else:
            patch_h, patch_w = patch_size
        
        # Convert to TensorFlow constants for graph optimization
        patch_h = tf.constant(patch_h, dtype=tf.int32)
        patch_w = tf.constant(patch_w, dtype=tf.int32)
        img_h = tf.constant(image_shape[0], dtype=tf.int32)
        img_w = tf.constant(image_shape[1], dtype=tf.int32)
        
        # Ensure patch_mask is float32 and add batch/channel dimensions for tf.image.resize
        patch_mask_float = tf.cast(patch_mask, tf.float32)
        expanded_mask = tf.expand_dims(tf.expand_dims(patch_mask_float, axis=0), axis=-1)
        
        # Use tf.image.resize with nearest neighbor - this is GPU-optimized
        pixel_mask = tf.image.resize(
            expanded_mask,
            [img_h, img_w],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        
        # Remove batch and channel dimensions
        pixel_mask = tf.squeeze(pixel_mask, axis=[0, -1])
        
        # Ensure binary output (handle any floating point precision issues)
        pixel_mask = tf.cast(pixel_mask >= 0.5, tf.float32)
        
        return pixel_mask
        
    @tf.function
    def _extract_features_with_genetic_mask(self, images, batch=True):
        """
        Extract features from images using genetic algorithm dynamic masks with TensorFlow operations.
        
        Args:
            images (tf.Tensor): Input images
            batch (bool): Whether images is a batch or single image
            
        Returns:
            tf.Tensor: Masked feature maps
        """
        self._ensure_feature_extractor()
        logger.debug(f"Extracting features with genetic mask. Batch input: {batch}")
        
        # Convert to tensor if needed
        if not isinstance(images, tf.Tensor):
            images = tf.convert_to_tensor(images, dtype=tf.float32)
            
        # Handle both single image and batch inputs
        if not batch:
            images = tf.expand_dims(images, axis=0)
            logger.debug("Expanded single image to batch dimension.")
            
        # Precompute patch features for all images in the batch
        batch_patch_features = self._precompute_patch_features(images)
        
        # Ensure genetic_rules is a tf.Tensor if possible, or handle its conversion within the map function
        # For simplicity and assuming genetic_rules might be a list of Python dicts,
        
        def process_single_image_tf(args): # Renamed function
            img, patch_features = args
            
            # Generate dynamic mask using genetic rules (direct call, assuming utils.generate_dynamic_mask is pure TF)
            # Ensure self.genetic_rules is accessible and properly formatted for TF ops inside utils.
            patch_mask = utils.generate_dynamic_mask(patch_features, self.n_patches_h, self.n_patches_w, self.genetic_rules)
            
            patch_mask.set_shape([self.n_patches_h, self.n_patches_w])
            logger.debug("Dynamic patch mask generated for a single image.")
            
            # Convert patch mask to pixel mask
            pixel_mask = self.convert_patch_mask_to_pixel_mask(patch_mask)
            
            # Extract full feature maps
            feature_maps = self.feature_extractor.extract_patch_features(img)
            
            # Apply mask to feature maps
            # Expand pixel mask to match feature maps dimensions
            pixel_mask_expanded = tf.expand_dims(pixel_mask, axis=-1)
            pixel_mask_tiled = tf.tile(pixel_mask_expanded, [1, 1, tf.shape(feature_maps)[-1]])
            
            # Apply the mask
            masked_features = feature_maps * pixel_mask_tiled
            logger.debug("Pixel mask applied to feature maps.")
                
            return masked_features
        
        # Process all images in the batch
        batch_masked_features = tf.map_fn(
            process_single_image_tf, # Updated function name
            (images, batch_patch_features),
            fn_output_signature=tf.TensorSpec(
                shape=[self.input_shape[0], self.input_shape[1], self.feature_channels],
                dtype=tf.float32
            ),
            parallel_iterations=10
        )
        logger.debug("Batch masked feature extraction complete.")
        
        # If input was a single image, remove batch dimension
        if not batch:
            batch_masked_features = batch_masked_features[0]
            logger.debug("Removed batch dimension for single image output.")
            
        return batch_masked_features
        
    def extract_batch_features(self, images):
        """
        Extract features from a batch of images using genetic dynamic masks.
        Public method for external use.
        
        Args:
            images (tf.Tensor or np.ndarray): Batch of input images
            
        Returns:
            tf.Tensor: Batch of masked feature maps
        """
        logger.info("Extracting batch features using genetic dynamic masks.")
        return self._extract_features_with_genetic_mask(images, batch=True)
        
    def extract_single_image_features(self, image):
        """
        Extract features from a single image using genetic dynamic masks.
        Public method for external use.
        
        Args:
            image (tf.Tensor or np.ndarray): Single input image
            
        Returns:
            tf.Tensor: Masked feature maps
        """
        logger.info("Extracting features for a single image using genetic dynamic mask.")
        return self._extract_features_with_genetic_mask(image, batch=False)
        
    def load_model(self, model_path):
        """
        Load a saved model from disk.
        
        Args:
            model_path (str): Path to the saved model
        """
        logger.info(f"Loading underlying model from {model_path}.")
        self.model.load(model_path)
        # Update use_features from loaded model
        self.use_features = self.model.use_features
        logger.info(f"Underlying model loaded. Use features: {self.use_features}")
    
    def save_model(self, model_path):
        """
        Save the current model to disk.
        
        Args:
            model_path (str): Path to save the model
        """
        logger.info(f"Saving underlying model to {model_path}.")
        self.model.save(model_path)
        logger.info("Underlying model saved.")
        
    def get_model(self):
        """Get the underlying Keras model"""
        logger.debug("Retrieving underlying Keras model.")
        return self.model.model
    
    def set_model(self, model):
        """Set the underlying Keras model"""
        logger.debug("Setting underlying Keras model.")
        self.model.model = model
        # If it's a Keras model, infer features usage from inputs
        self.use_features = len(model.inputs) > 1
        logger.debug(f"Underlying model set. Use features: {self.use_features}")
    
    def prepare_dataset(self, dataset):
        """
        Prepare a TensorFlow dataset for training by adding feature extraction.
        
        Args:
            dataset (tf.data.Dataset): Input dataset with (images, labels)
            
        Returns:
            tf.data.Dataset: Dataset with features added
        """
        if not self.use_features:
            logger.info("Feature extraction not enabled. Returning original dataset.")
            return dataset
            
        logger.info("Preparing dataset with feature extraction.")
        # Function to add features to each batch
        def add_features_to_batch(images, labels):
            logger.debug("Adding features to a batch in dataset preparation.")
            # Extract features using genetic dynamic masks
            features = self._extract_features_with_genetic_mask(images)
            
            # Ensure features have the right shape
            features = tf.ensure_shape(features, [None, self.input_shape[0], self.input_shape[1], self.feature_channels])
            
            return {'image_input': images, 'feature_input': features}, labels
            
        # Apply our mapping function to the dataset
        return dataset.map(
            add_features_to_batch,
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)
        
    def train(self, train_dataset, validation_dataset=None, model_path='ai_detector_model.h5'):
        """
        Train the model with the prepared dataset.
        
        Args:
            train_dataset (tf.data.Dataset): Training dataset
            validation_dataset (tf.data.Dataset): Optional validation dataset
            model_path (str): Path to save the model
            
        Returns:
            dict: Training history
        """
        logger.info("Starting model wrapper training.")
        # Prepare datasets with features if using feature extraction
        if self.use_features:
            logger.info("Preparing training and validation datasets with genetic features.")
            prepared_train_ds = self.prepare_dataset(train_dataset)
            prepared_val_ds = self.prepare_dataset(validation_dataset) if validation_dataset else None
        else:
            logger.info("Training without genetic features. Using original datasets.")
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