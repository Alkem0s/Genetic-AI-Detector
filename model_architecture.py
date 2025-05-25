import logging
import tensorflow as tf
from tensorflow.keras import layers, models, Model, callbacks # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

import global_config
import utils

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
        self.config = config
        self.input_shape = (config.image_size, config.image_size, 3)
        self.feature_channels = feature_channels
        self.use_features = config.use_feature_extraction
        self.model = self._build_model()
        
    def _build_model(self):
        """
        Build the model architecture, either with or without feature extraction.
        
        Returns:
            tf.keras.Model: The compiled model
        """
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
            x_shape = x.shape[1:3]
            f_shape = f.shape[1:3]
            
            if x_shape != f_shape:
                # Resize feature map to match x
                f = layers.Resizing(x_shape[0], x_shape[1])(f)
            
            # Enhanced attention mechanism
            # Global attention weights
            attn_weights = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
            # Apply attention to feature maps
            f_weighted = layers.Multiply()([f, attn_weights])
            # Cross-attention
            attention = layers.Multiply()([x, f_weighted])  # Element-wise multiplication as attention
            enhanced = layers.Add()([x, attention])  # Combine original with attention-weighted
            
            # Continue with the CNN
            x = layers.Conv2D(128, (3, 3), activation='relu')(enhanced)
        else:
            # Without feature extraction, add an extra convolutional layer to maintain model capacity
            x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        
        # Common downstream layers
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization()(x)
        
        # Flatten and dense layers
        x = layers.Flatten()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(224, activation='relu')(x)
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
        
        return model
    
    def get_data_augmentation(self):
        """
        Create data augmentation pipeline.
        
        Returns:
            tf.keras.Sequential: Data augmentation pipeline
        """
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
        # Set up callbacks
        callbacks_list = [
            callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            callbacks.ModelCheckpoint(output_model_path, save_best_only=True)
        ]
        
        # Train the model
        history = self.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=self.config.epochs,
            callbacks=callbacks_list
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
        return self.model.evaluate(test_dataset)
    
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
        return self.model.predict(inputs)
    
    def save(self, filepath):
        """
        Save the model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        self.model.save(filepath)
    
    def load(self, filepath):
        """
        Load a saved model from disk.
        
        Args:
            filepath (str): Path to the saved model
        """
        self.model = tf.keras.models.load_model(filepath)
        # Determine if this is a feature-using model or not by checking the input shape
        self.use_features = len(self.model.inputs) > 1


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
        self.input_shape = (config.image_size, config.image_size, 3)
        self.use_features = config.use_feature_extraction
        self.feature_channels = feature_channels
        self.patch_size = config.patch_size
        self.genetic_rules = genetic_rules
        
        # Calculate patch grid dimensions
        self.n_patches_h = self.input_shape[0] // self.patch_size
        self.n_patches_w = self.input_shape[1] // self.patch_size

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
            
        # Cache for precomputed patch features during training
        self.patch_features_cache = {}
        
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
        
    def load_genetic_rules(self, rules_path):
        """
        Load genetic algorithm rules from a file.
        
        Args:
            rules_path (str): Path to the saved genetic rules file
        """
        try:
            with open(rules_path, 'rb') as f:
                self.genetic_rules = pickle.load(f)
            logger.info(f"Loaded {len(self.genetic_rules)} genetic rules from {rules_path}")
        except Exception as e:
            logger.error(f"Failed to load genetic rules: {e}")
            raise
            
    def save_genetic_rules(self, rules_path):
        """
        Save the current genetic rules to a file.
        
        Args:
            rules_path (str): Path to save the genetic rules
        """
        if self.genetic_rules:
            try:
                with open(rules_path, 'wb') as f:
                    pickle.dump(self.genetic_rules, f)
                logger.info(f"Saved {len(self.genetic_rules)} genetic rules to {rules_path}")
            except Exception as e:
                logger.error(f"Failed to save genetic rules: {e}")
                raise
        else:
            logger.warning("No genetic rules to save")
            
    def _ensure_feature_extractor(self):
        """Initialize the feature extractor if not already done"""
        if self.feature_extractor is None:
            from feature_extractor import FeatureExtractor
            self.feature_extractor = FeatureExtractor()
            logger.info("Feature extractor initialized")
            
    @tf.function
    def _tf_hash_image(self, image):
        """
        Create a hash key for an image using TensorFlow operations.
        
        Args:
            image (tf.Tensor): Input image
            
        Returns:
            tf.Tensor: Hash value as tensor
        """
        # Convert image to bytes and compute hash
        image_bytes = tf.py_function(
            func=lambda x: hash(x.numpy().tobytes()),
            inp=[image],
            Tout=tf.int64
        )
        return image_bytes
            
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
        
        # Convert to tensor if needed
        if not isinstance(images, tf.Tensor):
            images = tf.convert_to_tensor(images, dtype=tf.float32)
        
        # Process images in a batch for efficiency using tf.map_fn
        def extract_single_patch_features(img):
            # Generate a unique key for this image (using tensor operations)
            img_key = self._tf_hash_image(img)
            
            # For simplicity in TensorFlow context, we'll always compute features
            # Caching can be handled at a higher level if needed
            patch_features = self.feature_extractor.extract_patch_features(
                img, patch_size=self.patch_size
            )
            return patch_features
        
        # Use tf.map_fn to process all images in the batch
        batch_features = tf.map_fn(
            extract_single_patch_features,
            images,
            fn_output_signature=tf.TensorSpec(
                shape=[self.n_patches_h, self.n_patches_w, len(global_config.feature_weights)],
                dtype=tf.float32
            ),
            parallel_iterations=10
        )
        
        return batch_features
        
    @tf.function
    def convert_patch_mask_to_pixel_mask(self, patch_mask):
        """
        Convert a patch-level mask to a pixel-level mask using TensorFlow operations.
        
        Args:
            patch_mask (tf.Tensor): Binary mask of shape (n_patches_h, n_patches_w)
            
        Returns:
            tf.Tensor: Binary mask of shape (height, width)
        """
        # Create pixel mask with the same dimensions as the input image
        pixel_mask = tf.zeros((self.input_shape[0], self.input_shape[1]), dtype=tf.float32)
        
        # Use tf.py_function for the nested loop logic
        def patch_to_pixel_conversion(patch_mask_np, pixel_mask_np):
            # Convert tensors to numpy for processing
            patch_mask_np = patch_mask_np.numpy()
            pixel_mask_np = pixel_mask_np.numpy()
            
            # Expand each patch to corresponding pixels
            for h in range(patch_mask_np.shape[0]):
                for w in range(patch_mask_np.shape[1]):
                    if patch_mask_np[h, w] == 1:
                        # Calculate the corresponding pixel coordinates
                        y_start = h * self.patch_size
                        y_end = min((h + 1) * self.patch_size, self.input_shape[0])
                        x_start = w * self.patch_size
                        x_end = min((w + 1) * self.patch_size, self.input_shape[1])
                        
                        # Set all pixels in this patch to 1
                        pixel_mask_np[y_start:y_end, x_start:x_end] = 1.0
                        
            return pixel_mask_np
        
        # Use tf.py_function to handle the conversion
        pixel_mask = tf.py_function(
            func=patch_to_pixel_conversion,
            inp=[patch_mask, pixel_mask],
            Tout=tf.float32
        )
        
        # Set the shape explicitly
        pixel_mask.set_shape([self.input_shape[0], self.input_shape[1]])
        
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
        
        # Convert to tensor if needed
        if not isinstance(images, tf.Tensor):
            images = tf.convert_to_tensor(images, dtype=tf.float32)
            
        # Handle both single image and batch inputs
        if not batch:
            images = tf.expand_dims(images, axis=0)
            
        # Precompute patch features for all images in the batch
        batch_patch_features = self._precompute_patch_features(images)
        
        def process_single_image(args):
            img, patch_features = args
            
            # Generate dynamic mask using genetic rules (using tf.py_function)
            patch_mask = tf.py_function(
                func=lambda pf: utils.generate_dynamic_mask(pf.numpy(), self.genetic_rules),
                inp=[patch_features],
                Tout=tf.float32
            )
            patch_mask.set_shape([self.n_patches_h, self.n_patches_w])
            
            # Convert patch mask to pixel mask
            pixel_mask = self.convert_patch_mask_to_pixel_mask(patch_mask)
            
            # Extract full feature maps
            _, feature_maps, _ = self.feature_extractor.extract_all_features(img)
            
            # Apply mask to feature maps
            # Expand pixel mask to match feature maps dimensions
            pixel_mask_expanded = tf.expand_dims(pixel_mask, axis=-1)
            pixel_mask_tiled = tf.tile(pixel_mask_expanded, [1, 1, tf.shape(feature_maps)[-1]])
            
            # Apply the mask
            masked_features = feature_maps * pixel_mask_tiled
                
            return masked_features
        
        # Process all images in the batch
        batch_masked_features = tf.map_fn(
            process_single_image,
            (images, batch_patch_features),
            fn_output_signature=tf.TensorSpec(
                shape=[self.input_shape[0], self.input_shape[1], self.feature_channels],
                dtype=tf.float32
            ),
            parallel_iterations=10
        )
        
        # If input was a single image, remove batch dimension
        if not batch:
            batch_masked_features = batch_masked_features[0]
            
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
        return self._extract_features_with_genetic_mask(image, batch=False)
        
    def load_model(self, model_path):
        """
        Load a saved model from disk.
        
        Args:
            model_path (str): Path to the saved model
        """
        self.model.load(model_path)
        # Update use_features from loaded model
        self.use_features = self.model.use_features
    
    def save_model(self, model_path):
        """
        Save the current model to disk.
        
        Args:
            model_path (str): Path to save the model
        """
        self.model.save(model_path)
        
    def get_model(self):
        """Get the underlying Keras model"""
        return self.model.model
    
    def set_model(self, model):
        """Set the underlying Keras model"""
        self.model.model = model
        # If it's a Keras model, infer features usage from inputs
        self.use_features = len(model.inputs) > 1
    
    def prepare_dataset(self, dataset):
        """
        Prepare a TensorFlow dataset for training by adding feature extraction.
        
        Args:
            dataset (tf.data.Dataset): Input dataset with (images, labels)
            
        Returns:
            tf.data.Dataset: Dataset with features added
        """
        if not self.use_features:
            return dataset
            
        # Function to add features to each batch
        def add_features_to_batch(images, labels):
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
        # Prepare datasets with features if using feature extraction
        if self.use_features:
            prepared_train_ds = self.prepare_dataset(train_dataset)
            prepared_val_ds = self.prepare_dataset(validation_dataset) if validation_dataset else None
        else:
            prepared_train_ds = train_dataset
            prepared_val_ds = validation_dataset
            
        # Train the model
        history = self.model.train(
            prepared_train_ds,
            validation_dataset=prepared_val_ds,
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
        # Prepare test dataset with features if needed
        if self.use_features:
            prepared_test_ds = self.prepare_dataset(test_dataset)
        else:
            prepared_test_ds = test_dataset
            
        return self.model.evaluate(prepared_test_ds)
        
    def predict_image(self, image):
        """
        Make a prediction on a single image.
        
        Args:
            image (tf.Tensor or np.ndarray): Input image
            
        Returns:
            dict: Prediction results
        """
        # Convert to tensor if needed
        if not isinstance(image, tf.Tensor):
            image = tf.convert_to_tensor(image, dtype=tf.float32)
        
        # Ensure single image has batch dimension
        if len(tf.shape(image)) == 3:
            image = tf.expand_dims(image, axis=0)
        
        # If using features, we need to extract them using genetic masks
        if self.use_features:
            features = self.extract_single_image_features(image[0])  # Remove batch dim for extraction
            features = tf.expand_dims(features, axis=0)  # Add batch dim back
            inputs = {'image_input': image, 'feature_input': features}
        else:
            inputs = image
        
        # Get raw prediction
        prediction = self.model.predict(inputs)
        
        # Extract scalar value from prediction tensor
        if isinstance(prediction, tf.Tensor):
            prediction_value = tf.squeeze(prediction).numpy()
        else:
            prediction_value = prediction[0][0] if len(prediction.shape) > 1 else prediction[0]
        
        # Convert to Python float
        prediction_value = float(prediction_value)
        
        # Determine result
        is_ai = prediction_value > 0.5
        confidence = prediction_value if is_ai else 1 - prediction_value
        
        return {
            'is_ai': bool(is_ai),
            'label': "AI-generated" if is_ai else "Human-generated",
            'confidence': float(confidence),
            'raw_prediction': float(prediction_value)
        }