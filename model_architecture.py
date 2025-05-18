import tensorflow as tf
from tensorflow.keras import layers, models, Model, callbacks # type: ignore
import numpy as np
import matplotlib.pyplot as plt

class AIDetectorModel:
    """
    Class containing model architecture and training functionality for detecting AI-generated images.
    Can be configured to use image inputs only or combined with feature maps from feature extraction.
    """
    def __init__(self, input_shape=(224, 224, 3), feature_channels=3, use_features=True):
        """
        Initialize the model architecture.
        
        Args:
            input_shape (tuple): Shape of input images (height, width, channels)
            feature_channels (int): Number of feature channels from feature extraction
            use_features (bool): Whether to use feature extraction or run image-only model
        """
        self.input_shape = input_shape
        self.feature_channels = feature_channels
        self.use_features = use_features
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
            
            # Simple attention mechanism
            attention = layers.Multiply()([x, f])  # Element-wise multiplication as attention
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
    
    def train(self, train_dataset, validation_dataset=None, epochs=50, output_model_path='human_ai_detector_model.h5'):
        """
        Train the model with callbacks for early stopping and learning rate reduction.
        
        Args:
            train_dataset (tf.data.Dataset): Training dataset
            validation_dataset (tf.data.Dataset): Validation dataset
            epochs (int): Maximum number of epochs to train
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
            epochs=epochs,
            validation_data=validation_dataset,
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
    Wrapper class to provide higher-level functionality for model usage.
    Handles feature extraction and model training in a unified way.
    """
    def __init__(self, input_shape=(224, 224, 3), feature_channels=3, use_features=True, mask=None):
        """
        Initialize the model wrapper.
        
        Args:
            input_shape (tuple): Shape of input images
            feature_channels (int): Number of feature channels
            use_features (bool): Whether to use feature extraction
            mask (np.ndarray, optional): Mask for feature extraction
        """
        self.input_shape = input_shape
        self.use_features = use_features
        self.feature_channels = feature_channels
        self.mask = mask
        self.model = AIDetectorModel(input_shape, feature_channels, use_features)
        self.feature_extractor = None  # Will be initialized when needed
        
    def set_mask(self, mask):
        """Set the feature extraction mask"""
        self.mask = mask
        
    def load_model(self, model_path):
        """
        Load a saved model from disk.
        
        Args:
            model_path (str): Path to the saved model
        """
        self.model.load(model_path)
        # Update use_features from loaded model
        self.use_features = self.model.use_features
    
    def get_model(self):
        """
        Get the underlying Keras model.
        
        Returns:
            tf.keras.Model: The model instance
        """
        return self.model.model
    
    def set_model(self, model):
        """
        Set the underlying model.
        
        Args:
            model: Keras model to use
        """
        self.model.model = model
        # If it's a Keras model, infer features usage from inputs
        self.use_features = len(model.inputs) > 1
    
    def _extract_features(self, images):
        """
        Extract features from images using the mask.
        
        Args:
            images (tf.Tensor): Input images
            
        Returns:
            tf.Tensor: Extracted features
        """
        if self.feature_extractor is None:
            # Import on demand to avoid circular imports
            from old_feature_extractor import AIFeatureExtractor
            self.feature_extractor = AIFeatureExtractor()
        
        # Check if we have the mask
        if self.mask is None:
            raise ValueError("Feature extraction mask not set. Call set_mask() first.")
        
        # Handle TensorFlow tensors
        if isinstance(images, tf.Tensor):
            images_np = images.numpy()
        else:
            images_np = images
            
        # Extract features
        _, feature_maps = self.feature_extractor.generate_feature_maps(images_np, self.mask)
        
        return tf.convert_to_tensor(feature_maps, dtype=tf.float32)
    
    def _map_dataset_with_features(self, images, labels):
        """Map function to add features to a dataset batch"""
        if self.use_features:
            features = self._extract_features(images)
            return {'image_input': images, 'feature_input': features}, labels
        else:
            return images, labels
    
    def _prepare_dataset_with_features(self, dataset):
        """
        Prepare a dataset with feature extraction.
        
        Args:
            dataset (tf.data.Dataset): Input dataset with (images, labels)
            
        Returns:
            tf.data.Dataset: Dataset with (images, features, labels)
        """
        if self.use_features:
            # This approach doesn't work well with dataset pipeline
            # We would need to modify each batch on-the-fly
            # for better memory efficiency
            return dataset.map(
                lambda images, labels: self._map_dataset_with_features(images, labels),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        else:
            return dataset
    
    def train_with_datasets(self, train_ds, val_split=0.2, epochs=50, model_path='ai_detector_model.h5'):
        """
        Train the model using TensorFlow datasets.
        
        Args:
            train_ds (tf.data.Dataset): Training dataset with (images, labels)
            val_split (float): Proportion of data to use for validation 
                               (ignored if using dataset pipeline)
            epochs (int): Number of training epochs
            model_path (str): Path to save the model
            
        Returns:
            dict: Training history
        """
        # If using features, we need to process the dataset
        if self.use_features:
            # We'll create a map function for each batch to add features on-the-fly
            # This way we don't need to preprocess the entire dataset up front
            
            @tf.function
            def add_features_to_batch(images, labels):
                # Generate features using the mask
                if self.feature_extractor is None:
                    # Import on demand to avoid circular imports
                    from old_feature_extractor import AIFeatureExtractor
                    self.feature_extractor = AIFeatureExtractor()
                
                # Use a tf.py_function to handle numpy operations
                def extract_features_batch(img_batch):
                    # Convert to numpy, extract features, convert back to tensor
                    img_np = img_batch.numpy()
                    _, feature_maps = self.feature_extractor.generate_feature_maps(img_np, self.mask)
                    return tf.convert_to_tensor(feature_maps, dtype=tf.float32)
                
                features = tf.py_function(
                    func=extract_features_batch,
                    inp=[images],
                    Tout=tf.float32
                )
                
                # Ensure features have the right shape
                features.set_shape([None, self.input_shape[0], self.input_shape[1], self.feature_channels])
                
                return {'image_input': images, 'feature_input': features}, labels
            
            # Apply our mapping function to the dataset
            processed_train_ds = train_ds.map(
                add_features_to_batch,
                num_parallel_calls=tf.data.AUTOTUNE
            ).prefetch(tf.data.AUTOTUNE)
            
            # Create validation set (taking val_split from the training set)
            # For simplicity in this update, we'll skip validation
            # A proper implementation would create a validation set
            validation_ds = None
            
        else:
            # For non-feature model, use the dataset directly
            processed_train_ds = train_ds
            validation_ds = None
        
        # Train the model
        history = self.model.train(
            processed_train_ds, 
            validation_dataset=validation_ds,
            epochs=epochs,
            output_model_path=model_path
        )
        
        return history
    
    def predict_image(self, image):
        """
        Make a prediction on a single image.
        
        Args:
            image (tf.Tensor or np.ndarray): Input image
            
        Returns:
            dict: Prediction results
        """
        # If using features, we need to extract them first
        if self.use_features:
            features = self._extract_features(image)
            inputs = {'image_input': image, 'feature_input': features}
        else:
            inputs = image
        
        # Get raw prediction
        prediction = self.model.predict(inputs)
        if isinstance(prediction, np.ndarray):
            if len(prediction.shape) > 1:
                prediction = prediction[0][0]  # Extract single value for batch
            else:
                prediction = prediction[0]
        
        # Determine result
        is_ai = prediction > 0.5
        confidence = prediction if is_ai else 1 - prediction
        
        return {
            'is_ai': bool(is_ai),
            'label': "AI-generated" if is_ai else "Human-generated",
            'confidence': float(confidence),
            'raw_prediction': float(prediction)
        }