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
        x = layers.Dense(512, activation='relu')(x)
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
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ])
    
    def prepare_datasets(self, X_images, X_features, y_labels, test_split=0.2, batch_size=32):
        """
        Prepare training and validation datasets with data augmentation.
        
        Args:
            X_images (np.ndarray): Image data
            X_features (np.ndarray): Feature maps data
            y_labels (np.ndarray): Labels (0 for human, 1 for AI)
            test_split (float): Proportion of data to use for validation
            batch_size (int): Batch size for training
            
        Returns:
            tuple: (augmented_train_dataset, test_dataset)
        """
        # Create training dataset based on whether we're using features
        if self.use_features:
            train_dataset = tf.data.Dataset.from_tensor_slices((
                {"image_input": X_images, "feature_input": X_features},
                y_labels
            ))
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices((
                X_images,
                y_labels
            ))
        
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
        
        # Apply data augmentation
        data_augmentation = self.get_data_augmentation()
        
        def augment_images_with_features(x, y):
            return ({"image_input": data_augmentation(x["image_input"], training=True), 
                   "feature_input": x["feature_input"]}, y)
        
        def augment_images_only(x, y):
            return (data_augmentation(x, training=True), y)
        
        if self.use_features:
            augmented_train_dataset = train_dataset.map(augment_images_with_features)
        else:
            augmented_train_dataset = train_dataset.map(augment_images_only)
        
        # Prepare test dataset
        if self.use_features:
            test_dataset = tf.data.Dataset.from_tensor_slices((
                {"image_input": X_images, "feature_input": X_features},
                y_labels
            )).batch(batch_size)
        else:
            test_dataset = tf.data.Dataset.from_tensor_slices((
                X_images,
                y_labels
            )).batch(batch_size)
        
        return augmented_train_dataset, test_dataset
        
    def train(self, train_dataset, validation_dataset, epochs=50, output_model_path='human_ai_detector_model.h5'):
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
    
    def predict(self, images, feature_maps=None):
        """
        Make predictions on new images.
        
        Args:
            images (np.ndarray): Input images
            feature_maps (np.ndarray, optional): Feature maps for the images
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        if self.use_features:
            return self.model.predict({
                'image_input': images,
                'feature_input': feature_maps
            })
        else:
            return self.model.predict(images)
    
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
    """
    def __init__(self, model_path=None, input_shape=(224, 224, 3), feature_channels=3, use_features=True):
        """
        Initialize the model wrapper.
        
        Args:
            model_path (str, optional): Path to load an existing model
            input_shape (tuple): Shape of input images
            feature_channels (int): Number of feature channels
            use_features (bool): Whether to use feature extraction
        """
        self.use_features = use_features
        self.model = AIDetectorModel(input_shape, feature_channels, use_features)
        if model_path:
            self.model.load(model_path)
            # Update use_features from loaded model
            self.use_features = self.model.use_features
    
    def train_model(self, X_train, X_train_features, y_train, 
                validation_split=0.2, epochs=50, batch_size=32, 
                model_path='human_ai_detector_model.h5'):
        """
        Train the model with validation split from training data
        
        Args:
            X_train (np.ndarray): Training images
            X_train_features (np.ndarray): Training feature maps (can be dummy if not using features)
            y_train (np.ndarray): Training labels
            validation_split (float): Proportion of training data to use for validation
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            model_path (str): Path to save the trained model
            
        Returns:
            dict: Training history
        """
        # Split training data into train/validation
        if 0 < validation_split < 1:
            from sklearn.model_selection import train_test_split
            indices = np.arange(len(X_train))
            train_idx, val_idx = train_test_split(
                indices, 
                test_size=validation_split,
                random_state=42,
                stratify=y_train
            )
            
            # Split images, features, and labels
            X_tr = X_train[train_idx]
            X_val = X_train[val_idx]
            X_tr_features = X_train_features[train_idx] if self.use_features else None
            X_val_features = X_train_features[val_idx] if self.use_features else None
            y_tr = y_train[train_idx]
            y_val = y_train[val_idx]
        else:
            X_tr, X_tr_features, y_tr = X_train, X_train_features, y_train
            X_val, X_val_features, y_val = None, None, None

        # Prepare augmented training dataset
        train_dataset, _ = self.model.prepare_datasets(
            X_tr, 
            X_tr_features if self.use_features else None,
            y_tr,
            test_split=0.0,  # No test split needed
            batch_size=batch_size
        )
        
        # Prepare validation dataset (no augmentation)
        val_dataset = None
        if X_val is not None:
            if self.use_features:
                val_dataset = tf.data.Dataset.from_tensor_slices(
                    ({"image_input": X_val, "feature_input": X_val_features}, y_val)
                ).batch(batch_size)
            else:
                val_dataset = tf.data.Dataset.from_tensor_slices(
                    (X_val, y_val)
                ).batch(batch_size)

        # Train the model
        history = self.model.train(
            train_dataset, 
            val_dataset, 
            epochs=epochs,
            output_model_path=model_path
        )
        
        return history
    
    def evaluate_model(self, X_test, X_test_features, y_test, batch_size=32):
        """
        Evaluate the model on test data.
        
        Args:
            X_test (np.ndarray): Test images
            X_test_features (np.ndarray): Test feature maps (can be dummy if not using features)
            y_test (np.ndarray): Test labels
            batch_size (int): Batch size for evaluation
            
        Returns:
            tuple: (test_loss, test_accuracy)
        """
        # Prepare test dataset
        if self.use_features:
            test_dataset = tf.data.Dataset.from_tensor_slices((
                {"image_input": X_test, "feature_input": X_test_features},
                y_test
            )).batch(batch_size)
        else:
            test_dataset = tf.data.Dataset.from_tensor_slices((
                X_test, y_test
            )).batch(batch_size)
        
        # Evaluate the model
        return self.model.evaluate(test_dataset)
    
    def predict_image(self, image, feature_map=None):
        """
        Make prediction on a single image.
        
        Args:
            image (np.ndarray): Input image (should be preprocessed)
            feature_map (np.ndarray, optional): Feature map for the image
            
        Returns:
            dict: Prediction results with label and confidence
        """
        # Ensure image has batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Ensure feature map has batch dimension if we're using features
        if self.use_features and len(feature_map.shape) == 3:
            feature_map = np.expand_dims(feature_map, axis=0)
        
        # Make prediction
        prediction = self.model.predict(image, feature_map)[0][0] if self.use_features else self.model.predict(image)[0][0]
        
        # Determine result
        is_ai = prediction > 0.5
        confidence = prediction if is_ai else 1 - prediction
        
        return {
            'is_ai': bool(is_ai),
            'label': "AI-generated" if is_ai else "Human-generated",
            'confidence': float(confidence),
            'raw_prediction': float(prediction)
        }
    
    def predict_batch(self, images, feature_maps=None):
        """
        Make predictions on a batch of images.
        
        Args:
            images (np.ndarray): Input images (should be preprocessed)
            feature_maps (np.ndarray, optional): Feature maps for the images
            
        Returns:
            list: List of prediction results with labels and confidences
        """
        # Make predictions
        if self.use_features:
            predictions = self.model.predict(images, feature_maps).flatten()
        else:
            predictions = self.model.predict(images).flatten()
        
        # Process results
        results = []
        for pred in predictions:
            is_ai = pred > 0.5
            confidence = pred if is_ai else 1 - pred
            
            results.append({
                'is_ai': bool(is_ai),
                'label': "AI-generated" if is_ai else "Human-generated",
                'confidence': float(confidence),
                'raw_score': float(pred)
            })
        
        return results
    
    def get_model(self):
        """
        Get the underlying model.
        
        Returns:
            AIDetectorModel: The model instance
        """
        return self.model
    
    def set_model(self, model):
        """
        Set the underlying model.
        
        Args:
            model: AIDetectorModel or Keras model to use
        """
        self.model = model
        if hasattr(model, 'use_features'):
            self.use_features = model.use_features
        else:
            # If it's a Keras model, infer from inputs
            self.use_features = len(model.inputs) > 1
    
    def save_model(self, filepath):
        """
        Save the model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        self.model.save(filepath)
    
    def load_model(self, filepath):
        """
        Load a saved model from disk.
        
        Args:
            filepath (str): Path to the saved model
        """
        self.model.load(filepath)
        # Update use_features from loaded model
        self.use_features = self.model.use_features