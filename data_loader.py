import os
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataLoader:
    """
    A memory-efficient data loader that uses TensorFlow's data pipeline
    to load and process images on-the-fly instead of loading everything into memory.
    """
    
    def __init__(self, config):
        """
        Initialize the data loader with configuration settings.
        
        Args:
            config: Configuration object with image_size, batch_size, etc.
        """
        self.image_size = config.image_size
        self.batch_size = config.batch_size
        self.random_seed = config.random_seed
        self.test_size = config.test_size
        self.max_images = config.max_images 
        
    def _parse_csv(self, csv_path):
        """
        Parse the CSV file containing image paths and labels.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            DataFrame with file paths and labels
        """
        df = pd.read_csv(csv_path)
        
        # Check that required columns exist
        required_cols = ['file_name', 'label']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")
        
        # Check for missing files
        missing_files = [path for path in df['file_name'] if not os.path.exists(path)]
        if missing_files:
            print(f"Warning: {len(missing_files)} image files not found")
            print(f"First few missing: {missing_files[:5] if len(missing_files) > 5 else missing_files}")
            
            # Remove missing files from dataframe
            df = df[df['file_name'].apply(os.path.exists)]
            
        if self.max_images is not None and self.max_images > 0:
            max_samples = min(self.max_images, len(df))
            df = df.sample(n=max_samples, random_state=self.random_seed)
            print(f"Subsampled to {max_samples} images based on config.max_images")

        print(f"Loaded CSV with {len(df)} valid image entries")
        return df


    def _process_path(self, file_path, label, use_center_crop=True):
        """
        Process a single image file path.

        Args:
            file_path: Path to the image file
            label: Label for the image
            use_center_crop: If True, use center crop instead of letterboxing

        Returns:
            Tuple of (processed_image, label)
        """
        # Read the image file
        img = tf.io.read_file(file_path)

        # Decode and process the image
        try:
            img = tf.image.decode_jpeg(img, channels=3)
        except:
            try:
                img = tf.image.decode_png(img, channels=3)
            except:
                img = tf.image.decode_image(img, channels=3, expand_animations=False)

        if use_center_crop:
            # Resize to slightly larger than target size to allow for crop
            resize_size = int(self.image_size * 1.15)
            img = tf.image.resize(img, [resize_size, resize_size])
            img = tf.image.central_crop(img, central_fraction=self.image_size / resize_size)
            img = tf.image.resize(img, [self.image_size, self.image_size])
        else:
            # Get original dimensions
            original_height = tf.shape(img)[0]
            original_width = tf.shape(img)[1]

            # Calculate scaling factor to preserve aspect ratio
            height_ratio = tf.cast(self.image_size, tf.float32) / tf.cast(original_height, tf.float32)
            width_ratio = tf.cast(self.image_size, tf.float32) / tf.cast(original_width, tf.float32)
            scale_ratio = tf.minimum(height_ratio, width_ratio)

            # Calculate new dimensions
            new_height = tf.cast(tf.cast(original_height, tf.float32) * scale_ratio, tf.int32)
            new_width = tf.cast(tf.cast(original_width, tf.float32) * scale_ratio, tf.int32)

            # Resize image while maintaining aspect ratio
            img = tf.image.resize(img, [new_height, new_width], preserve_aspect_ratio=True)

            # Use actual resized shape (in case resize rounds)
            resized_shape = tf.shape(img)
            pad_top = (self.image_size - resized_shape[0]) // 2
            pad_left = (self.image_size - resized_shape[1]) // 2

            # Pad image to fit target size
            img = tf.image.pad_to_bounding_box(
                img,
                offset_height=pad_top,
                offset_width=pad_left,
                target_height=self.image_size,
                target_width=self.image_size
            )

        # Normalize to [0, 1]
        img = tf.cast(img, tf.float32) / 255.0

        return img, label

    def process_path(self, file_path, label, use_center_crop=True):
        """
        Public wrapper for _process_path.
        """
        return self._process_path(file_path, label, use_center_crop)


    def _configure_for_performance(self, dataset, is_training=False):
        """
        Configure dataset for optimal performance.
        
        Args:
            dataset: TensorFlow dataset
            is_training: Whether this is for training (adds shuffling and augmentation)
            
        Returns:
            Optimized dataset
        """
        # Cache the dataset in memory for small datasets or to disk for larger ones
        # dataset = dataset.cache()
        
        if is_training:
            # Shuffle only training data
            dataset = dataset.shuffle(buffer_size=min(10000, self.batch_size * 100), 
                                     seed=self.random_seed)
            
            # Add data augmentation for training
            dataset = dataset.map(
                lambda x, y: (self._augment_image(x), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        # Batch the data
        dataset = dataset.batch(self.batch_size)
        
        # Prefetch for performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _augment_image(self, image):
        """
        Apply data augmentation to an image.
        
        Args:
            image: Input image tensor
            
        Returns:
            Augmented image tensor
        """
        # Random flips
        image = tf.image.random_flip_left_right(image)
        
        # Random brightness, contrast, and saturation adjustments
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        
        # Make sure values stay in [0, 1] range
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image
    
    def create_datasets(self, config):
        """
        Create train and test datasets from CSV file.
        
        Args:
            csv_path: Path to CSV with image paths and labels
            
        Returns:
            Tuple of (train_ds, val_ds)
        """
        # Parse CSV
        df = self._parse_csv(config.data)
        
        # Split into train and test
        train_df, test_df = train_test_split(
            df, 
            test_size=self.test_size, 
            random_state=self.random_seed, 
            stratify=df['label'] 
        )
        
        print(f"Split data into {len(train_df)} training and {len(test_df)} testing samples")
        
        # Create TensorFlow datasets
        train_ds = tf.data.Dataset.from_tensor_slices(
            (train_df['file_name'].values, train_df['label'].values)
        )
        test_ds = tf.data.Dataset.from_tensor_slices(
            (test_df['file_name'].values, test_df['label'].values)
        )
        
        # Process paths and load images on-the-fly
        train_ds = train_ds.map(
            lambda path, label: self._process_path(path, label),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        test_ds = test_ds.map(
            lambda path, label: self._process_path(path, label),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Configure for performance
        train_ds = self._configure_for_performance(train_ds, is_training=True)
        test_ds = self._configure_for_performance(test_ds, is_training=False)
        
        # Create a small sample dataset for feature extractor/genetic algorithm
        # to avoid loading all images for those steps
        sample_size = min(1000, len(train_df))
        print(f"Creating sample dataset of {sample_size} images for feature extraction")
        
        sample_df = train_df.sample(sample_size, random_state=self.random_seed)
        sample_images = []
        sample_labels = []
        
        if config.use_genetic_algorithm:
            # Load a small batch for feature extraction/genetic algorithm
            for path, label in zip(sample_df['file_name'], sample_df['label']):
                try:
                    # Use the internal _process_path method for consistent preprocessing
                    img_tensor, lbl = self._process_path(path, label, use_center_crop=False)
                    # Convert TensorFlow tensor to numpy array
                    img_array = img_tensor.numpy()
                    sample_images.append(img_array)
                    sample_labels.append(lbl)
                except Exception as e:
                    print(f"Error loading sample image {path}: {e}")

            sample_images = np.array(sample_images)
            sample_labels = np.array(sample_labels)
            
        return train_ds, test_ds, sample_images, sample_labels