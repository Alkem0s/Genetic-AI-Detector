import os
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import global_config as config

class DataLoader:
    """
    A memory-efficient data loader that uses TensorFlow's data pipeline
    to load and process images on-the-fly instead of loading everything into memory.
    """
    
    def __init__(self):
        """
        Initialize the data loader with configuration settings.
        """
        self.image_size = config.image_size
        self.batch_size = config.cnn_batch_size
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
        
        # Print original label distribution
        print("Original label distribution:")
        label_counts = df['label'].value_counts().sort_index()
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  Label {label}: {count} images ({percentage:.1f}%)")
            
        if self.max_images is not None and self.max_images > 0:
            max_samples = min(self.max_images, len(df))
            
            # Use stratified sampling to maintain label balance
            try:
                df = self._stratified_subsample(df, max_samples)
                print(f"Subsampled to {max_samples} images using stratified sampling")
                
                # Print subsampled label distribution
                print("Subsampled label distribution:")
                label_counts = df['label'].value_counts().sort_index()
                for label, count in label_counts.items():
                    percentage = (count / len(df)) * 100
                    print(f"  Label {label}: {count} images ({percentage:.1f}%)")
                    
            except Exception as e:
                print(f"Warning: Stratified sampling failed ({e}), using random sampling")
                df = df.sample(n=max_samples, random_state=self.random_seed)
                print(f"Subsampled to {max_samples} images using random sampling")

        print(f"Loaded CSV with {len(df)} valid image entries")
        return df

    def _stratified_subsample(self, df, max_samples):
        """
        Perform stratified subsampling to maintain label proportions.
        
        Args:
            df: Original DataFrame
            max_samples: Maximum number of samples to keep
            
        Returns:
            Subsampled DataFrame with balanced labels
        """
        # Get unique labels and their counts
        label_counts = df['label'].value_counts()
        unique_labels = label_counts.index.tolist()
        
        # Calculate minimum samples per label to ensure representation
        min_samples_per_label = max(1, max_samples // (len(unique_labels) * 4))  # At least 1, but allow some imbalance
        
        subsampled_dfs = []
        remaining_samples = max_samples
        
        # First pass: ensure minimum representation for each label
        for label in unique_labels:
            label_df = df[df['label'] == label]
            available_samples = len(label_df)
            
            if available_samples == 0:
                continue
                
            # Take minimum samples or all available if less than minimum
            samples_to_take = min(min_samples_per_label, available_samples, remaining_samples)
            
            if samples_to_take > 0:
                sampled_df = label_df.sample(n=samples_to_take, random_state=self.random_seed)
                subsampled_dfs.append(sampled_df)
                remaining_samples -= samples_to_take
        
        # Second pass: distribute remaining samples proportionally
        if remaining_samples > 0:
            # Calculate proportions based on original distribution
            total_original = len(df)
            
            for label in unique_labels:
                if remaining_samples <= 0:
                    break
                    
                label_df = df[df['label'] == label]
                already_sampled = sum(len(sub_df[sub_df['label'] == label]) 
                                    for sub_df in subsampled_dfs)
                available_samples = len(label_df) - already_sampled
                
                if available_samples <= 0:
                    continue
                
                # Calculate proportional additional samples
                original_proportion = len(label_df) / total_original
                additional_samples = int(remaining_samples * original_proportion)
                additional_samples = min(additional_samples, available_samples, remaining_samples)
                
                if additional_samples > 0:
                    # Get samples not already selected
                    already_selected_indices = set()
                    for sub_df in subsampled_dfs:
                        if label in sub_df['label'].values:
                            already_selected_indices.update(sub_df[sub_df['label'] == label].index)
                    
                    available_df = label_df[~label_df.index.isin(already_selected_indices)]
                    
                    if len(available_df) >= additional_samples:
                        additional_sampled = available_df.sample(n=additional_samples, 
                                                               random_state=self.random_seed + label)
                        subsampled_dfs.append(additional_sampled)
                        remaining_samples -= additional_samples
        
        # Combine all subsampled dataframes
        result_df = pd.concat(subsampled_dfs, ignore_index=True)
        
        # Shuffle the final result
        result_df = result_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        return result_df

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
        
        if is_training:
            # Shuffle only training data
            dataset = dataset.shuffle(buffer_size=min(10000, self.batch_size * 100), 
                                     seed=self.random_seed)
            
            if config.use_augmentation:
                # Use data augmentation for training
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
    
    def create_datasets(self):
        """
        Create train and test datasets from CSV file.
        
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
        sample_size = config.sample_size
        print(f"Creating sample dataset of {sample_size} images for feature extraction")
        
        # Use stratified sampling for the sample dataset too
        try:
            sample_df = self._stratified_subsample(train_df, sample_size)
            print("Sample dataset label distribution:")
            sample_label_counts = sample_df['label'].value_counts().sort_index()
            for label, count in sample_label_counts.items():
                percentage = (count / len(sample_df)) * 100
                print(f"  Label {label}: {count} images ({percentage:.1f}%)")
        except:
            print("Using random sampling for sample dataset")
            sample_df = train_df.sample(sample_size, random_state=self.random_seed)
        
        sample_images = []
        sample_labels = []
        
        if config.use_feature_extraction:
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