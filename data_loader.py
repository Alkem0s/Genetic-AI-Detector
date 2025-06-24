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
        # Get label counts and proportions
        label_counts = df['label'].value_counts()
        total_samples = len(df)
        
        # Calculate target samples per label based on original proportions
        target_samples_per_label = {}
        allocated_samples = 0
        
        # First, calculate proportional allocation
        for label in label_counts.index:
            proportion = label_counts[label] / total_samples
            target_count = int(max_samples * proportion)
            
            # Ensure at least 1 sample per label if the label exists and we have room
            if target_count == 0 and label_counts[label] > 0 and allocated_samples < max_samples:
                target_count = 1
            
            # Don't exceed available samples for this label
            target_count = min(target_count, label_counts[label])
            
            target_samples_per_label[label] = target_count
            allocated_samples += target_count
        
        # If we haven't allocated all samples, distribute the remainder
        # proportionally among labels that have more samples available
        remaining_samples = max_samples - allocated_samples
        
        if remaining_samples > 0:
            # Calculate how many additional samples we can take from each label
            available_additional = {}
            for label in label_counts.index:
                available_additional[label] = max(0, label_counts[label] - target_samples_per_label[label])
            
            # Distribute remaining samples proportionally among labels with availability
            total_additional_available = sum(available_additional.values())
            
            if total_additional_available > 0:
                for label in label_counts.index:
                    if available_additional[label] > 0:
                        proportion = available_additional[label] / total_additional_available
                        additional = min(
                            int(remaining_samples * proportion),
                            available_additional[label]
                        )
                        target_samples_per_label[label] += additional
                        remaining_samples -= additional
                        
                        if remaining_samples <= 0:
                            break
            
            # If there are still remaining samples, add them one by one to labels with availability
            while remaining_samples > 0:
                added_any = False
                for label in label_counts.index:
                    if remaining_samples <= 0:
                        break
                    if target_samples_per_label[label] < label_counts[label]:
                        target_samples_per_label[label] += 1
                        remaining_samples -= 1
                        added_any = True
                
                # If we can't add any more samples, break to avoid infinite loop
                if not added_any:
                    break
        
        # Sample from each label according to target counts
        subsampled_dfs = []
        for label, target_count in target_samples_per_label.items():
            if target_count > 0:
                label_df = df[df['label'] == label]
                sampled_df = label_df.sample(n=target_count, random_state=self.random_seed)
                subsampled_dfs.append(sampled_df)
        
        # Combine and shuffle
        result_df = pd.concat(subsampled_dfs, ignore_index=True)
        result_df = result_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        # Print allocation summary
        print(f"Stratified sampling allocated {len(result_df)} samples:")
        actual_counts = result_df['label'].value_counts().sort_index()
        for label in sorted(target_samples_per_label.keys()):
            original = label_counts[label]
            target = target_samples_per_label[label]
            actual = actual_counts.get(label, 0)
            original_pct = (original / total_samples) * 100
            actual_pct = (actual / len(result_df)) * 100 if len(result_df) > 0 else 0
            print(f"  Label {label}: {actual}/{original} samples ({original_pct:.1f}% → {actual_pct:.1f}%)")
        
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