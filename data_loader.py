import os
import tensorflow as tf
import pandas as pd
import numpy as np
import PIL.Image
import time
from sklearn.model_selection import train_test_split
from pathlib import Path
import global_config as config
import logging

logger = logging.getLogger(__name__)

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
        
    def _crawl_generators(self, generator_list, split_type=None, limit_per_cls=None):
        """
        Crawl the dataset_sampled directory for specific generators.
        
        Args:
            generator_list: List of generator names to include
            split_type: Optional filter for 'train' or 'val' folders
            
        Returns:
            DataFrame with file paths and labels
        """
        data = []
        base_dir = Path(config.dataset_sampled_dir)
        
        for gen in generator_list:
            gen_path = base_dir / gen
            if not gen_path.exists():
                logger.warning(f"Warning: Generator folder '{gen}' not found in {base_dir}")
                continue
                
            # If split_type is specified (e.g. 'train'), only look in that subfolder
            splits = [split_type] if split_type else ["train", "val"]
            
            for split in splits:
                split_path = gen_path / split
                if not split_path.exists():
                    continue
                
                cls_data = {}
                for cls_name, label in [("ai", 1), ("real", 0)]:
                    cls_path = split_path / cls_name
                    if not cls_path.exists():
                        continue
                        
                    # Get all image files
                    all_imgs = [f for f in cls_path.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
                    # Sort to guarantee filesystem-independent order across runs
                    all_imgs.sort(key=lambda x: x.name)
                    
                    # Randomly shuffle before checking to ensure variety if we hit the limit
                    import random
                    seed_val = self.random_seed if self.random_seed is not None else 42
                    local_rand = random.Random(seed_val)
                    local_rand.shuffle(all_imgs)
                    
                    if limit_per_cls:
                        valid_imgs = all_imgs[:limit_per_cls]
                    else:
                        valid_imgs = all_imgs
                        
                    logger.info(f"    Found {len(valid_imgs)} valid images in {cls_path.parent.name}/{cls_name}")
                    
                    cls_data[cls_name] = valid_imgs

                # --- PER-GENERATOR BALANCING ---
                n_ai = len(cls_data.get("ai", []))
                n_real = len(cls_data.get("real", []))
                n_match = min(n_ai, n_real)
                
                if n_match > 0:
                    for cls_name in ["ai", "real"]:
                        if cls_name in cls_data:
                            selected = cls_data[cls_name][:n_match]
                            label = 1 if cls_name == "ai" else 0
                            for img_file in selected:
                                data.append({
                                    'file_name': str(img_file),
                                    'label': label,
                                    'generator': gen,
                                    'split': split
                                })
                            
        df = pd.DataFrame(data)
        if not df.empty:
            logger.info(f"Crawled {len(df)} images from {len(generator_list)} generators (balanced per-generator).")
        return df

    def _stratified_subsample(self, df, max_samples):
        """
        Subsample the dataframe to a maximum number of samples while preserving
        both label and generator proportions if both columns exist, otherwise fallback
        to label proportions.
        """
        if df.empty or len(df) <= max_samples:
            return df

        # Determine stratification columns
        strat_cols = []
        if 'generator' in df.columns:
            strat_cols.append('generator')
        if 'label' in df.columns:
            strat_cols.append('label')

        if not strat_cols:
            # Fallback to simple random sampling if no stratification columns are present
            return df.sample(n=max_samples, random_state=self.random_seed).reset_index(drop=True)

        # Get the group counts
        group_counts = df.groupby(strat_cols).size()
        total_samples = len(df)
        
        # Calculate target samples per group based on original proportions
        target_samples_per_group = {}
        allocated_samples = 0
        
        for group, count in group_counts.items():
            proportion = count / total_samples
            target_count = int(max_samples * proportion)
            # Ensure at least 1 sample if group exists and we have room
            if target_count == 0 and count > 0 and allocated_samples < max_samples:
                target_count = 1
            target_count = min(target_count, count)
            target_samples_per_group[group] = target_count
            allocated_samples += target_count

        # Distribute remaining samples
        remaining_samples = max_samples - allocated_samples
        if remaining_samples > 0:
            available_additional = {group: max(0, count - target_samples_per_group[group]) 
                                    for group, count in group_counts.items()}
            total_additional_available = sum(available_additional.values())
            
            if total_additional_available > 0:
                for group in group_counts.index:
                    if available_additional[group] > 0:
                        proportion = available_additional[group] / total_additional_available
                        additional = min(int(remaining_samples * proportion), available_additional[group])
                        target_samples_per_group[group] += additional
                        remaining_samples -= additional
                        if remaining_samples <= 0:
                            break
                            
            # Add one by one if there are still remaining due to rounding
            while remaining_samples > 0:
                added_any = False
                for group in group_counts.index:
                    if remaining_samples <= 0:
                        break
                    if target_samples_per_group[group] < group_counts[group]:
                        target_samples_per_group[group] += 1
                        remaining_samples -= 1
                        added_any = True
                if not added_any:
                    break

        # Sample from each group according to target counts
        subsampled_dfs = []
        for group, target_count in target_samples_per_group.items():
            if target_count > 0:
                # Filter df for this group
                if len(strat_cols) == 1:
                    mask = (df[strat_cols[0]] == group)
                else:
                    mask = (df[strat_cols[0]] == group[0]) & (df[strat_cols[1]] == group[1])
                group_df = df[mask]
                sampled_df = group_df.sample(n=target_count, random_state=self.random_seed)
                subsampled_dfs.append(sampled_df)

        # Combine and shuffle
        result_df = pd.concat(subsampled_dfs, ignore_index=True)
        result_df = result_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)

        # Print allocation summary
        logger.info(f"Stratified sampling allocated {len(result_df)} samples:")
        actual_counts = result_df.groupby(strat_cols).size()
        for group in group_counts.index:
            original = group_counts[group]
            target = target_samples_per_group[group]
            actual = actual_counts.get(group, 0)
            original_pct = (original / total_samples) * 100
            actual_pct = (actual / len(result_df)) * 100 if len(result_df) > 0 else 0
            group_name = f"{group[0]}/{group[1]}" if len(strat_cols) == 2 else str(group)
            logger.info(f"  Group {group_name}: {actual}/{original} samples ({original_pct:.1f}% → {actual_pct:.1f}%)")
        
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

        # Get actual dimensions
        img_shape = tf.shape(img)
        h, w = tf.cast(img_shape[0], tf.float32), tf.cast(img_shape[1], tf.float32)

        # 1. Check for exact match to bypass heavy processing
        is_exact_match = tf.logical_and(
            tf.equal(h, self.image_size),
            tf.equal(w, self.image_size)
        )

        if is_exact_match:
            # Already perfect
            img = tf.cast(img, tf.float32) / 255.0
            img.set_shape([self.image_size, self.image_size, 3])
            return img, label

        # 3. Process non-square or larger images
        if use_center_crop:
            # PREVENT BIAS: Proportional resize instead of stretching
            # We resize so the shorter side is exactly self.image_size
            scale = self.image_size / tf.reduce_min([h, w])
            new_h = tf.cast(h * scale, tf.int32)
            new_w = tf.cast(w * scale, tf.int32)
            
            # Proportional resize (no squishing)
            img = tf.image.resize(img, [new_h, new_w], method=tf.image.ResizeMethod.BILINEAR)
            
            # Center crop to target size
            img = tf.image.resize_with_crop_or_pad(img, self.image_size, self.image_size)
        else:
            # Letterbox approach (preserve full content but add black bars)
            # This can introduce "padding bias", so center_crop is generally preferred for AI detection
            img = tf.image.resize(img, [self.image_size, self.image_size], preserve_aspect_ratio=True)
            img = tf.image.resize_with_crop_or_pad(img, self.image_size, self.image_size)

        # Normalize to [0, 1]
        img = tf.cast(img, tf.float32) / 255.0
        img.set_shape([self.image_size, self.image_size, 3])

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
        image.set_shape([self.image_size, self.image_size, 3])
        
        return image
    
    def _balance_generators(self, df, stage_name):
        """
        Ensure all generators contribute an equal number of samples.
        """
        if df.empty:
            return df
            
        # Find min count across all (generator, label) pairs
        counts = df.groupby(['generator', 'label']).size()
        min_count = counts.min()
        
        logger.info(f"  [{stage_name}] Balancing generators to {min_count} images per class/model...")
        
        balanced_dfs = []
        for (gen, label), group in df.groupby(['generator', 'label']):
            balanced_dfs.append(group.sample(n=min_count, random_state=self.random_seed))
            
        return pd.concat(balanced_dfs).sample(frac=1, random_state=self.random_seed).reset_index(drop=True)

    def create_datasets(self, create_sample=True):
        """
        Create train and test datasets from folder structure.
        
        Returns:
            Tuple of (train_ds, val_ds, sample_images, sample_labels)
        """
        logger.info("Using folder-based dataset discovery...")
        
        train_gens = config.train_generators
        val_gens = config.val_generators
        
        # If no generators specified, discover all folders in dataset_sampled
        if not train_gens:
            base_dir = Path(config.dataset_sampled_dir)
            if base_dir.exists():
                train_gens = [d.name for d in base_dir.iterdir() if d.is_dir()]
                logger.info(f"Auto-discovered generators: {train_gens}")
            else:
                raise FileNotFoundError(f"Dataset directory {config.dataset_sampled_dir} not found.")
        
        if not val_gens:
            val_gens = train_gens # Use same generators for validation by default
            
        train_df = self._crawl_generators(train_gens, split_type="train", limit_per_cls=config.max_train_per_gen)
        test_df = self._crawl_generators(val_gens, split_type="val", limit_per_cls=config.max_val_per_gen)
        
        # --- GLOBAL GENERATOR BALANCING ---
        # Ensure every generator contributes the same amount to prevent bias
        train_df = self._balance_generators(train_df, "Training")
        test_df = self._balance_generators(test_df, "Validation/Test")
        
        # --- TOTAL SAMPLE SIZE SUBSAMPLING (with automatic stratification) ---
        max_train = getattr(config, 'max_train_samples', None)
        if max_train is not None and not train_df.empty and len(train_df) > max_train:
            logger.info(f"Subsampling balanced training dataset from {len(train_df)} to {max_train} total samples...")
            train_df = self._stratified_subsample(train_df, max_train)
            
        max_val = getattr(config, 'max_val_samples', None)
        if max_val is not None and not test_df.empty and len(test_df) > max_val:
            logger.info(f"Subsampling balanced validation dataset from {len(test_df)} to {max_val} total samples...")
            test_df = self._stratified_subsample(test_df, max_val)
        
        if train_df.empty:
            raise ValueError(f"No training data found in {config.dataset_sampled_dir}")
        
        if test_df.empty:
            logger.warning("Warning: No validation data found in specified generators. Splitting from training.")
            train_df, test_df = train_test_split(
                train_df, test_size=self.test_size, random_state=self.random_seed, stratify=train_df['label']
            )
        
        logger.info(f"Final setup: {len(train_df)} training and {len(test_df)} testing samples")
        
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
        
        # Configure for performance (kept unbatched and unshuffled; batching and shuffling
        # are handled inside ModelWrapper / FeaturePipeline)
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
        
        sample_images = np.array([])
        sample_labels = np.array([])
        
        if create_sample:
            # Create a small sample dataset for feature extractor/genetic algorithm
            # to avoid loading all images for those steps
            sample_size = config.sample_size
            logger.info(f"Creating sample dataset of {sample_size} images for feature extraction")
            
            # Use stratified sampling for the sample dataset too
            try:
                sample_df = self._stratified_subsample(train_df, sample_size)
                logger.info("Sample dataset label distribution:")
                sample_label_counts = sample_df['label'].value_counts().sort_index()
                for label, count in sample_label_counts.items():
                    percentage = (count / len(sample_df)) * 100
                    logger.info(f"  Label {label}: {count} images ({percentage:.1f}%)")
            except:
                logger.info("Using random sampling for sample dataset")
                sample_df = train_df.sample(sample_size, random_state=self.random_seed)
            
            sample_imgs_list = []
            sample_lbls_list = []
            
            if config.use_feature_extraction:
                # Use tf.data pipeline for fast parallel loading
                sample_ds = tf.data.Dataset.from_tensor_slices(
                    (sample_df['file_name'].values, sample_df['label'].values)
                )
                sample_ds = sample_ds.map(
                    lambda path, label: self._process_path(path, label, use_center_crop=True),
                    num_parallel_calls=tf.data.AUTOTUNE
                ).batch(256).prefetch(tf.data.AUTOTUNE)
                
                for img_batch, lbl_batch in sample_ds:
                    sample_imgs_list.append(img_batch.numpy())
                    sample_lbls_list.append(lbl_batch.numpy())
                    
                if sample_imgs_list:
                    sample_images = np.concatenate(sample_imgs_list, axis=0)
                    sample_labels = np.concatenate(sample_lbls_list, axis=0)
            
        return train_ds, test_ds, sample_images, sample_labels

    def create_val_sample(
        self,
        sample_size: int = None,
        generators: list = None,
    ):
        """
        Load a fixed-size, balanced image sample from the *validation* generators
        (``config.val_generators``) using the ``val`` split of the dataset.

        This is the cross-generator held-out set — it must NEVER be used during
        optimisation, only for the final "Honest Validation" step.

        Args:
            sample_size: Number of images to return.  Defaults to
                         ``config.max_val_per_gen * len(val_generators)``.
            generators:  Override the generator list (defaults to
                         ``config.val_generators``).

        Returns:
            Tuple of (images np.ndarray [N, H, W, 3], labels np.ndarray [N]).
        """
        import numpy as np

        gens = generators if generators is not None else config.val_generators
        if not gens:
            raise ValueError(
                "config.val_generators is empty — cannot create validation sample."
            )

        # Crawl the val split for the requested generators
        val_df = self._crawl_generators(
            gens,
            split_type="val",
            limit_per_cls=config.max_val_per_gen,
        )

        if val_df.empty:
            raise ValueError(
                f"No validation images found for generators: {gens}.  "
                f"Check that '{config.dataset_sampled_dir}/<gen>/val/' exists."
            )

        # Generator-level balancing (same logic as create_datasets)
        val_df = self._balance_generators(val_df, "ValSample")

        # Cap to requested sample size while preserving label balance
        default_size = config.max_val_per_gen * len(gens) * 2  # *2 for ai+real
        n = sample_size if sample_size is not None else default_size
        val_df = self._stratified_subsample(val_df, n)

        logger.info(
            f"Val sample: {len(val_df)} images from generators {gens} "
            f"(split=val, limit_per_cls={config.max_val_per_gen})"
        )

        # Load images using the same tf.data pipeline as the training sample
        val_ds = tf.data.Dataset.from_tensor_slices(
            (val_df["file_name"].values, val_df["label"].values)
        ).map(
            lambda path, label: self._process_path(path, label, use_center_crop=True),
            num_parallel_calls=tf.data.AUTOTUNE,
        ).batch(256).prefetch(tf.data.AUTOTUNE)

        images_list, labels_list = [], []
        for img_batch, lbl_batch in val_ds:
            images_list.append(img_batch.numpy())
            labels_list.append(lbl_batch.numpy())

        if not images_list:
            raise RuntimeError("Val sample pipeline produced no images.")

        return np.concatenate(images_list, axis=0), np.concatenate(labels_list, axis=0)