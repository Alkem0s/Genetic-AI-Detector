# utils.py
import tensorflow as tf
import global_config as config


def get_timestamp():
    import datetime
    return datetime.datetime.now().isoformat()


@tf.function
def convert_patch_mask_to_pixel_mask(patch_mask):
    """
    Convert patch masks to pixel-level masks with explicit shape handling
    
    Args:
        patch_mask (tf.Tensor): 2D or 3D binary mask:
            - Single mask: (n_patches_h, n_patches_w)
            - Batch of masks: (batch_size, n_patches_h, n_patches_w)
    
    Returns:
        tf.Tensor: Pixel-level mask(s) of shape (height, width) or (batch_size, height, width)
    """
    # Get configuration values
    patch_size = config.patch_size
    image_shape = (config.image_size, config.image_size)
    
    # Get static rank if available
    input_rank = patch_mask.shape.rank
    
    # Handle different input dimensions
    if input_rank == 2:
        patch_mask = tf.expand_dims(patch_mask, 0)
    elif input_rank is None:  # Handle unknown rank
        patch_mask = tf.cond(
            tf.equal(tf.rank(patch_mask), 2),
            lambda: tf.expand_dims(patch_mask, 0),
            lambda: patch_mask
        )
    
    # Add channel dimension
    expanded_mask = tf.expand_dims(tf.cast(patch_mask, tf.float32), axis=-1)
    
    # Set known dimensions for the tensor
    expanded_mask = tf.ensure_shape(expanded_mask, [None, None, None, 1])
    
    # Resize with nearest neighbor
    pixel_mask = tf.image.resize(
        expanded_mask,
        image_shape,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    
    # Remove channel dimension
    pixel_mask = tf.squeeze(pixel_mask, axis=-1)
    
    # Remove batch dimension if input was 2D
    if input_rank == 2:
        pixel_mask = tf.squeeze(pixel_mask, axis=0)
    elif input_rank is None:
        pixel_mask = tf.cond(
            tf.equal(tf.rank(patch_mask), 3),
            lambda: tf.squeeze(pixel_mask, axis=0),
            lambda: pixel_mask
        )
    
    return tf.cast(pixel_mask >= 0.5, tf.float32)


@tf.function
def remove_black_bars(img, threshold=10):
    """
    Remove black padding bars from an image while maintaining content aspect ratio.
    Handles TensorFlow Tensors.
    
    Args:
        img: Input image (tf.Tensor)
        threshold: Intensity threshold for considering pixels as black (0-255)
    
    Returns:
        Cropped image (tf.Tensor) without black bars
    """
    # Ensure image is float for calculations, then convert to uint8 for consistency if needed
    original_dtype = img.dtype
    img_float = tf.cast(img, tf.float32)

    # Convert to grayscale for bar detection
    if len(img_float.shape) == 3 and img_float.shape[-1] == 3: # RGB
        gray = tf.image.rgb_to_grayscale(img_float)
    else: # Grayscale or already processed
        gray = img_float

    # Squeeze the channel dimension for 2D mask operations
    gray = tf.squeeze(gray)

    # Find content boundaries
    mask = gray > tf.cast(threshold, tf.float32)
    
    # Check if the entire image is black
    if tf.reduce_all(tf.logical_not(mask)):
        return img

    # Find the indices of True values (non-black pixels)
    coords = tf.where(mask)
    
    # Get min/max coordinates
    y0 = tf.reduce_min(coords[:, 0])
    x0 = tf.reduce_min(coords[:, 1])
    y1 = tf.reduce_max(coords[:, 0]) + 1
    x1 = tf.reduce_max(coords[:, 1]) + 1
    
    # Crop image using tf.slice
    cropped = tf.slice(img, [y0, x0, 0] if len(img.shape) == 3 else [y0, x0], 
                             [y1 - y0, x1 - x0, img.shape[-1]] if len(img.shape) == 3 else [y1 - y0, x1 - x0])
    
    return tf.cast(cropped, original_dtype)


# ---------------------------------------------------------------------------
# JPEG Robustness Utilities
# ---------------------------------------------------------------------------

def apply_jpeg_compression(image_tensor: tf.Tensor, quality: int) -> tf.Tensor:
    """
    Encode and decode a float32 image tensor through JPEG at a given quality level.

    Args:
        image_tensor: Float32 tensor in [0, 1] of shape (H, W, 3).
        quality:      JPEG quality level (0–100).  Lower = more compression loss.

    Returns:
        Float32 tensor in [0, 1] of the same shape after JPEG round-trip.
    """
    # Convert [0, 1] → uint8 for JPEG codec
    uint8_img = tf.cast(tf.clip_by_value(image_tensor * 255.0, 0, 255), tf.uint8)
    # Encode as JPEG bytes
    jpeg_bytes = tf.image.encode_jpeg(uint8_img, quality=quality)
    # Decode back to uint8 tensor
    decoded = tf.image.decode_jpeg(jpeg_bytes, channels=3)
    # Restore [0, 1] float32
    return tf.cast(decoded, tf.float32) / 255.0


def evaluate_robustness(model_wrapper, test_ds, quality_levels=None, logger=None):
    """
    Evaluate model accuracy after JPEG compression at several quality levels.

    Runs *test_ds* through ``apply_jpeg_compression`` at each quality level,
    evaluates on the compressed images, and returns accuracy drop relative to
    the clean baseline.

    Args:
        model_wrapper:  A ModelWrapper instance (must already have features
                        prepared if use_features is True).
        test_ds:        Raw tf.data.Dataset yielding (images, labels) batches.
        quality_levels: List of JPEG quality ints to test (default: from config).
        logger:         Optional Python logger; falls back to print().

    Returns:
        dict: {quality_level: {'accuracy': float, 'accuracy_drop': float}}
    """
    import logging as _logging
    import global_config as _cfg

    if quality_levels is None:
        quality_levels = getattr(_cfg, 'jpeg_quality_levels', [50, 75])

    _log = logger or _logging.getLogger(__name__)

    def _log_info(msg):
        if logger:
            logger.info(msg)
        else:
            _log.info(msg)

    # ---- Baseline accuracy on clean images ----------------------------------------
    if model_wrapper.use_features:
        model_wrapper.precompute_features(test_ds, "robustness_clean")
        clean_prepared = model_wrapper.prepare_dataset(test_ds, "robustness_clean")
    else:
        clean_prepared = test_ds

    clean_results = model_wrapper.model.model.evaluate(clean_prepared, verbose=0)
    baseline_acc = clean_results[1] if len(clean_results) > 1 else float('nan')
    _log_info(f"[Robustness] Baseline accuracy (no JPEG): {baseline_acc:.4f}")

    results = {}

    for q in quality_levels:
        # Apply JPEG compression to every image in the dataset
        def compress_batch(images, labels, _q=q):
            compressed = tf.map_fn(
                lambda img: apply_jpeg_compression(img, _q),
                images,
                fn_output_signature=tf.float32,
            )
            return compressed, labels

        compressed_ds = test_ds.map(compress_batch, num_parallel_calls=tf.data.AUTOTUNE)

        if model_wrapper.use_features:
            cache_name = f"robustness_q{q}"
            model_wrapper.precompute_features(compressed_ds, cache_name)
            prepared = model_wrapper.prepare_dataset(compressed_ds, cache_name)
        else:
            prepared = compressed_ds

        eval_results = model_wrapper.model.model.evaluate(prepared, verbose=0)
        acc = eval_results[1] if len(eval_results) > 1 else float('nan')
        drop = baseline_acc - acc

        _log_info(
            f"[Robustness] JPEG quality={q}: accuracy={acc:.4f}, "
            f"drop={drop:+.4f}"
        )
        results[q] = {'accuracy': acc, 'accuracy_drop': drop}

    return results