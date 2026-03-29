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