import numpy as np


def convert_patch_mask_to_pixel_mask(patch_mask, 
                                   image_shape=None, 
                                   patch_size=None):
    """
    Convert a 2D patch mask to a pixel-level mask.
    Handles both fixed-size patches and variable patches based on image dimensions.
    
    Args:
        patch_mask (np.ndarray): 2D array where 1 indicates selected patches
        image_shape (tuple): Optional (height, width) of original image for variable patches
        patch_size (int or tuple): Optional fixed patch size (h, w) or single int for square
    
    Returns:
        np.ndarray: Pixel-level mask with 1s in selected patch areas
    """
    if len(patch_mask.shape) != 2:
        raise ValueError("Patch mask must be a 2D array")

    # Calculate patch dimensions
    if patch_size is not None:
        # Handle fixed patch size
        if isinstance(patch_size, int):
            patch_h = patch_w = patch_size
        else:
            patch_h, patch_w = patch_size
    elif image_shape is not None:
        # Calculate variable patch size from image dimensions
        img_h, img_w = image_shape[:2]
        patch_h = img_h // patch_mask.shape[0]
        patch_w = img_w // patch_mask.shape[1]
    else:
        raise ValueError("Must provide either image_shape or patch_size")

    # Initialize pixel mask
    if image_shape is not None:
        pixel_mask = np.zeros(image_shape[:2])
    else:
        # Calculate total size from patch dimensions
        total_h = patch_mask.shape[0] * patch_h
        total_w = patch_mask.shape[1] * patch_w
        pixel_mask = np.zeros((total_h, total_w))

    # Populate pixel mask
    for i in range(patch_mask.shape[0]):
        for j in range(patch_mask.shape[1]):
            if patch_mask[i, j] == 1:
                h_start = i * patch_h
                h_end = (i+1) * patch_h
                w_start = j * patch_w
                w_end = (j+1) * patch_w
                
                # Handle edge cases where image dimensions don't divide evenly
                if image_shape is not None:
                    h_end = min(h_end, image_shape[0])
                    w_end = min(w_end, image_shape[1])
                
                pixel_mask[h_start:h_end, w_start:w_end] = 1

    return pixel_mask