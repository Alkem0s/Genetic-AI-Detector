import numpy as np
import os
from collections import Counter
import numpy as np
from PIL import Image
from PIL import ExifTags
import os
import imagehash
import warnings

from ai_detection_config import AIDetectionConfig
from structural_features import StructuralFeaturesExtractor
from texture_features import TextureFeaturesExtractor
from utils import convert_patch_mask_to_pixel_mask

class AIFeatureExtractor:
    """
    Extract features from images that may indicate AI generation
    Enhanced with multiple detection strategies for better accuracy
    """
    @staticmethod
    def analyze_metadata(image_path):
        """
        Analyze image metadata for AI-generation indicators with improved false positive handling.
        Return a confidence score for AI generation based on metadata.
        
        Improvements:
        - Added whitelist for common photo software
        - Improved scoring with confidence weights
        - Better handling of missing metadata (common in both AI and natural images)
        - More comprehensive detection of AI-specific metadata patterns
        """
        metadata_score = 0
        metadata_findings = {}
        total_weight = 0
        
        # Common legitimate image editors and camera software
        legitimate_software = [
            'photoshop', 'lightroom', 'capture one', 'luminar', 'affinity photo',
            'gimp', 'pixelmator', 'camera raw', 'dxo', 'acdsee', 'canon', 'nikon',
            'sony', 'fujifilm', 'olympus', 'panasonic', 'leica', 'hasselblad', 
            'samsung', 'iphone', 'pixel', 'adobe', 'aperture', 'paintshop', 
            'corel', 'indesign', 'illustrator', 'snapseed', 'instagram', 'vsco'
        ]
        
        # Known AI generator software tags with confidence weights
        ai_software_keywords = {
            'stable diffusion': 0.9, 'dall-e': 0.9, 'midjourney': 0.9, 
            'generative ai': 0.8, 'neural network': 0.5, 'gan': 0.7, 
            'diffusion': 0.7, 'ai generated': 0.9, 'dalle': 0.9, 
            'openai': 0.6, 'gpt': 0.7, 'nvidia stylegan': 0.8, 
            'deep learning': 0.5, 'latent diffusion': 0.8,
            'dreambooth': 0.8, 'runway': 0.8, 'synthesized': 0.6,
            'ml-generated': 0.8, 'ai-created': 0.9, 'ai-synthesized': 0.9
        }
        
        # Helper function to check software strings
        def check_software_string(value, location_name):
            nonlocal metadata_score, total_weight, metadata_findings
            
            if not isinstance(value, str):
                return
                
            value_lower = value.lower()
            
            # Check for AI generator indicators
            for keyword, weight in ai_software_keywords.items():
                if keyword in value_lower:
                    metadata_score += weight
                    total_weight += weight
                    metadata_findings[f"ai_keyword_in_{location_name}"] = f"{keyword} ({weight})"
            
            # Check for legitimate software to reduce false positives
            for software in legitimate_software:
                if software in value_lower:
                    # Reduce score if legitimate software is found
                    metadata_score -= 0.3
                    total_weight += 0.3  # Still count this in weight
                    metadata_findings[f"legitimate_software_{location_name}"] = software
                    break
        
        try:
            # Check if path exists
            if not os.path.exists(image_path):
                return 0, {"error": "File not found"}
            
            # Open image with PIL to read metadata
            with Image.open(image_path) as img:
                # Check for common AI generator software in metadata
                exif_data = img.getexif() if hasattr(img, 'getexif') else {}
                
                # Simplified EXIF tag lookups with dictionary
                exif_tags = {tag_id: ExifTags.TAGS.get(tag_id, str(tag_id)) 
                            for tag_id in exif_data.keys()}
                
                # Check all exif fields for AI-related keywords
                for tag_id, value in exif_data.items():
                    tag_name = exif_tags[tag_id]
                    check_software_string(value, tag_name)
                
                # Intelligent handling of missing metadata
                # Only slightly suspicious if there's no EXIF data
                if not exif_data and len(img.info) < 3:  # Very minimal metadata
                    metadata_score += 0.1  # Much lower weight than before
                    total_weight += 0.1
                    metadata_findings["minimal_metadata"] = True
                
                # Check image info
                info = img.info
                for key, value in info.items():
                    check_software_string(value, f"info_{key}")
                
                # Look for specific Software tag using direct lookup
                if 'Software' in exif_data:
                    software = exif_data['Software']
                    metadata_findings["software"] = software
                    check_software_string(software, "software_tag")
                    # Apply higher weight for explicit software tag
                    for keyword in ai_software_keywords:
                        if keyword in software.lower():
                            metadata_score += 0.2  # Additional weight for software tag
                            total_weight += 0.2
                
                # Check PNG specific metadata
                if img.format == 'PNG':
                    # Check PNG text chunks which might contain generator info
                    if hasattr(img, 'text') and img.text:
                        for key, value in img.text.items():
                            check_software_string(value, f"png_text_{key}")
        
        except Exception as e:
            warnings.warn(f"Error analyzing metadata: {str(e)}")
            metadata_findings["error"] = str(e)
        
        # Normalize score by total weight for consistency
        final_score = metadata_score / max(total_weight, 1.0) if total_weight > 0 else 0
        
        # Cap the score
        final_score = max(0.0, min(final_score, 1.0))
        
        return final_score, metadata_findings

    @staticmethod
    def check_perceptual_hash(image, threshold=0.85):
        """
        Check image perceptual hash for patterns common in AI-generated images
        with improved false positive handling.
        
        Improvements:
        - Reduced to 2 hash types (phash + dhash) for efficiency
        - Simplified analysis of hash regularity patterns
        - Better distinction between natural and AI symmetry
        - Simplified entropy calculation using bit density
        """
        # Convert to PIL Image for hashing
        if isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            image_pil = Image.fromarray((image * 255).astype(np.uint8))
        else:
            try:
                image_pil = Image.open(image)
            except:
                # If image is already a PIL Image or path doesn't work
                return np.zeros((224, 224)), 0
        
        try:
            # Calculate only two types of perceptual hashes for efficiency
            phash = imagehash.phash(image_pil)
            dhash = imagehash.dhash(image_pil)
            
            # Convert to binary strings
            phash_bin = bin(int(str(phash), 16))[2:].zfill(64)
            dhash_bin = bin(int(str(dhash), 16))[2:].zfill(64)
            
            # Check for patterns common in AI images
            
            # 1. Advanced regularity check - look for unnatural patterns
            # Analyze bit-run lengths (sequences of consecutive 0s or 1s)
            def analyze_runs(hash_bin):
                runs = []
                current_run = 1
                for i in range(1, len(hash_bin)):
                    if hash_bin[i] == hash_bin[i-1]:
                        current_run += 1
                    else:
                        runs.append(current_run)
                        current_run = 1
                runs.append(current_run)
                return runs
            
            phash_runs = analyze_runs(phash_bin)
            dhash_runs = analyze_runs(dhash_bin)
            
            # Calculate run statistics
            phash_max_run = max(phash_runs) if phash_runs else 0
            dhash_max_run = max(dhash_runs) if dhash_runs else 0
            
            # Analyze regularity (many AI images have unnaturally regular patterns)
            phash_regularity = sum(phash_bin[i] == phash_bin[i+1] for i in range(len(phash_bin)-1)) / (len(phash_bin)-1)
            dhash_regularity = sum(dhash_bin[i] == dhash_bin[i+1] for i in range(len(dhash_bin)-1)) / (len(dhash_bin)-1)
            
            regularity_score = (phash_regularity + dhash_regularity) / 2
            
            # If max run is very long, this could be a natural uniform area
            if phash_max_run > 12 or dhash_max_run > 12:
                regularity_score *= 0.7  # Reduce impact
            
            # 2. Detect unnatural symmetry (more common in AI images)
            phash_symmetry = sum(phash_bin[i] == phash_bin[len(phash_bin)-i-1] for i in range(len(phash_bin)//2)) / (len(phash_bin)//2)
            dhash_symmetry = sum(dhash_bin[i] == dhash_bin[len(dhash_bin)-i-1] for i in range(len(dhash_bin)//2)) / (len(dhash_bin)//2)
            
            symmetry_score = (phash_symmetry + dhash_symmetry) / 2
            
            # 3. Cross-hash consistency check
            # AI images often have more consistent relationships between different hash types
            cross_similarity_pd = sum(phash_bin[i] == dhash_bin[i] for i in range(len(phash_bin))) / len(phash_bin)
            
            # Natural images usually have lower cross-hash similarity
            # Adjust the cross similarity score by how far it deviates from natural range
            natural_cross_sim_range = (0.4, 0.6)  # Typical range for natural images
            
            # Calculate how unnaturally consistent the hashes are
            if cross_similarity_pd > natural_cross_sim_range[1]:
                cross_sim_score = (cross_similarity_pd - natural_cross_sim_range[1]) / (1 - natural_cross_sim_range[1])
            elif cross_similarity_pd < natural_cross_sim_range[0]:
                cross_sim_score = (natural_cross_sim_range[0] - cross_similarity_pd) / natural_cross_sim_range[0]
            else:
                cross_sim_score = 0  # Within natural range
                
            # 4. Simplified entropy calculation using bit density
            phash_density = phash_bin.count('1') / len(phash_bin)
            dhash_density = dhash_bin.count('1') / len(dhash_bin)
            
            # Natural images usually have density close to 0.5
            entropy_score = abs(phash_density - 0.5) + abs(dhash_density - 0.5)
            entropy_score = min(entropy_score, 1.0)
            
            # Combine metrics with weighted importance
            hash_ai_score = (
                regularity_score * 0.35 +
                symmetry_score * 0.35 +
                cross_sim_score * 0.2 +
                entropy_score * 0.1
            )
            
            # Apply threshold with hysteresis to reduce false positives
            if hash_ai_score < threshold * 0.8:
                hash_ai_score *= 0.7  # Reduce lower scores further
            
            # Cap at 1.0
            hash_ai_score = min(hash_ai_score, 1.0)
            
            # Return simplified hash metrics for analysis
            hash_details = {
                "phash_regularity": phash_regularity,
                "dhash_regularity": dhash_regularity,
                "symmetry_score": symmetry_score,
                "cross_hash_similarity": cross_similarity_pd,
                "bit_density_score": entropy_score,
                "hash_ai_score": hash_ai_score
            }
            
            # Create a feature map based on the hash analysis
            h, w = 224, 224  # Default size
            if isinstance(image, np.ndarray):
                h, w = image.shape[:2]
            
            # Create a feature map
            feature_map = np.zeros((h, w))
            
            # If hash analysis indicates AI generation, create gradient highlighting
            if hash_ai_score > threshold:
                # Create a radial gradient from center
                center_y, center_x = h // 2, w // 2
                Y, X = np.ogrid[:h, :w]
                dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
                max_dist = np.sqrt(center_x**2 + center_y**2)
                
                # Normalize distances to create radial gradient
                normalized_dist = dist_from_center / max_dist
                
                # Scale by confidence
                confidence_factor = (hash_ai_score - threshold) / (1 - threshold)
                feature_map = np.maximum(0, (1 - normalized_dist) * confidence_factor)
            
            return feature_map, hash_ai_score
            
        except Exception as e:
            warnings.warn(f"Error in perceptual hash analysis: {str(e)}")
            return np.zeros((224, 224)), 0

    # No need to implement process_large_image as it's already defined in utils.py

    @staticmethod
    def extract_all_features(image, config=None, image_path=None):
        """Extract all AI detection features and combine them using dynamic weighting"""
        # Use default config if none provided
        if config is None:
            config = AIDetectionConfig()
        
        # Set dynamic resolution based on image size
        if hasattr(config, 'set_dynamic_resolution'):
            config.set_dynamic_resolution(image.shape[:2])
        
        # Check if image is large and should use pyramid processing
        from utils import process_large_image
        if max(image.shape[:2]) > 1024:
            return process_large_image(image, config, image_path)
        
        # Extract features with config parameters
        gradient_map, gradient_score = StructuralFeaturesExtractor.extract_gradient_perfection(
            image, 
            threshold=config.gradient_threshold,
            patch_size=config.default_patch_size
        )
        
        pattern_map, pattern_score = StructuralFeaturesExtractor.detect_unnatural_patterns(
            image,
            threshold=config.pattern_threshold
        )
        
        edge_map, edge_score = StructuralFeaturesExtractor.analyze_edge_consistency(
            image,
            threshold=config.edge_threshold,
            patch_size=config.default_patch_size
        )
        
        symmetry_map, symmetry_score = StructuralFeaturesExtractor.detect_symmetry(
            image,
            threshold=config.symmetry_threshold
        )
        
        noise_map, noise_score = TextureFeaturesExtractor.detect_noise_patterns(
            image,
            patch_size=config.default_patch_size
        )
        
        texture_map, texture_score = TextureFeaturesExtractor.analyze_texture_quality(
            image,
            patch_size=config.default_patch_size
        )
        
        color_map, color_score = TextureFeaturesExtractor.analyze_color_coherence(
            image,
            patch_size=config.default_patch_size
        )
        
        hash_map, hash_score = AIFeatureExtractor.check_perceptual_hash(
            image,
            threshold=config.hash_threshold
        )

        # Metadata analysis if path is provided
        metadata_score = 0
        metadata_details = {}
        if image_path and os.path.exists(image_path):
            metadata_score, metadata_details = AIFeatureExtractor.analyze_metadata(image_path)

        # Calculate dynamic weights based on feature scores
        feature_scores = {
            'gradient': gradient_score,
            'pattern': pattern_score,
            'edge': edge_score,
            'symmetry': symmetry_score,
            'noise': noise_score,
            'texture': texture_score,
            'color': color_score,
            'hash': hash_score
        }
        
        # Calculate confidence in each feature
        total_score = sum(feature_scores.values())
        dynamic_weights = {}
        
        if total_score > 0:
            # Higher scores get proportionally higher weights
            for feature, score in feature_scores.items():
                # Base weight from config + dynamic component based on score confidence
                base_weight = config.feature_weights.get(feature, 1.0)
                dynamic_weight = base_weight * (1.0 + (score / max(total_score, 1.0)))
                dynamic_weights[feature] = dynamic_weight
        else:
            # Fallback to config weights if all scores are 0
            dynamic_weights = config.feature_weights
        
        # Normalize weights to sum to 1.0
        weight_sum = sum(dynamic_weights.values())
        if weight_sum > 0:
            for feature in dynamic_weights:
                dynamic_weights[feature] /= weight_sum
        
        # Combine feature maps using dynamic weights
        feature_stack = np.zeros((*image.shape[:2], len(config.feature_indices)))
        for feature_name, idx in config.feature_indices.items():
            feature_map = locals()[f"{feature_name}_map"]
            # Use dynamic weight instead of fixed config weight
            feature_stack[:, :, idx] = feature_map * dynamic_weights.get(feature_name, 1.0)

        # Create combined map using weighted sum
        combined_map = np.sum(feature_stack, axis=-1)
        
        return combined_map, feature_stack, {
            'gradient_score': gradient_score,
            'pattern_score': pattern_score,
            'noise_score': noise_score,
            'edge_score': edge_score,
            'symmetry_score': symmetry_score,
            'texture_score': texture_score,
            'color_score': color_score,
            'hash_score': hash_score,
            'metadata_score': metadata_score,
            'metadata_details': metadata_details,
            'dynamic_weights': dynamic_weights,
            'config': config.__dict__  # Return config settings for traceability
        }

    @staticmethod
    def generate_feature_maps(images, patch_mask=None, image_paths=None):
        """Generate feature maps for a batch of images with GPU acceleration support"""
        # Prepare arrays to store results
        batch_size = len(images)
        if batch_size == 0:
            return np.array([]), []
        
        # Get reference shape from first image
        first_img = images[0]
        height, width = first_img.shape[:2]
        n_features = len(AIDetectionConfig().feature_indices)
        
        # Pre-allocate arrays for better performance
        # This helps with both CPU and potential GPU operations
        all_feature_maps = np.zeros((batch_size, height, width, n_features), dtype=np.float32)
        all_scores = []
        
        # Process in batch where possible
        # For now we'll process images sequentially, but with code structure ready for GPU batching
        for i, img in enumerate(images):
            # Get path for metadata analysis if available
            img_path = None
            if image_paths is not None and i < len(image_paths):
                img_path = image_paths[i]
            
            # Extract features
            combined_map, feature_stack, scores = AIFeatureExtractor.extract_all_features(img, image_path=img_path)
            all_scores.append(scores)
            
            # Store feature maps
            all_feature_maps[i] = feature_stack
        
        # Apply patch masking in batch if provided
        if patch_mask is not None:
            # Convert patch mask to pixel mask if needed
            if len(patch_mask.shape) == 2:
                pixel_mask = convert_patch_mask_to_pixel_mask(
                    patch_mask,
                    image_shape=(height, width)
                )
            else:
                # Use directly if already pixel mask
                pixel_mask = patch_mask
            
            # Apply mask to all feature maps in the entire batch at once
            # This is much more efficient than iterating through each image
            for i in range(batch_size):
                for c in range(n_features):
                    all_feature_maps[i, :, :, c] *= pixel_mask
        
        return all_feature_maps, all_scores