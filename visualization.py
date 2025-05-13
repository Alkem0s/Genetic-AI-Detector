import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore


class Visualizer:
    """Class for visualizing AI detection features and model results"""
    
    @staticmethod
    def plot_training_history(history, save_path='training_history.png'):
        """Plot and save the training history metrics
        
        Args:
            history: Keras history object from model training
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 5))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"Training history plot saved to {save_path}")
    
    @staticmethod
    def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):
        """Plot and save confusion matrix
        
        Args:
            cm: Confusion matrix from sklearn
            save_path: Path to save the plot
        """
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Set labels
        classes = ['Human', 'AI']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(save_path)
        plt.close()
        
        print(f"Confusion matrix plot saved to {save_path}")
    
    @staticmethod
    def plot_genetic_optimization_stats(stats, save_path='genetic_optimization.png'):
        """Plot genetic algorithm optimization statistics
        
        Args:
            stats: Statistics dictionary from genetic algorithm
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        # Plot fitness evolution
        generations = range(1, len(stats['avg']) + 1)
        
        plt.plot(generations, stats['avg'], label='Average Fitness', marker='o', linestyle='-')
        plt.plot(generations, stats['max'], label='Best Fitness', marker='s', linestyle='-')
        plt.plot(generations, stats['min'], label='Worst Fitness', marker='^', linestyle='-')
        
        plt.title('Genetic Algorithm Optimization Progress')
        plt.xlabel('Generation')
        plt.ylabel('Fitness Score')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"Genetic optimization stats plot saved to {save_path}")
    
    @staticmethod
    def visualize_optimized_patch_mask(patch_mask, save_path='optimized_patches.png'):
        """Visualize the optimized patch mask from genetic algorithm
        
        Args:
            patch_mask: 2D numpy array of optimized patch mask
            save_path: Path to save the visualization
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(patch_mask, cmap='viridis')
        plt.title('Optimized Feature Patch Selection')
        plt.colorbar(label='Selection Importance')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"Optimized patch mask visualization saved to {save_path}")
    
    @staticmethod
    def visualize_ai_features(image_path, feature_extractor, save_path='ai_feature_visualization.png'):
        """Visualize AI features detected in an image
        
        Args:
            image_path: Path to the image file
            feature_extractor: AIFeatureExtractor instance
            save_path: Path to save the visualization
        """
        # Load and preprocess image
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        
        # Extract AI features
        combined_map, feature_stack, scores = feature_extractor.extract_all_features(img_array)
        
        # Get feature indices from config
        config = feature_extractor.config
        feature_indices = config.feature_indices
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(img_array)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Gradient perfection map
        gradient_idx = feature_indices['gradient']
        axes[0, 1].imshow(feature_stack[:,:,gradient_idx], cmap='hot')
        axes[0, 1].set_title('Perfect Gradient Detection')
        axes[0, 1].axis('off')
        
        # Unnatural patterns map
        pattern_idx = feature_indices['pattern']
        axes[0, 2].imshow(feature_stack[:,:,pattern_idx], cmap='hot')
        axes[0, 2].set_title('Unnatural Pattern Detection')
        axes[0, 2].axis('off')
        
        # Noise pattern map
        noise_idx = feature_indices['noise']
        axes[1, 0].imshow(feature_stack[:,:,noise_idx], cmap='hot')
        axes[1, 0].set_title('Noise Pattern Analysis')
        axes[1, 0].axis('off')
        
        # Combined feature map
        axes[1, 1].imshow(combined_map, cmap='hot')
        axes[1, 1].set_title('Combined AI Features')
        axes[1, 1].axis('off')
        
        # Overlay on original
        overlay = img_array.copy()
        # Add a red tint to areas identified as AI features
        red_mask = np.zeros_like(overlay)
        red_mask[:,:,0] = combined_map  # Red channel
        overlay = np.clip(overlay + 0.5 * red_mask, 0, 1)
        
        axes[1, 2].imshow(overlay)
        axes[1, 2].set_title('AI Features Overlay')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"AI feature visualization saved to {save_path}")
        
        return combined_map, feature_stack, scores
    
    @staticmethod
    def visualize_prediction_results(image_path, prediction, confidence, feature_maps=None, 
                                    save_path='prediction_visualization.png'):
        """Visualize prediction results for a single image
        
        Args:
            image_path: Path to the image file
            prediction: Model prediction (0=Human, 1=AI)
            confidence: Prediction confidence (0-1)
            feature_maps: Feature maps used for prediction (optional)
            save_path: Path to save the visualization
        """
        # Load image
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        
        # Create figure
        if feature_maps is not None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes = [axes[0], axes[1], None]  # Create placeholder for the third axis
        
        # Display original image
        axes[0].imshow(img_array)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Display prediction results with color-coded confidence
        result_text = "AI-generated" if prediction > 0.5 else "Human-generated"
        conf_pct = confidence * 100
        
        # Create a colored box based on prediction and confidence
        if prediction > 0.5:  # AI prediction
            color = (1 - confidence, 0, confidence)  # More red for higher confidence
        else:  # Human prediction
            color = (0, confidence, 1 - confidence)  # More blue for higher confidence
        
        # Create a simple visualization of the result
        result_img = np.ones((224, 224, 3))
        result_img = result_img * np.array(color).reshape(1, 1, 3)
        
        # Add text to the prediction visualization
        axes[1].imshow(result_img)
        axes[1].text(112, 100, f"{result_text}\nConfidence: {conf_pct:.1f}%", 
                    horizontalalignment='center',
                    color='white', fontsize=16, fontweight='bold')
        axes[1].set_title('Prediction Result')
        axes[1].axis('off')
        
        # If feature maps are provided, show them
        if feature_maps is not None and axes[2] is not None:
            # Show combined feature map (assuming the shape is correct)
            if len(feature_maps.shape) == 4:  # Batch dimension
                feature_map = np.mean(feature_maps[0], axis=-1)  # Average across channels
            else:
                feature_map = np.mean(feature_maps, axis=-1)
                
            axes[2].imshow(feature_map, cmap='hot')
            axes[2].set_title('Feature Map')
            axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"Prediction visualization saved to {save_path}")