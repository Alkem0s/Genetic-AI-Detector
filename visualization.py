# visualization.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import tensorflow as tf
import seaborn as sns
import logging
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import os

import global_config as config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Visualizer')

class Visualizer:
    """
    Visualizer class for AI image detection system.
    Provides visualization tools for feature maps, genetic masks,
    prediction results, and model performance evaluation.
    """

    def __init__(self):
        """
        Initialize the visualizer.
        """
        if config.output_dir and not os.path.exists(config.output_dir):
            os.makedirs(config.output_dir)
            logger.info(f"Created output directory: {config.output_dir}")

    def _save_figure(self, fig, filename):
        """Helper method to save figures if output directory is set"""
        if config.output_dir:
            path = os.path.join(config.output_dir, filename)
            fig.savefig(path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved figure to {path}")

    def plot_feature_maps(self, feature_maps, titles=None, max_cols=4, figsize=(14, 10)):
        """
        Plot feature maps extracted from an image.

        Args:
            feature_maps (np.ndarray): Feature maps of shape (height, width, n_features)
            titles (list): Optional list of titles for each feature map
            max_cols (int): Maximum number of columns in the grid
            figsize (tuple): Figure size

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        n_features = feature_maps.shape[2]

        if titles is None:
            titles = [f"Feature {i+1}" for i in range(n_features)]

        # Calculate grid dimensions
        n_cols = min(n_features, max_cols)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        # Flatten axes array for easier indexing
        if n_rows > 1 or n_cols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]

        # Plot each feature map
        for i in range(n_features):
            ax = axes[i]
            feature = feature_maps[:, :, i]

            # Normalize for better visualization
            vmin, vmax = np.min(feature), np.max(feature)

            im = ax.imshow(feature, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_title(titles[i])
            ax.axis('off')

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Turn off any unused subplots
        for i in range(n_features, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        self._save_figure(fig, "feature_maps.png")

        return fig

    def plot_genetic_mask(self, image, mask, figsize=(10, 5)):
        """
        Visualize the genetic algorithm generated mask overlaid on the image.

        Args:
            image (np.ndarray): Original image
            mask (np.ndarray): Binary mask of the same size as image
            figsize (tuple): Figure size

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        # Mask only
        axes[1].imshow(mask, cmap='viridis')
        axes[1].set_title("Genetic Mask")
        axes[1].axis('off')

        # Masked image (overlay)
        # Create an RGB mask for overlay with transparency
        h, w = mask.shape[:2]
        mask_rgb = np.zeros((h, w, 4), dtype=np.float32)
        mask_rgb[..., 0] = 1.0  # Red channel
        mask_rgb[..., 3] = mask * 0.5  # Alpha channel (50% transparency)

        axes[2].imshow(image)
        axes[2].imshow(mask_rgb)
        axes[2].set_title("Overlay")
        axes[2].axis('off')

        plt.tight_layout()
        self._save_figure(fig, "genetic_mask_overlay.png")

        return fig

    def plot_patch_mask(self, image, patch_mask, figsize=(10, 5)):
        """
        Visualize a patch-level mask.

        Args:
            image (np.ndarray): Original image
            patch_mask (np.ndarray): Binary patch mask
            figsize (tuple): Figure size

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        patch_size = config.patch_size

        h, w = image.shape[:2]
        n_patches_h, n_patches_w = patch_mask.shape

        pixel_mask = np.zeros((h, w), dtype=np.float32)

        for i in range(n_patches_h):
            for j in range(n_patches_w):
                if patch_mask[i, j] == 1:
                    y1 = i * patch_size
                    y2 = min((i + 1) * patch_size, h)
                    x1 = j * patch_size
                    x2 = min((j + 1) * patch_size, w)
                    pixel_mask[y1:y2, x1:x2] = 1.0

        # Create figure
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(1, 3, figure=fig)

        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image)
        ax1.set_title("Original Image")
        ax1.axis('off')

        # Patch mask (grid visualization)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(patch_mask, cmap='viridis')
        ax2.set_title("Patch Mask")
        ax2.axis('off')

        # Add grid lines to show patches
        for i in range(n_patches_w + 1):
            ax2.axvline(i - 0.5, color='gray', linewidth=0.5)
        for i in range(n_patches_h + 1):
            ax2.axhline(i - 0.5, color='gray', linewidth=0.5)

        # Overlay on original image
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(image)

        # Create a semi-transparent overlay
        mask_rgb = np.zeros((h, w, 4), dtype=np.float32)
        mask_rgb[..., 0] = 1.0  # Red channel
        mask_rgb[..., 3] = pixel_mask * 0.6  # Alpha channel (60% transparency)

        ax3.imshow(mask_rgb)
        ax3.set_title("Overlay")
        ax3.axis('off')

        plt.tight_layout()
        self._save_figure(fig, "patch_mask_visualization.png")

        return fig

    def plot_training_history(self, history, figsize=(12, 5)):
        """
        Plot training history metrics.

        Args:
            history: Keras history object or dictionary with history data
            figsize (tuple): Figure size

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Convert history to dict if it's a Keras History object
        if hasattr(history, 'history'):
            history = history.history

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Plot accuracy
        axes[0].plot(history['accuracy'], label='Training')
        if 'val_accuracy' in history:
            axes[0].plot(history['val_accuracy'], label='Validation')
        axes[0].set_title('Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.7)

        # Plot loss
        axes[1].plot(history['loss'], label='Training')
        if 'val_loss' in history:
            axes[1].plot(history['val_loss'], label='Validation')
        axes[1].set_title('Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        self._save_figure(fig, "training_history.png")

        return fig

    def plot_genetic_algorithm_progress(self, history, figsize=(12, 5)):
        """
        Plot genetic algorithm optimization progress.

        Args:
            history (list): List of dictionaries with 'generation', 'max_fitness', 'avg_fitness'
            figsize (tuple): Figure size

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        generations = [h['generation'] for h in history]
        max_fitness = [h['max_fitness'] for h in history]
        avg_fitness = [h['avg_fitness'] for h in history]

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(generations, max_fitness, 'b-', label='Best Fitness')
        ax.plot(generations, avg_fitness, 'r--', label='Average Fitness')

        ax.set_title('Genetic Algorithm Optimization Progress')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness Score')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

        # Add annotations for initial and final fitness
        if len(generations) > 0:
            # Initial fitness
            ax.annotate(f"Initial: {max_fitness[0]:.4f}",
                       xy=(generations[0], max_fitness[0]),
                       xytext=(generations[0] + 1, max_fitness[0] - 0.05),
                       arrowprops=dict(arrowstyle="->", lw=1.5))

            # Final fitness
            ax.annotate(f"Final: {max_fitness[-1]:.4f}",
                       xy=(generations[-1], max_fitness[-1]),
                       xytext=(generations[-1] - 5, max_fitness[-1] + 0.05),
                       arrowprops=dict(arrowstyle="->", lw=1.5))

        plt.tight_layout()
        self._save_figure(fig, "genetic_algorithm_progress.png")

        return fig

    def plot_confusion_matrix(self, y_true, y_pred, figsize=(8, 6)):
        """
        Plot confusion matrix for binary classification.

        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            figsize (tuple): Figure size

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot using seaborn for better styling
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)

        # Add percentage annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text = ax.text(j + 0.5, i + 0.7, f"({cm_norm[i, j]:.1%})",
                              ha="center", va="center", color="black", fontsize=9)

        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        ax.set_xticklabels(['Human', 'AI'])
        ax.set_yticklabels(['Human', 'AI'])

        plt.tight_layout()
        self._save_figure(fig, "confusion_matrix.png")

        return fig

    def plot_roc_curve(self, y_true, y_score, figsize=(8, 6)):
        """
        Plot ROC curve for binary classification.

        Args:
            y_true (np.ndarray): True labels
            y_score (np.ndarray): Prediction scores (probabilities)
            figsize (tuple): Figure size

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot ROC curve
        ax.plot(fpr, tpr, 'b-', label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Random')

        # Add optimal threshold point (closest to top-left corner)
        optimal_idx = np.argmin(np.sqrt((1-tpr)**2 + fpr**2))
        optimal_threshold = thresholds[optimal_idx]

        ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro',
                label=f'Optimal threshold: {optimal_threshold:.3f}')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        self._save_figure(fig, "roc_curve.png")

        return fig

    def plot_precision_recall_curve(self, y_true, y_score, figsize=(8, 6)):
        """
        Plot precision-recall curve for binary classification.

        Args:
            y_true (np.ndarray): True labels
            y_score (np.ndarray): Prediction scores (probabilities)
            figsize (tuple): Figure size

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot precision-recall curve
        ax.plot(recall, precision, 'g-', label=f'Precision-Recall curve (AUC = {pr_auc:.3f})')

        # Add F1-optimal threshold point
        f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]

        ax.plot(recall[optimal_idx], precision[optimal_idx], 'ro',
                label=f'F1-optimal threshold: {optimal_threshold:.3f}')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="lower left")
        ax.grid(True, linestyle='--', alpha=0.7)

        # Add average precision annotation
        ax.text(0.5, 0.2, f'F1 Score (optimal): {f1_scores[optimal_idx]:.3f}',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        plt.tight_layout()
        self._save_figure(fig, "precision_recall_curve.png")

        return fig

    def plot_feature_importance(self, feature_importance, figsize=(10, 6)):
        """
        Plot feature importance from genetic algorithm.

        Args:
            feature_importance (dict): Dictionary of feature importance scores
            figsize (tuple): Figure size

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Sort features by importance
        features = list(feature_importance.keys())
        scores = list(feature_importance.values())

        sorted_indices = np.argsort(scores)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot horizontal bar chart
        y_pos = np.arange(len(features))
        ax.barh(y_pos, [scores[i] for i in sorted_indices], align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([features[i] for i in sorted_indices])

        # Add value annotations
        for i, v in enumerate([scores[j] for j in sorted_indices]):
            ax.text(v + 0.01, i, f"{v:.3f}", va='center')

        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance from Genetic Algorithm')
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)

        plt.tight_layout()
        self._save_figure(fig, "feature_importance.png")

        return fig

    def visualize_prediction(self, image, prediction_result, mask=None, patch_mask=None, figsize=(14, 6)):
        """
        Comprehensive visualization of a model prediction.

        Args:
            image (np.ndarray): Input image
            prediction_result (dict): Prediction result with keys:
                - is_ai (bool): True if AI-generated
                - label (str): Human/AI label
                - confidence (float): Confidence score
                - raw_prediction (float): Raw model output
            mask (np.ndarray, optional): Pixel-level mask if available
            patch_mask (np.ndarray, optional): Patch-level mask if available
            figsize (tuple): Figure size

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        patch_size = config.patch_size

        # Determine number of subplots needed
        has_mask = mask is not None
        has_patch_mask = patch_mask is not None and patch_size is not None

        n_plots = 1 + has_mask + has_patch_mask

        # Create figure
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(1, n_plots + 1, figure=fig, width_ratios=[2]*n_plots + [1])

        # Original image with prediction label
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image)
        ax1.set_title("Input Image")
        ax1.axis('off')

        plot_idx = 1

        # Pixel mask overlay if available
        if has_mask:
            ax2 = fig.add_subplot(gs[0, plot_idx])
            ax2.imshow(image)

            # Create a semi-transparent overlay
            h, w = mask.shape[:2]
            mask_rgb = np.zeros((h, w, 4), dtype=np.float32)
            mask_rgb[..., 0] = 1.0  # Red channel
            mask_rgb[..., 3] = mask * 0.6  # Alpha channel

            ax2.imshow(mask_rgb)
            ax2.set_title("Pixel Mask Overlay")
            ax2.axis('off')

            plot_idx += 1

        # Patch mask overlay if available
        if has_patch_mask:
            ax3 = fig.add_subplot(gs[0, plot_idx])

            # Convert patch mask to pixel mask
            h, w = image.shape[:2]
            n_patches_h, n_patches_w = patch_mask.shape

            pixel_mask = np.zeros((h, w), dtype=np.float32)

            for i in range(n_patches_h):
                for j in range(n_patches_w):
                    if patch_mask[i, j] == 1:
                        y1 = i * patch_size
                        y2 = min((i + 1) * patch_size, h)
                        x1 = j * patch_size
                        x2 = min((j + 1) * patch_size, w)
                        pixel_mask[y1:y2, x1:x2] = 1.0

            ax3.imshow(image)

            # Create a semi-transparent overlay
            mask_rgb = np.zeros((h, w, 4), dtype=np.float32)
            mask_rgb[..., 0] = 1.0  # Red channel for AI
            mask_rgb[..., 3] = pixel_mask * 0.6  # Alpha channel

            ax3.imshow(mask_rgb)

            # Draw patch grid
            for i in range(n_patches_w + 1):
                x = i * patch_size
                ax3.axvline(x, color='white', linewidth=0.5, alpha=0.5)
            for i in range(n_patches_h + 1):
                y = i * patch_size
                ax3.axhline(y, color='white', linewidth=0.5, alpha=0.5)

            ax3.set_title("Patch Mask Overlay")
            ax3.axis('off')

            plot_idx += 1

        # Prediction details in text box
        ax_text = fig.add_subplot(gs[0, -1])
        ax_text.axis('off')

        # Create text box with prediction details
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        textstr = f"Prediction: {prediction_result['label']}\n"
        textstr += f"Confidence: {prediction_result['confidence']:.2%}\n"
        textstr += f"Raw Score: {prediction_result['raw_prediction']:.4f}"

        # Add color coding
        if prediction_result['is_ai']:
            box_color = 'lightcoral'
            text_color = 'darkred'
        else:
            box_color = 'lightgreen'
            text_color = 'darkgreen'

        props['facecolor'] = box_color

        # Place text box
        ax_text.text(0.05, 0.5, textstr, transform=ax_text.transAxes, fontsize=12,
                    verticalalignment='center', bbox=props, color=text_color)

        plt.tight_layout()
        self._save_figure(fig, "prediction_visualization.png")

        return fig

    def plot_batch_predictions(self, images, predictions, labels=None, figsize=(15, 10), max_images=16):
        """
        Plot predictions for a batch of images with optional ground truth labels.

        Args:
            images (np.ndarray): Batch of images
            predictions (list): List of prediction dictionaries
            labels (np.ndarray, optional): Ground truth labels (0=Human, 1=AI)
            figsize (tuple): Figure size
            max_images (int): Maximum number of images to show

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Limit number of images to display
        n_images = min(len(images), max_images)

        # Calculate grid dimensions
        n_cols = min(4, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols

        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        # Flatten axes array for easier indexing
        if n_rows > 1 or n_cols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]

        # Plot each image with prediction
        for i in range(n_images):
            ax = axes[i]
            ax.imshow(images[i])

            # Prediction info
            pred = predictions[i]
            pred_label = "AI" if pred['is_ai'] else "Human"
            confidence = pred['confidence']

            title = f"Pred: {pred_label} ({confidence:.1%})"

            # Add ground truth if available
            if labels is not None:
                true_label = "AI" if labels[i] == 1 else "Human"
                title = f"True: {true_label} | {title}"

                # Color based on correct/incorrect
                is_correct = (labels[i] == 1) == pred['is_ai']
                color = 'green' if is_correct else 'red'
                ax.spines['bottom'].set_color(color)
                ax.spines['top'].set_color(color)
                ax.spines['right'].set_color(color)
                ax.spines['left'].set_color(color)
                ax.spines['bottom'].set_linewidth(5)
                ax.spines['top'].set_linewidth(5)
                ax.spines['right'].set_linewidth(5)
                ax.spines['left'].set_linewidth(5)

            ax.set_title(title)
            ax.axis('off')

        # Turn off any unused subplots
        for i in range(n_images, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        self._save_figure(fig, "batch_predictions.png")

        return fig