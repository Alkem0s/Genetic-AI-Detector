# main.py
import os
import json
import pickle
import logging
import numpy as np
import tensorflow as tf

from data_loader import DataLoader
from genetic_algorithm import GeneticFeatureOptimizer
from model_architecture import ModelWrapper
from visualization import Visualizer

import global_config as config

logger = logging.getLogger('ai_detector')
logger.setLevel(logging.INFO)

# Check if handlers already exist to prevent duplicate output
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.propagate = False

    logger.info("Logging configured.")


def setup_environment():
    """Setup environment variables and configurations"""
    # Set environment variables
    os.environ['TF_CUDNN_USE_AUTOTUNE'] = '1'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

    # Create output directory if it doesn't exist
    os.makedirs(config.output_dir, exist_ok=True)

    # Create feature cache directory
    os.makedirs(os.path.join(config.output_dir, config.feature_cache_dir), exist_ok=True)

    # Setup GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU(s) available: {len(gpus)}")
        except RuntimeError as e:
            logger.error(f"Error setting up GPU: {e}")
    else:
        logger.info("No GPUs found, using CPU")

    # Set random seeds for reproducibility
    np.random.seed(config.random_seed)
    tf.random.set_seed(config.random_seed)

    # Enable mixed precision if specified
    if config.mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info("Mixed precision enabled (float16)")

    # Prepare base path for saving/loading complete model state
    model_base_path = os.path.join(config.output_dir, config.model_path.split('.')[0])
    return model_base_path


def train_ai_detector(model_base_path):
    """Train an AI-generated content detector using genetic optimization and CNN"""
    logger.info("=== Step 1: Loading and preparing datasets ===")

    # Initialize the data loader
    data_loader = DataLoader()

    # Create the datasets
    train_ds, test_ds, sample_images, sample_labels = data_loader.create_datasets()
    logger.info(f"Created TensorFlow dataset pipeline for training and testing")

    best_rules = None
    ga_stats = None

    if config.use_feature_extraction:
        logger.info("=== Step 2: Running genetic algorithm for feature optimization ===")

        # Initialize the genetic feature optimizer
        genetic_optimizer = GeneticFeatureOptimizer(
            images=sample_images,
            labels=sample_labels,
        )

        # Run the genetic algorithm optimization
        best_ind, ga_stats = genetic_optimizer.run()
    else:
        logger.info("=== Step 2: Skipping genetic algorithm ===")

    logger.info("=== Step 3: Building and training the model ===")

    # Initialize model wrapper with optimized genetic rules (or None if not using features)
    model_wrapper = ModelWrapper(
        genetic_rules=best_ind.rules_tensor if config.use_feature_extraction else None,
    )

    # Train the model with the optimized features
    history = model_wrapper.train(
        train_dataset=train_ds,
        validation_dataset=test_ds,
        model_path=config.model_path
    )

    # Save the complete model state (NN model + genetic rules)
    model_wrapper.save_complete_model_state(model_base_path)
    logger.info(f"Saved complete model state (NN model and genetic rules) to {model_base_path}.*")

    logger.info("=== Step 4: Evaluating the model ===")
    model = model_wrapper.get_model()

    # Evaluate on the test dataset
    metrics, y_true, y_pred = evaluate_model(model, test_ds)

    # Print evaluation results
    logger.info(f"Test accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Test loss: {metrics['loss']:.4f}")

    # Visualize training results if requested
    if config.visualize: # Check visualize for general plots
        visualizer = Visualizer()
        visualizer.plot_training_history(history)

        if y_true is not None and y_pred is not None:
            visualizer.plot_confusion_matrix(y_true, y_pred)

    # Save metrics summary
    save_metrics(metrics, history)

    logger.info(f"Model training complete. Complete model state saved to {model_base_path}.*")
    return model_wrapper, best_rules, {"ga_stats": ga_stats, "training_history": history}


def evaluate_model(model, test_ds):
    """Evaluate model on test dataset and compute metrics"""
    # Evaluate the model
    results = model.evaluate(test_ds, verbose=1)

    # TensorFlow will return all metrics defined in the model
    if isinstance(results, list):
        metrics = {
            'loss': results[0],
            'accuracy': results[1]
        }
    else:
        metrics = {
            'loss': results,
            'accuracy': None
        }

    # Compute detailed metrics
    try:
        y_true = []
        y_pred = []

        # Collect predictions on batches
        for images, labels in test_ds:
            pred = model.predict(images, verbose=0)
            pred_classes = (pred > 0.5).astype(int).flatten()

            y_true.extend(labels.numpy())
            y_pred.extend(pred_classes)

        # Calculate metrics
        from sklearn.metrics import classification_report, confusion_matrix

        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=['Human', 'AI'], output_dict=True
        )

        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

        logger.info("\nClassification Report:")
        logger.info(classification_report(y_true, y_pred, target_names=['Human', 'AI']))

        logger.info("\nConfusion Matrix:")
        logger.info(metrics['confusion_matrix'])

    except Exception as e:
        logger.warning(f"Could not compute detailed metrics: {e}")
        y_true = None
        y_pred = None

    return metrics, y_true, y_pred


def save_metrics(metrics, history):
    """Save metrics and training history"""
    # Save training history
    if history and hasattr(history, 'history'):
        history_path = os.path.join(config.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            # Convert numpy values to Python native types
            history_dict = {}
            for key, values in history.history.items():
                history_dict[key] = [float(x) for x in values]
            json.dump(history_dict, f)

    # Save metrics summary
    metrics_path = os.path.join(config.output_dir, 'metrics_summary.json')
    with open(metrics_path, 'w') as f:
        # Create a JSON-serializable version of metrics
        metrics_json = {
            'accuracy': float(metrics['accuracy']) if metrics['accuracy'] is not None else None,
            'loss': float(metrics['loss']) if metrics['loss'] is not None else None
        }

        # Add classification report if available
        if 'classification_report' in metrics:
            metrics_json['classification_report'] = metrics['classification_report']

        json.dump(metrics_json, f)

    logger.info(f"Training history and metrics saved to {config.output_dir}")


def load_model_and_rules(model_base_path):
    """Load a trained model and genetic rules using the ModelWrapper's combined load method."""
    logger.info(f"Loading existing model and rules from {model_base_path}.*")

    # Initialize model wrapper
    model_wrapper = ModelWrapper()

    # Load the complete model state
    try:
        model_wrapper.load_complete_model_state(model_base_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find complete model state at {model_base_path}.* : {e}")

    # The genetic rules are now loaded within the model_wrapper
    best_rules = model_wrapper.genetic_rules
    logger.info("Complete model state loaded successfully.")

    return model_wrapper, best_rules


def predict_image(image_path, model_wrapper):
    """Make a prediction on a single image"""
    logger.info(f"Making prediction for image: {image_path}")

    # Load and preprocess the image using DataLoader's method
    data_loader = DataLoader()
    img, _ = data_loader.process_path(image_path, 0)  # Dummy label, not used
    img = tf.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction
    result = model_wrapper.predict_image(img)

    logger.info(f"Prediction: {result['label']} with {result['confidence']:.2%} confidence")

    # Visualize features if requested
    if config.visualize and config.use_feature_extraction:
        visualizer = Visualizer()
        visualizer.plot_features(model_wrapper.extract_features(img))

    return result


def main():
    """Main function to coordinate the workflow"""

    # Setup environment and get base path for model and rules
    model_base_path = setup_environment()

    # Print configuration mode
    if config.use_feature_extraction:
        logger.info("=== Running in FULL MODE (with genetic algorithm and feature extraction) ===")
    else:
        logger.info("=== Running in BASELINE MODE (without feature extraction) ===")

    model_wrapper = None
    best_rules = None

    # Load or train model
    # Check for the existence of the combined model state
    model_state_exists = os.path.exists(f"{model_base_path}_config.json")

    if config.skip_training and model_state_exists:
        model_wrapper, best_rules = load_model_and_rules(model_base_path)
    else:
        model_wrapper, best_rules, stats = train_ai_detector(model_base_path)

    # Make prediction if requested
    if config.predict:
        if not config.predict_path:
            logger.error("Prediction path is not specified in config. Cannot make prediction.")
        else:
            predict_image(config.predict_path, model_wrapper)

    logger.info("AI detector pipeline completed successfully")
    return model_wrapper, best_rules


if __name__ == "__main__":
    main()