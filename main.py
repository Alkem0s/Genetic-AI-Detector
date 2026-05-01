# main.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

logger = logging.getLogger('ai_detector_main')
logger.setLevel(logging.INFO)

def setup_environment():
    """Setup environment variables and configurations"""
    # Set environment variables
    os.environ['TF_CUDNN_USE_AUTOTUNE'] = '1'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

    # Create output directory if it doesn't exist
    os.makedirs(config.output_dir, exist_ok=True)

    # Create feature cache directory
    os.makedirs(os.path.join(config.output_dir, config.feature_cache_dir), exist_ok=True)

    # Configure root logger to capture all module logs
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler to logger
    log_file_path = os.path.join(config.output_dir, 'ai_detector.log')
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    logger.info(f"Console output will also be saved to {log_file_path}")

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

    # Prepare base path for saving/loading model state
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
            config=config,
        )

        # Run the genetic algorithm optimization
        results = genetic_optimizer.run()
    else:
        logger.info("=== Step 2: Skipping genetic algorithm ===")

    logger.info("=== Step 3: Building and training the model ===")

    # Initialize model wrapper with optimized genetic rules (or None if not using features)
    model_wrapper = ModelWrapper(
        genetic_rules=results['best_individual'].rules_tensor if config.use_feature_extraction else None,
    )

    # Train the model with the optimized features
    history = model_wrapper.train(
        train_dataset=train_ds,
        validation_dataset=test_ds,
        model_path=os.path.join(config.output_dir, config.model_path)
    )

    # Save the model state (NN model + genetic rules)
    model_wrapper.save(model_base_path)
    logger.info(f"Saved model state (NN model and genetic rules) to {model_base_path}.*")

    logger.info("=== Step 4: Evaluating the model ===")
    metrics, y_true, y_pred = evaluate_model(model_wrapper, test_ds)

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

    logger.info(f"Model training complete. Model state saved to {model_base_path}.*")
    return model_wrapper, best_rules, {"ga_stats": ga_stats, "training_history": history}


def evaluate_model(model_wrapper, test_ds):
    from metrics import calculate_precision_recall_f1, calculate_balanced_accuracy

    # Prepare the test dataset through the wrapper
    if model_wrapper.use_features:
        model_wrapper.precompute_features(test_ds, "test")
        prepared_ds = model_wrapper.prepare_dataset(test_ds, "test")
    else:
        prepared_ds = test_ds

    # Evaluate using the inner Keras model on the already-prepared dataset
    results = model_wrapper.model.model.evaluate(prepared_ds, verbose=1)
    metrics = {
        'loss': results[0],
        'accuracy': results[1] if len(results) > 1 else None
    }

    # Collect predictions
    try:
        y_true_list, y_pred_list = [], []
        for batch in prepared_ds:
            inputs, labels = batch
            pred = model_wrapper.model.model.predict_on_batch(inputs)
            pred_classes = (pred > 0.5).astype(int).flatten()
            y_true_list.extend(labels.numpy())
            y_pred_list.extend(pred_classes)

        y_true = tf.cast(y_true_list, tf.int64)
        y_pred = tf.cast(y_pred_list, tf.int64)

        # Calculate metrics using metrics.py functions
        precision, recall, f1 = calculate_precision_recall_f1(y_true, y_pred)
        balanced_accuracy = calculate_balanced_accuracy(y_true, y_pred)

        # Build confusion matrix components via TensorFlow
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.cast(y_pred, tf.float32)
        tp = int(tf.reduce_sum(y_true_f * y_pred_f).numpy())
        fp = int(tf.reduce_sum((1 - y_true_f) * y_pred_f).numpy())
        fn = int(tf.reduce_sum(y_true_f * (1 - y_pred_f)).numpy())
        tn = int(tf.reduce_sum((1 - y_true_f) * (1 - y_pred_f)).numpy())

        precision_val = float(precision.numpy())
        recall_val = float(recall.numpy())
        f1_val = float(f1.numpy())
        balanced_acc_val = float(balanced_accuracy.numpy())

        # Build a classification_report-compatible dict for downstream consumers
        human_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        human_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        human_f1 = (2 * human_precision * human_recall / (human_precision + human_recall)
                    if (human_precision + human_recall) > 0 else 0.0)
        total = tp + fp + fn + tn
        metrics['classification_report'] = {
            'Human': {
                'precision': human_precision,
                'recall': human_recall,
                'f1-score': human_f1,
                'support': tn + fp,
            },
            'AI': {
                'precision': precision_val,
                'recall': recall_val,
                'f1-score': f1_val,
                'support': tp + fn,
            },
            'balanced_accuracy': balanced_acc_val,
        }

        # Confusion matrix as a plain nested list [[TN, FP], [FN, TP]]
        metrics['confusion_matrix'] = [[tn, fp], [fn, tp]]

        metrics['precision'] = precision_val
        metrics['recall'] = recall_val
        metrics['f1'] = f1_val
        metrics['balanced_accuracy'] = balanced_acc_val

        logger.info("\nClassification Report:")
        logger.info(f"{'':>10} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
        for cls, name in [(metrics['classification_report']['Human'], 'Human'),
                          (metrics['classification_report']['AI'], 'AI')]:
            logger.info(
                f"{name:>10} {cls['precision']:>10.4f} {cls['recall']:>10.4f} "
                f"{cls['f1-score']:>10.4f} {cls['support']:>10}"
            )
        logger.info(f"\n{'balanced_accuracy':>10}: {balanced_acc_val:.4f}")

        logger.info("\nConfusion Matrix (rows=actual, cols=predicted):")
        logger.info(f"              Predicted Human  Predicted AI")
        logger.info(f"Actual Human  {tn:>14}  {fp:>12}")
        logger.info(f"Actual AI     {fn:>14}  {tp:>12}")

        y_true = y_true_list
        y_pred = y_pred_list

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

    try:
        model_wrapper = ModelWrapper.load(model_base_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find model state at {model_base_path}.* : {e}")

    # The genetic rules are now loaded within the model_wrapper
    best_rules = model_wrapper.genetic_rules
    logger.info("Model state loaded successfully.")

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