import os
import json
import pickle
import logging
import numpy as np
import tensorflow as tf
from types import SimpleNamespace

# Import our custom modules
from data_loader import DataLoader
from genetic_algorithm import GeneticFeatureOptimizer
from model_architecture import ModelWrapper # Corrected import for ModelWrapper
from visualization import Visualizer
from feature_extractor import FeatureExtractor

# Configure logging
logger = logging.getLogger('ai_detector')
logger.setLevel(logging.INFO)

# Check if handlers already exist to prevent duplicate output
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # *** IMPORTANT FIX: Prevent propagation to the root logger ***
    logger.propagate = False

    logger.info("Logging configured.")


def load_detector_config(config_path):
    """Load configuration from JSON file and merge with defaults"""
    # Default configuration
    defaults = {
        "output_dir": "output",
        "image_size": None,
        "batch_size": 32,
        "max_images": 10000,
        "epochs": 50,
        "test_size": 0.2,
        "random_seed": 42,
        "model_base_name": "ai_detector_model", # Changed from model_path
        "visualize": False,
        "predict": None,
        "skip_training": False,
        "use_feature_extraction": True,
        "mixed_precision": True,
        "feature_cache_dir": "feature_cache"
    }
    required = ["data"]

    # Load user configuration
    with open(config_path, 'r') as f:
        user_config = json.load(f)

    # Check required parameters
    for key in required:
        if key not in user_config:
            raise ValueError(f"Missing required parameter in config: {key}")

    # Merge defaults with user configuration
    config = defaults.copy()
    config.update(user_config)

    # Convert to SimpleNamespace
    return SimpleNamespace(**config)

def load_ga_config(config_path):
    """Load genetic algorithm configuration from JSON file and merge with defaults"""
    # Default GA configuration
    defaults = {
        "population_size": 50,
        "n_generations": 20,
        "sample_size": 250,
        "crossover_prob": 0.8,
        "mutation_prob": 0.2,
        "tournament_size": 3,
        "num_elites": 5,
        "use_multiprocessing": True,
        "rules_per_individual": 5,
        "max_possible_rules": 100
    }
    # Load user configuration
    with open(config_path, 'r') as f:
        user_config = json.load(f)

    # Merge defaults with user configuration
    config = defaults.copy()
    config.update(user_config)

    # Convert to SimpleNamespace
    return SimpleNamespace(**config)

def setup_environment(config):
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
    model_base_path = os.path.join(config.output_dir, config.model_base_name)

    return model_base_path


def train_ai_detector(detector_config, ga_config, model_base_path):
    """Train an AI-generated content detector using genetic optimization and CNN"""
    logger.info("=== Step 1: Loading and preparing datasets ===")

    # Initialize the data loader
    data_loader = DataLoader(detector_config)

    # Create the datasets
    train_ds, test_ds, sample_images, sample_labels = data_loader.create_datasets(detector_config, ga_config)
    logger.info(f"Created TensorFlow dataset pipeline for training and testing")

    best_rules = None
    ga_stats = None

    if detector_config.use_feature_extraction:
        logger.info("=== Step 2: Running genetic algorithm for feature optimization ===")

        # Initialize the genetic feature optimizer
        genetic_optimizer = GeneticFeatureOptimizer(
            images=sample_images,
            labels=sample_labels,
            detector_config=detector_config,
            ga_config=ga_config,
        )

        # Run the genetic algorithm optimization
        best_rules, ga_stats = genetic_optimizer.run()

        # best_rules will be a tf.Tensor from the genetic_algorithm module
        # No need to save genetic rules separately here, as ModelWrapper will handle it with save_complete_model_state
        logger.info(f"Optimized {tf.shape(best_rules)[0] if best_rules is not None else 0} genetic rules.")

        if detector_config.visualize: # Check visualize for GA-related plots
            visualizer = Visualizer(output_dir=detector_config.output_dir)
            visualizer.plot_genetic_algorithm_progress(ga_stats['history'])

            sample_image = sample_images[0]
            if detector_config.patch_size: # Ensure patch_size exists before calling
                patch_mask = genetic_optimizer.rules_to_mask(best_rules)
                visualizer.plot_patch_mask(
                    image=sample_image,
                    patch_mask=patch_mask,
                    patch_size=detector_config.patch_size
                )
            else:
                logger.warning("patch_size not defined in detector_config. Cannot visualize patch mask.")
    else:
        logger.info("=== Step 2: Skipping genetic algorithm (use_feature_extraction is False) ===")

    logger.info("=== Step 3: Building and training the model ===")

    # Initialize model wrapper with optimized genetic rules (or None if not using features)
    model_wrapper = ModelWrapper(
        config=detector_config,
        genetic_rules=best_rules, # Pass the TensorFlow tensor directly
    )

    # Train the model with the optimized features
    history = model_wrapper.train(
        train_dataset=train_ds,
        validation_dataset=test_ds,
        model_path=f"{model_base_path}_model.h5" # Specific path for the H5 model
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
    if detector_config.visualize: # Check visualize for general plots
        visualizer = Visualizer(output_dir=detector_config.output_dir)
        visualizer.plot_training_history(history)

        if y_true is not None and y_pred is not None:
            visualizer.plot_confusion_matrix(y_true, y_pred)

    # Save metrics summary
    save_metrics(metrics, history, detector_config.output_dir)

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


def save_metrics(metrics, history, output_dir):
    """Save metrics and training history to files"""
    # Save training history
    if history and hasattr(history, 'history'):
        history_path = os.path.join(output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            # Convert numpy values to Python native types
            history_dict = {}
            for key, values in history.history.items():
                history_dict[key] = [float(x) for x in values]
            json.dump(history_dict, f)

    # Save metrics summary
    metrics_path = os.path.join(output_dir, 'metrics_summary.json')
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

    logger.info(f"Training history and metrics saved to {output_dir}")


def load_model_and_rules(model_base_path, config):
    """Load a trained model and genetic rules using the ModelWrapper's combined load method"""
    logger.info(f"Loading existing model and rules from {model_base_path}.*")

    # Initialize model wrapper (rules will be loaded into it)
    model_wrapper = ModelWrapper(config=config)

    # Load the complete model state
    try:
        model_wrapper.load_complete_model_state(model_base_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find complete model state at {model_base_path}.* : {e}")

    # The genetic rules are now loaded within the model_wrapper
    best_rules = model_wrapper.genetic_rules
    logger.info("Complete model state loaded successfully.")

    return model_wrapper, best_rules


def predict_image(image_path, model_wrapper, config):
    """Make a prediction on a single image"""
    logger.info(f"Making prediction for image: {image_path}")

    # Load and preprocess the image using DataLoader's method
    data_loader = DataLoader(config)
    img, _ = data_loader.process_path(image_path, 0)  # Dummy label, not used
    img = tf.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction
    result = model_wrapper.predict_image(img)

    logger.info(f"Prediction: {result['label']} with {result['confidence']:.2%} confidence")

    # Visualize features if requested
    if config.visualize and config.use_feature_extraction:
        # Pass the model_wrapper directly for visualization
        Visualizer.visualize_ai_features(
            image_path,
            model_wrapper=model_wrapper,
            save_path=os.path.join(config.output_dir, 'ai_feature_visualization.png')
        )

    return result


def main():
    """Main function to coordinate the workflow"""
    # Load configuration from JSON file
    detector_config = load_detector_config('detector_config.json')
    ga_config = load_ga_config('ga_config.json')

    # Setup environment and get base path for model and rules
    model_base_path = setup_environment(detector_config)

    # Print configuration mode
    if detector_config.use_feature_extraction:
        logger.info("=== Running in FULL MODE (with genetic algorithm and feature extraction) ===")
    else:
        logger.info("=== Running in BASELINE MODE (without feature extraction) ===")

    model_wrapper = None
    best_rules = None

    # Load or train model
    # Check for the existence of the combined model state
    model_state_exists = os.path.exists(f"{model_base_path}_config.json")

    if detector_config.skip_training and model_state_exists:
        model_wrapper, best_rules = load_model_and_rules(model_base_path, detector_config)
    else:
        model_wrapper, best_rules, stats = train_ai_detector(detector_config, ga_config, model_base_path)

    # Make prediction if requested
    if detector_config.predict:
        if not detector_config.predict_path:
            logger.error("Prediction path is not specified in config. Cannot make prediction.")
        else:
            predict_image(detector_config.predict_path, model_wrapper, detector_config)

    logger.info("AI detector pipeline completed successfully")
    return model_wrapper, best_rules


if __name__ == "__main__":
    main()