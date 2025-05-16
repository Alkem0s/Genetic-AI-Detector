import os
import json
import numpy as np
from types import SimpleNamespace
import tensorflow as tf

# Import our custom modules 
from data_loader import DataLoader
from feature_extractor import AIFeatureExtractor
from genetic_algorithm import GeneticFeatureOptimizer
from model_architecture import ModelWrapper
from visualization import Visualizer


def load_config(config_path):
    """Load configuration from JSON file and merge with defaults"""
    # Default configuration
    defaults = {
        "output_dir": "output",
        "image_size": 224,
        "batch_size": 32,
        "max_images": 1000,
        "epochs": 50,
        "test_size": 0.2,
        "random_seed": 42,
        "model_path": "ai_detector_model.h5",
        "mask_path": "optimized_patch_mask.npy",
        "visualize": False,
        "predict": None,
        "skip_training": False,
        "population_size": 50,
        "n_generations": 20,
        "patch_size": 16,
        "use_genetic_algorithm": True,
        "use_feature_extraction": True,
        "sample_size_for_ga": 1000,  # Use only this many samples for GA
        "mixed_precision": True      # Enable mixed precision for better performance
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


def setup_environment(config):
    """Setup environment variables and configurations"""
    # Create output directory if it doesn't exist
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Set random seeds for reproducibility
    np.random.seed(config.random_seed)
    tf.random.set_seed(config.random_seed)
    
    # Setup GPU memory growth to avoid OOM errors
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU(s) available: {len(gpus)}")
        except RuntimeError as e:
            print(f"Error setting up GPU: {e}")
    else:
        print("No GPUs found, using CPU")
    
    # Enable mixed precision if specified
    if config.mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision enabled (float16)")
    
    return os.path.join(config.output_dir, config.model_path), os.path.join(config.output_dir, config.mask_path)


def generate_default_mask(config):
    """Generate a default mask when genetic algorithm is disabled"""
    # Create a simple uniform mask without optimization
    print("Generating default feature extraction mask (no genetic optimization)")
    mask_size = config.image_size // config.patch_size
    return np.ones((mask_size, mask_size), dtype=np.float32)


def update_feature_maps_on_batch(images, mask, feature_extractor):
    """Process feature maps in small batches to save memory"""
    batch_size = 32  # Small batch size for feature extraction
    num_samples = len(images)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    all_features = []
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        
        batch_images = images[start_idx:end_idx]
        batch_features, _ = feature_extractor.generate_feature_maps(batch_images, mask)
        
        all_features.append(batch_features)
    
    return np.vstack(all_features)


def train_model_workflow(config, model_path, mask_path):
    """Execute the full training workflow"""
    print("\n=== Step 1: Loading and preparing datasets ===")
    # Initialize the data loader
    data_loader = DataLoader(config)
    
    # Create the datasets
    train_ds, test_ds, sample_images, sample_labels = data_loader.create_datasets(config.data)
    
    print(f"Created TensorFlow dataset pipeline for training and testing")
    
    # Default mask (needed even when not using feature extraction for consistency)
    if config.use_genetic_algorithm and config.use_feature_extraction:
        print("\n=== Step 2: Running genetic algorithm for feature optimization ===")
        # Initialize and run genetic feature optimizer on a sample of the data
        optimizer = GeneticFeatureOptimizer(
            feature_extractor=AIFeatureExtractor,
            images=sample_images,  # Use sample images for GA
            labels=sample_labels,  # Use sample labels for GA
            config=config,
            population_size=config.population_size,
            n_generations=config.n_generations
        )
        
        best_mask, stats = optimizer.run()
        
        # Visualize genetic algorithm results if requested
        if config.visualize:
            Visualizer.plot_genetic_optimization_stats(stats, 
                                                    save_path=os.path.join(config.output_dir, 'genetic_optimization.png'))
            Visualizer.visualize_optimized_patch_mask(best_mask, 
                                                    save_path=os.path.join(config.output_dir, 'optimized_patches.png'))
    else:
        print("\n=== Step 2: Skipping genetic algorithm, using default mask ===")
        best_mask = generate_default_mask(config)
        stats = None
    
    # Save the mask (optimized or default)
    np.save(mask_path, best_mask)
    print(f"Mask saved to {mask_path}")
    
    print("\n=== Step 3: Building and training the model ===")
    # Feature extraction is now done inside the model during training
    
    # Initialize model wrapper with appropriate configuration
    model_wrapper = ModelWrapper(
        input_shape=(config.image_size, config.image_size, 3),
        feature_channels=3,  # Default feature channels
        use_features=config.use_feature_extraction,
        mask=best_mask      # Pass the mask to the model
    )
    
    # Custom training loop for the model
    history = model_wrapper.train_with_datasets(
        train_ds=train_ds,
        val_split=0.2,
        epochs=config.epochs,
        model_path=model_path
    )
    
    print("\n=== Step 4: Evaluating the model ===")
    # Get the trained model
    model = model_wrapper.get_model()
    
    # Evaluate on the test dataset
    metrics = evaluate_model_with_dataset(model, test_ds, config)
    
    # Print evaluation results
    print(f"\nTest accuracy: {metrics['accuracy']:.4f}")
    print(f"Test loss: {metrics['loss']:.4f}")
    
    # Visualize results
    if config.visualize:
        # Plot training history
        Visualizer.plot_training_history(history, 
                                        save_path=os.path.join(config.output_dir, 'training_history.png'))
        
        # Plot confusion matrix if available
        if 'confusion_matrix' in metrics:
            Visualizer.plot_confusion_matrix(metrics['confusion_matrix'], 
                                           save_path=os.path.join(config.output_dir, 'confusion_matrix.png'))
    
    # Save model artifacts
    save_model_artifacts(model, best_mask, history, metrics, 
                         model_path=model_path, 
                         mask_path=mask_path,
                         output_dir=config.output_dir)
    
    return model, best_mask


def evaluate_model_with_dataset(model, test_ds, config):
    """Evaluate the model using a TensorFlow dataset"""
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
    
    # Basic metrics report
    metrics_report = {
        'accuracy': metrics['accuracy'],
        'loss': metrics['loss']
    }
    
    # Compute confusion matrix and classification report (optional)
    # This would require collecting all predictions and true labels
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
        
        metrics_report['classification_report'] = classification_report(
            y_true, y_pred, target_names=['Human', 'AI'], output_dict=True
        )
        
        metrics_report['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Human', 'AI']))
        
        print("\nConfusion Matrix:")
        print(metrics_report['confusion_matrix'])
        
    except Exception as e:
        print(f"Warning: Could not compute detailed metrics: {e}")
    
    return metrics_report


def load_existing_model(model_path, mask_path, config):
    """Load an existing model and mask"""
    print(f"\nLoading existing model from {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found!")
    
    model_wrapper = ModelWrapper(
        input_shape=(config.image_size, config.image_size, 3),
        use_features=config.use_feature_extraction
    )
    model_wrapper.load_model(model_path)
    
    print(f"Loading mask from {mask_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file {mask_path} not found!")
    
    best_mask = np.load(mask_path)
    model_wrapper.set_mask(best_mask)
    
    return model_wrapper.get_model(), best_mask


def predict_image(image_path, model, best_mask, config):
    """Make a prediction on a single image"""
    print(f"\n=== Making prediction for image: {image_path} ===")
    
    # Load the image as a TensorFlow tensor
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [config.image_size, config.image_size])
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    
    # Create model wrapper to use for prediction
    model_wrapper = ModelWrapper(
        input_shape=(config.image_size, config.image_size, 3),
        use_features=config.use_feature_extraction,
        mask=best_mask
    )
    model_wrapper.set_model(model)
    
    # Make prediction
    result = model_wrapper.predict_image(img)
    
    print(f"Prediction: {result['label']} with {result['confidence']:.2%} confidence")
    
    # Visualize the prediction and features
    if config.visualize and config.use_feature_extraction:
        Visualizer.visualize_ai_features(
            image_path, 
            AIFeatureExtractor,
            mask=best_mask,
            save_path=os.path.join(config.output_dir, 'ai_feature_visualization.png')
        )
        
        # Optional: Visualize prediction results
        # (This would need to be adapted to the new workflow)
    
    return result


def save_model_artifacts(model, best_mask, history, metrics, 
                         model_path='ai_detector_model.h5',
                         mask_path='optimized_patch_mask.npy',
                         output_dir='output'):
    """
    Save model and related artifacts.
    
    Args:
        model: Trained model
        best_mask: Optimized patch mask
        history: Training history
        metrics: Evaluation metrics
        model_path: Path to save model
        mask_path: Path to save mask
        output_dir: Directory to save artifacts
    """
    # Save model
    model.save(model_path)
    
    # Save patch mask
    np.save(mask_path, best_mask)
    
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
    
    print(f"Model saved to {model_path}")
    print(f"Optimized patch mask saved to {mask_path}")
    print(f"Training history and metrics saved to {output_dir}")
    
    return {
        'model_path': model_path,
        'mask_path': mask_path,
        'history_path': os.path.join(output_dir, 'training_history.json'),
        'metrics_path': metrics_path 
    }


def main():
    """Main function to coordinate the workflow"""
    
    # Load configuration from JSON file
    config = load_config('config.json')
    
    # Print configuration mode
    if config.use_genetic_algorithm and config.use_feature_extraction:
        print("\n=== Running in FULL MODE (with genetic algorithm and feature extraction) ===")
    elif config.use_feature_extraction:
        print("\n=== Running in FEATURE EXTRACTION MODE (without genetic algorithm) ===")
    elif not config.use_feature_extraction:
        print("\n=== Running in BASELINE MODE (without genetic algorithm or feature extraction) ===")
    
    # Setup environment
    model_path, mask_path = setup_environment(config)
    
    # Load or train model
    if config.skip_training and os.path.exists(model_path) and os.path.exists(mask_path):
        model, best_mask = load_existing_model(model_path, mask_path, config)
    else:
        model, best_mask = train_model_workflow(config, model_path, mask_path)
    
    # Make prediction if requested
    if config.predict:
        predict_image(config.predict, model, best_mask, config)


if __name__ == "__main__":
    main()