import os
import json
import numpy as np
from types import SimpleNamespace
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
# from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

# Import our custom modules
from preprocessor import load_and_split_data, preprocess_dataset, evaluate_model, save_model_artifacts
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
        "use_feature_extraction": True
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
    
    return os.path.join(config.output_dir, config.model_path), os.path.join(config.output_dir, config.mask_path)


def generate_default_mask(config):
    """Generate a default mask when genetic algorithm is disabled"""
    # Create a simple uniform mask without optimization
    print("Generating default feature extraction mask (no genetic optimization)")
    mask_size = config.image_size // config.patch_size
    return np.ones((mask_size, mask_size), dtype=np.float32)


def train_model_workflow(config, model_path, mask_path):
    """Execute the full training workflow"""
    print("\n=== Step 1: Loading and preprocessing data ===")
    X_train_paths, X_test_paths, y_train_unfiltered, y_test_unfiltered = load_and_split_data(
        config.data, 
        test_size=config.test_size, 
        random_state=config.random_seed
    )
    
    # Preprocess images
    X_train, X_test, train_valid_indices, test_valid_indices = preprocess_dataset(
        X_train_paths, 
        X_test_paths, 
        target_size=(config.image_size, config.image_size)
    )
    
    # Filter the labels to match the successfully loaded images
    y_train = y_train_unfiltered[train_valid_indices]
    y_test = y_test_unfiltered[test_valid_indices]
    
    print(f"Loaded {len(X_train)} training images and {len(X_test)} test images")
    
    # Default mask (needed even when not using feature extraction for consistency)
    if config.use_genetic_algorithm and config.use_feature_extraction:
        print("\n=== Step 2: Running genetic algorithm for feature optimization ===")
        # Initialize and run genetic feature optimizer
        optimizer = GeneticFeatureOptimizer(
            feature_extractor=AIFeatureExtractor,
            images=X_train,
            labels=y_train, 
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
    
    # Feature extraction step
    if config.use_feature_extraction:
        print("\n=== Step 3: Generating feature maps using mask ===")
        # Generate feature maps using the mask
        X_train_features, _ = AIFeatureExtractor.generate_feature_maps(X_train, best_mask, image_paths=X_train_paths)
        X_test_features, _ = AIFeatureExtractor.generate_feature_maps(X_test, best_mask, image_paths=X_test_paths)
        print(f"Generated feature maps of shape {X_train_features.shape} for training data")
        print(f"Generated feature maps of shape {X_test_features.shape} for test data")
        feature_channels = X_train_features.shape[-1]
    else:
        print("\n=== Step 3: Skipping feature extraction ===")
        # When feature extraction is disabled, set features to None
        X_train_features = None
        X_test_features = None
        feature_channels = 0  # No feature channels when extraction is disabled
    
    print("\n=== Step 4: Building and training the model ===")
    # Initialize model wrapper with appropriate configuration
    model_wrapper = ModelWrapper(
        input_shape=(config.image_size, config.image_size, 3),
        feature_channels=feature_channels,
        use_features=config.use_feature_extraction
    )
    
    # Train the model
    history = model_wrapper.train_model(
        X_train, 
        X_train_features, 
        y_train,
        validation_split=0.2,  # Uses 20% of X_train for validation
        epochs=config.epochs,
        batch_size=config.batch_size,
        model_path=model_path
    )
    
    print("\n=== Step 5: Evaluating the model ===")
    # Get the trained model
    model = model_wrapper.get_model()
    
    # Evaluate the model
    metrics = evaluate_model(model, X_test, X_test_features, y_test, config)
    
    # Print evaluation results
    print(f"\nTest accuracy: {metrics['accuracy']:.4f}")
    print("\nClassification Report:")
    print(metrics['classification_report'])
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    # Visualize results
    if config.visualize:
        # Plot training history
        Visualizer.plot_training_history(history, 
                                        save_path=os.path.join(config.output_dir, 'training_history.png'))
        
        # Plot confusion matrix
        Visualizer.plot_confusion_matrix(metrics['confusion_matrix'], 
                                        save_path=os.path.join(config.output_dir, 'confusion_matrix.png'))
    
    # Save model artifacts
    save_model_artifacts(model, best_mask, history, metrics, output_dir=config.output_dir)
    
    return model, best_mask


def load_existing_model(model_path, mask_path, config):
    """Load an existing model and mask"""
    print(f"\nLoading existing model from {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found!")
    
    model_wrapper = ModelWrapper(use_features=config.use_feature_extraction)
    model_wrapper.load_model(model_path)
    
    print(f"Loading mask from {mask_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file {mask_path} not found!")
    
    best_mask = np.load(mask_path)
    
    return model_wrapper.get_model(), best_mask


def predict_image(image_path, model, best_mask, config):
    """Make a prediction on a single image"""
    print(f"\n=== Making prediction for image: {image_path} ===")
    
    # Load and preprocess the image
    img = load_img(image_path, target_size=(config.image_size, config.image_size))
    img_array = np.expand_dims(img_to_array(img) / 255.0, axis=0)
    
    # Generate feature maps if feature extraction is enabled
    if config.use_feature_extraction:
        img_features, _ = AIFeatureExtractor.generate_feature_maps(img_array, best_mask)
    else:
        # When feature extraction is disabled, set features to None
        img_features = None
    
    # Create model wrapper to use for prediction
    model_wrapper = ModelWrapper(use_features=config.use_feature_extraction)
    model_wrapper.set_model(model)
    
    # Make prediction
    result = model_wrapper.predict_image(img_array, img_features)
    
    print(f"Prediction: {result['label']} with {result['confidence']:.2%} confidence")
    
    # Visualize the prediction and features
    if config.visualize and config.use_feature_extraction:
        Visualizer.visualize_ai_features(
            image_path, 
            AIFeatureExtractor, 
            save_path=os.path.join(config.output_dir, 'ai_feature_visualization.png')
        )
        
        Visualizer.visualize_prediction_results(
            image_path,
            result['raw_prediction'],
            result['confidence'],
            feature_maps=img_features,
            save_path=os.path.join(config.output_dir, 'prediction_visualization.png')
        )
    
    return result


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