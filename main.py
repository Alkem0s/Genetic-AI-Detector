# main.py
import os
import sys
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
from fitness_evaluation import generate_dynamic_mask

import global_config as config
import optuna_config as opt_config

# Silence TensorFlow spam
logging.getLogger('tensorflow').setLevel(logging.ERROR)

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
    
    # Silence detailed GA logs unless explicitly requested via global_config.verbose
    logging.getLogger('genetic_algorithm').setLevel(logging.INFO if not config.verbose else logging.DEBUG)
    
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


def load_optimized_config():
    """
    Load feature weights and GA parameters from JSON files.
    This is the STRICT source of truth. Fails if files are missing.
    """
    weights_path = opt_config.feature_weights_output_file
    ga_path = opt_config.ga_config_output_file
    
    if not os.path.exists(weights_path) or not os.path.exists(ga_path):
        logger.error("="*60)
        logger.error("FATAL ERROR: CONFIGURATION FILES MISSING")
        logger.error(f"Required: {weights_path} and {ga_path}")
        logger.error("Please run 'python run_optimization.py' to generate them or provide them manually.")
        logger.error("="*60)
        sys.exit(1)

    try:
        # Load weights
        with open(weights_path, 'r') as f:
            config.feature_weights = json.load(f)
        logger.info(f"Successfully loaded feature weights from {weights_path}")
        
        # Load GA config
        with open(ga_path, 'r') as f:
            optimized_ga = json.load(f)
        
        # Apply all parameters from JSON to the global config
        for key, value in optimized_ga.items():
            setattr(config, key, value)
            
        logger.info(f"Successfully loaded GA parameters from {ga_path}")
        
    except Exception as e:
        logger.error(f"FATAL: Error parsing configuration files: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------

def run_experiment(mask_mode: str, generator_train: list, generator_test: list) -> dict:
    """
    Train and evaluate one model configuration.

    Args:
        mask_mode:       ``"none"``, ``"ga"``, or ``"random"``.
        generator_train: List of generator names used as the training set.
        generator_test:  List of generator names used as the test set.

    Returns:
        dict with keys: mask_mode, generator_train, generator_test, metrics,
        robustness, mask_sparsity (if applicable), training_history.
    """
    logger.info(
        f"\n{'='*70}\n"
        f"  run_experiment  mask_mode={mask_mode!r}\n"
        f"  train={generator_train}  test={generator_test}\n"
        f"{'='*70}"
    )

    load_optimized_config()

    # --- Step 1: Data loading --------------------------------------------------
    logger.info("=== Step 1: Loading and preparing datasets ===")
    config.train_generators = generator_train
    config.val_generators   = generator_test

    data_loader = DataLoader()
    train_ds, test_ds, sample_images, sample_labels = data_loader.create_datasets()
    logger.info("Created TensorFlow dataset pipeline for training and testing")

    # --- Step 2: GA (only for "ga" mode) --------------------------------------
    results = None
    mask_sparsity_mean = None
    mask_sparsity_std  = None

    if mask_mode == 'ga':
        logger.info("=== Step 2: Running genetic algorithm for feature optimisation ===")
        genetic_optimizer = GeneticFeatureOptimizer(
            images=sample_images,
            labels=sample_labels,
            config=config,
        )
        with tf.profiler.experimental.Trace('genetic_optimization_phase'):
            results = genetic_optimizer.run()

        mask_sparsity_mean = results.get('mask_sparsity_mean')
        mask_sparsity_std  = results.get('mask_sparsity_std')
        logger.info(
            f"GA complete – sparsity mean={mask_sparsity_mean:.4f}, "
            f"std={mask_sparsity_std:.4f}"
        )

    elif mask_mode == 'random':
        logger.info(
            "=== Step 2: Running GA first to obtain sparsity target "
            "for random mask control ==="
        )
        genetic_optimizer = GeneticFeatureOptimizer(
            images=sample_images,
            labels=sample_labels,
            config=config,
        )
        with tf.profiler.experimental.Trace('genetic_optimization_phase'):
            results = genetic_optimizer.run()

        mask_sparsity_mean = results['mask_sparsity_mean']
        mask_sparsity_std  = results['mask_sparsity_std']
        logger.info(
            f"GA sparsity target for random masks: mean={mask_sparsity_mean:.4f}, "
            f"std={mask_sparsity_std:.4f}"
        )
        # Store on config so ModelWrapper can pick it up
        config._random_mask_sparsity = mask_sparsity_mean

    else:  # mask_mode == 'none'
        logger.info("=== Step 2: Skipping genetic algorithm (baseline mode) ===")

    # --- Step 3: Build & train model ------------------------------------------
    logger.info("=== Step 3: Building and training the model ===")

    genetic_rules = None
    if mask_mode == 'ga' and results is not None:
        genetic_rules = results['best_individual'].rules_tensor

    model_wrapper = ModelWrapper(
        genetic_rules=genetic_rules,
        mask_mode=mask_mode,
        random_mask_sparsity=mask_sparsity_mean,
    )

    # Unique model base path per experiment
    gen_train_tag = "_".join(generator_train)
    gen_test_tag  = "_".join(generator_test)
    experiment_tag = f"{mask_mode}__train_{gen_train_tag}__test_{gen_test_tag}"
    model_base_path = os.path.join(
        config.output_dir, f"model_{experiment_tag}"
    )

    with tf.profiler.experimental.Trace('model_training_phase'):
        history = model_wrapper.train(
            train_dataset=train_ds,
            validation_dataset=test_ds,
            model_path=f"{model_base_path}.h5",
        )

    model_wrapper.save(model_base_path)
    logger.info(f"Model state saved to {model_base_path}.*")

    # --- Step 4: Evaluate -----------------------------------------------------
    logger.info("=== Step 4: Evaluating the model ===")
    metrics, y_true, y_pred = evaluate_model(model_wrapper, test_ds)
    logger.info(f"Test accuracy: {metrics['accuracy']:.4f}  loss: {metrics['loss']:.4f}")

    # --- Step 4b: JPEG robustness evaluation ----------------------------------
    logger.info("=== Step 4b: JPEG robustness evaluation ===")
    from utils import evaluate_robustness
    robustness = evaluate_robustness(
        model_wrapper, test_ds,
        quality_levels=getattr(config, 'jpeg_quality_levels', [50, 75]),
        logger=logger,
    )

    # --- Step 5: Mask visualisation (10 sample images per model) --------------
    if mask_mode in ('ga', 'random') and len(sample_images) > 0:
        logger.info("=== Step 5: Generating mask visualisations ===")
        _visualize_masks_for_experiment(
            model_wrapper=model_wrapper,
            sample_images=sample_images,
            mask_mode=mask_mode,
            experiment_tag=experiment_tag,
            n_samples=10,
            ga_results=results,
        )

    # --- Step 6: Plot training curves -----------------------------------------
    if config.visualize:
        vis = Visualizer()
        vis.plot_training_history(history, filename=f"training_history_{experiment_tag}.png")
        if y_true is not None and y_pred is not None:
            vis.plot_confusion_matrix(y_true, y_pred, filename=f"confusion_matrix_{experiment_tag}.png")

    # --- Step 7: Save metrics to JSON -----------------------------------------
    out_json = _save_experiment_results(
        mask_mode=mask_mode,
        generator_train=generator_train,
        generator_test=generator_test,
        metrics=metrics,
        robustness=robustness,
        history=history,
        mask_sparsity_mean=mask_sparsity_mean,
        mask_sparsity_std=mask_sparsity_std,
        experiment_tag=experiment_tag,
    )
    logger.info(f"Results saved → {out_json}")

    return {
        'mask_mode': mask_mode,
        'generator_train': generator_train,
        'generator_test': generator_test,
        'metrics': metrics,
        'robustness': robustness,
        'mask_sparsity_mean': mask_sparsity_mean,
        'mask_sparsity_std':  mask_sparsity_std,
        'training_history': history.history if hasattr(history, 'history') else {},
        'results_json': out_json,
    }


def _visualize_masks_for_experiment(
    model_wrapper, sample_images, mask_mode, experiment_tag, n_samples, ga_results
):
    """Generate mask overlay PNGs for *n_samples* images from sample_images."""
    from fitness_evaluation import generate_dynamic_mask

    vis = Visualizer()
    n = min(n_samples, len(sample_images))
    n_patches_h = int(model_wrapper.n_patches_h.numpy())
    n_patches_w = int(model_wrapper.n_patches_w.numpy())

    for i in range(n):
        img = sample_images[i]  # float32, (H, W, 3)

        if mask_mode == 'ga' and ga_results is not None:
            # Extract patch features for this image
            img_tensor = tf.convert_to_tensor(img[np.newaxis], dtype=tf.float32)
            model_wrapper.pipeline._ensure_feature_extractor()
            patch_features = (
                model_wrapper.pipeline._feature_extractor
                .extract_patch_features(img_tensor[0])
            )
            patch_mask, _ = generate_dynamic_mask(
                patch_features,
                ga_results['best_individual'].rules_tensor,
            )
        elif mask_mode == 'random':
            patch_mask = model_wrapper._random_generator.generate(
                n_patches_h, n_patches_w
            )
        else:
            continue  # 'none' has no mask to visualise

        filename = f"mask_{experiment_tag}_sample{i:03d}.png"
        vis.visualize_mask(
            image_tensor=img,
            patch_mask=patch_mask,
            filename=filename,
            mask_mode=mask_mode,
        )


def _save_experiment_results(
    mask_mode, generator_train, generator_test,
    metrics, robustness, history,
    mask_sparsity_mean, mask_sparsity_std,
    experiment_tag,
):
    """Persist structured experiment results to output/ as JSON."""
    # Build a JSON-serialisable payload
    history_dict = {}
    if history and hasattr(history, 'history'):
        history_dict = {k: [float(v) for v in vs] for k, vs in history.history.items()}

    def _safe_float(v):
        try:
            return float(v) if v is not None else None
        except Exception:
            return None

    payload = {
        'mask_mode': mask_mode,
        'generator_train': generator_train,
        'generator_test':  generator_test,
        'mask_sparsity_mean': _safe_float(mask_sparsity_mean),
        'mask_sparsity_std':  _safe_float(mask_sparsity_std),
        'metrics': {
            'accuracy':          _safe_float(metrics.get('accuracy')),
            'loss':              _safe_float(metrics.get('loss')),
            'f1':                _safe_float(metrics.get('f1')),
            'balanced_accuracy': _safe_float(metrics.get('balanced_accuracy')),
            'precision':         _safe_float(metrics.get('precision')),
            'recall':            _safe_float(metrics.get('recall')),
            'confusion_matrix':  metrics.get('confusion_matrix'),
        },
        'robustness': {
            str(q): {'accuracy': _safe_float(v['accuracy']), 'accuracy_drop': _safe_float(v['accuracy_drop'])}
            for q, v in (robustness or {}).items()
        },
        'training_history': history_dict,
    }

    os.makedirs(config.output_dir, exist_ok=True)
    out_path = os.path.join(config.output_dir, f"results_{experiment_tag}.json")
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    return out_path


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_all_experiments():
    """
    Run the full 4-experiment comparison matrix:

    1. In-distribution  GA masks
    2. In-distribution  random masks (matched sparsity)
    3. Cross-generator  GA masks
    4. Cross-generator  random masks (matched sparsity)

    Results are saved as individual JSON files in output/.
    """
    gen_train = list(getattr(config, 'generator_train', config.train_generators))
    gen_test  = list(getattr(config, 'generator_test',  config.val_generators))

    experiments = [
        # (mask_mode, train_gens,  test_gens)
        ('ga',     gen_train, gen_train),   # in-distribution GA
        ('random', gen_train, gen_train),   # in-distribution random
        ('ga',     gen_train, gen_test),    # cross-generator GA
        ('random', gen_train, gen_test),    # cross-generator random
    ]

    all_results = []
    for mask_mode, g_train, g_test in experiments:
        res = run_experiment(
            mask_mode=mask_mode,
            generator_train=g_train,
            generator_test=g_test,
        )
        all_results.append(res)

    logger.info("="*60)
    logger.info("All experiments complete.  Summary:")
    for r in all_results:
        tag = f"{r['mask_mode']:8s}  train={r['generator_train']}  test={r['generator_test']}"
        acc = r['metrics'].get('accuracy')
        logger.info(f"  {tag}  →  accuracy={acc:.4f}" if acc is not None else f"  {tag}  →  N/A")
    logger.info("="*60)

    return all_results


# ---------------------------------------------------------------------------
# Legacy helpers (kept for backward compatibility / predict mode)
# ---------------------------------------------------------------------------

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
        
        # Profile the prediction loop
        with tf.profiler.experimental.Trace('model_evaluation_predictions'):
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

    mask_mode = getattr(config, 'mask_mode', 'ga')
    logger.info(f"=== Running in mask_mode='{mask_mode}' ===")

    model_wrapper = None
    best_rules = None

    # Load or train model
    model_state_exists = os.path.exists(f"{model_base_path}_config.json")

    if config.skip_training and model_state_exists:
        model_wrapper, best_rules = load_model_and_rules(model_base_path)
    else:
        gen_train = list(getattr(config, 'generator_train', config.train_generators))
        gen_test  = list(getattr(config, 'generator_test',  config.val_generators))
        result = run_experiment(
            mask_mode=mask_mode,
            generator_train=gen_train,
            generator_test=gen_test,
        )
        # For predict mode we need a loaded wrapper; reload from saved state
        model_base_path_exp = result.get('results_json', '').replace(
            os.path.join(config.output_dir, 'results_'), ''
        ).replace('.json', '')
        try:
            model_wrapper, best_rules = load_model_and_rules(
                os.path.join(config.output_dir, f"model_{model_base_path_exp}")
            )
        except Exception:
            pass  # predict mode won't work but training is complete

    # Make prediction if requested
    if config.predict:
        if not config.predict_path:
            logger.error("Prediction path is not specified in config. Cannot make prediction.")
        elif model_wrapper is not None:
            predict_image(config.predict_path, model_wrapper)

    logger.info("AI detector pipeline completed successfully")
    return model_wrapper, best_rules


if __name__ == "__main__":
    main()