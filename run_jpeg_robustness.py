# run_jpeg_robustness.py
import os
import sys
import json
import logging
import argparse
import numpy as np
import tensorflow as tf
import builtins
builtins.tf = tf
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Import project modules
import global_config as config
import optuna_config as opt_config
from data_loader import DataLoader
from model_architecture import ModelWrapper
from utils import apply_jpeg_compression

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger('jpeg_robustness_experiment')

def parse_args():
    parser = argparse.ArgumentParser(description="Run Standalone JPEG Robustness Experiment")
    parser.add_argument('--qualities', type=int, nargs='+', 
                        default=[100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
                        help="JPEG quality levels to evaluate (default: 100 down to 10)")
    parser.add_argument('--max-samples', type=int, default=1000,
                        help="Maximum test samples per setup to speed up evaluation")
    parser.add_argument('--output-dir', type=str, default='output',
                        help="Directory to save JSON metrics")
    parser.add_argument('--paper-dir', type=str, default='paper',
                        help="Directory to save final comparison figures")
    return parser.parse_args()

def _prepare_eval_dataset(dataset, max_samples):
    """
    Limit a dataset to *max_samples* items and fully populate a local cache.

    data_loader returns a test_ds with an outer .cache() sized for the full
    validation set (e.g. 5000 items).  If we call .take(k) on top of that and
    then iterate only k items, TF discards the partially-filled outer cache and
    warns.  The fix: apply .take(k).cache() to create a NEW, smaller cache that
    is immediately warmed by iterating through it once.  All subsequent passes
    (clean baseline + every JPEG quality level) read from this inner cache,
    never touching the outer one again.

    Args:
        dataset:     A tf.data.Dataset (unbatched, float32 images).
        max_samples: Number of items to retain, or None to keep all.

    Returns:
        A fully-warmed tf.data.Dataset cache.
    """
    if max_samples is not None:
        dataset = dataset.take(max_samples)
    # Build an in-memory cache of exactly these samples.
    dataset = dataset.cache()
    # Warm the cache: iterate once (batched for speed) so every element is stored
    # before evaluation begins.  This prevents the outer data_loader cache from
    # being partially read and discarded when the first model accesses this dataset.
    logger.info(f"Warming evaluation cache ({max_samples or 'all'} samples)...")
    for _ in dataset.batch(256).prefetch(tf.data.AUTOTUNE):
        pass
    logger.info("Cache warm-up complete.")
    return dataset


def evaluate_model_robustness(model_wrapper, test_ds, qualities, cache_prefix="robustness"):
    """
    Evaluate accuracy of a model wrapper over a range of JPEG qualities.

    *test_ds* must already be limited and cached (call ``_prepare_eval_dataset``
    before this function).  Every quality level and the clean baseline evaluate
    on exactly the same fixed set of images.

    For GA/RANDOM modes, raw patch features are precomputed and cached on disk
    *once* per quality variant (clean + each JPEG level).  Subsequent model modes
    (RANDOM after GA) reuse the same raw-feature cache because patch features are
    mode-independent — the per-mode mask is applied cheaply from the cached data.
    Use *cache_prefix* to namespace caches by setup (in-dist vs cross-gen) so
    they never cross-contaminate.
    """
    # 1. Baseline accuracy on clean images
    if model_wrapper.use_features:
        clean_cache = f"{cache_prefix}_clean"
        model_wrapper.precompute_features(test_ds, clean_cache)
        clean_prepared, _ = model_wrapper.prepare_dataset(test_ds, clean_cache, is_training=False)
    else:
        clean_prepared, _ = model_wrapper.prepare_dataset(test_ds, is_training=False)

    clean_results = model_wrapper.model.model.evaluate(clean_prepared, verbose=0)
    baseline_acc = clean_results[1] if len(clean_results) > 1 else 0.5
    logger.info(f"  Clean baseline accuracy: {baseline_acc:.4f}")

    results = {
        'clean': baseline_acc,
        'qualities': {},
        'drops': {}
    }

    # 2. Evaluate accuracy across JPEG qualities.
    # test_ds is a pre-warmed cache, so every iteration sees the same samples
    # in the same order — no cache truncation, no TF warnings.
    for q in qualities:
        # Default-arg capture (_q=q) prevents the classic loop-closure bug where
        # all iterations would see the last value of q if captured by reference.
        def compress_single(img, label, _q=q):
            compressed = apply_jpeg_compression(img, _q)
            compressed.set_shape([config.image_size, config.image_size, 3])
            return compressed, label

        compressed_ds = test_ds.map(
            lambda img, label: compress_single(img, label),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        if model_wrapper.use_features:
            # Precompute features from JPEG-compressed images once; subsequent
            # model modes for the same quality level reuse the cached features.
            q_cache = f"{cache_prefix}_q{q}"
            model_wrapper.precompute_features(compressed_ds, q_cache)
            prepared, _ = model_wrapper.prepare_dataset(compressed_ds, q_cache, is_training=False)
        else:
            prepared, _ = model_wrapper.prepare_dataset(compressed_ds, is_training=False)

        eval_results = model_wrapper.model.model.evaluate(prepared, verbose=0)
        acc = eval_results[1] if len(eval_results) > 1 else 0.5
        drop = baseline_acc - acc

        logger.info(f"  JPEG Quality {q:3d} -> Accuracy: {acc:.4f} (Drop: {drop:+.4f})")
        results['qualities'][q] = acc
        results['drops'][q] = drop

    return results

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.paper_dir, exist_ok=True)

    try:
        tf.keras.config.enable_unsafe_deserialization()
    except AttributeError:
        pass

    # Load optimized feature weights and GA config
    weights_path = opt_config.feature_weights_output_file
    ga_path = opt_config.ga_config_output_file
    if os.path.exists(weights_path):
        with open(weights_path, 'r') as f:
            raw_weights = json.load(f)
        config.feature_weights = {k: v for k, v in raw_weights.items() if not k.startswith("__")}
        logger.info(f"Loaded feature weights from {weights_path}")
    if os.path.exists(ga_path):
        with open(ga_path, 'r') as f:
            optimized_ga = json.load(f)
        for key, value in optimized_ga.items():
            setattr(config, key, value)
        logger.info(f"Loaded GA configuration from {ga_path}")

    logger.info("="*70)
    logger.info("STANDALONE JPEG ROBUSTNESS EXPERIMENT")
    logger.info(f"Qualities: {args.qualities}")
    logger.info(f"Max Samples: {args.max_samples}")
    logger.info("="*70)

    # 1. Prepare data loaders and discover datasets
    data_loader = DataLoader()
    
    # Experiment generator splits
    in_dist_train = list(config.train_generators)
    in_dist_test  = list(config.train_generators)
    cross_test    = list(config.val_generators)
    
    # Generate datasets.
    # create_sample=False: skip loading ~10 000 sample images not needed here.
    # For in-dist: test generators == train generators (held-out val split).
    # For cross-gen: BOTH train_generators and val_generators are set to the
    # cross-gen list so create_datasets loads only those generators — no ADM/
    # glide/wukong training data is included in the cross-gen dataset creation.
    logger.info("Loading In-distribution test dataset...")
    config.train_generators = in_dist_train
    config.val_generators   = in_dist_test
    _, in_dist_ds, _, _ = data_loader.create_datasets(create_sample=False)
    
    logger.info("Loading Cross-generator test dataset...")
    config.train_generators = cross_test   # load only cross-gen splits
    config.val_generators   = cross_test
    _, cross_ds, _, _ = data_loader.create_datasets(create_sample=False)

    # Restore original config generator lists
    config.train_generators = in_dist_train
    config.val_generators   = list(cross_test)

    # Pre-limit and pre-warm each test dataset ONCE before the model loop.
    # data_loader attaches a .cache() sized for the full validation set (e.g. 5000
    # items).  If we called .take(max_samples) inside evaluate_model_robustness,
    # every model invocation would create a fresh inner cache and partially read
    # the outer one, triggering the TF cache-truncation warning repeatedly.  By
    # warming a single shared cache here, all subsequent model calls read from
    # this cache without ever touching the outer one again.
    logger.info("Pre-limiting and warming evaluation datasets...")
    in_dist_ds = _prepare_eval_dataset(in_dist_ds, args.max_samples)
    cross_ds   = _prepare_eval_dataset(cross_ds,   args.max_samples)

    setups = [
        # (name, cache_prefix, test_dataset, train_gens, test_gens)
        ('In-Distribution', 'indist', in_dist_ds, in_dist_train, in_dist_test),
        ('Cross-Generator', 'cross',  cross_ds,   in_dist_train, cross_test)
    ]

    modes = ['none', 'ga', 'random']
    all_metrics = {}

    for setup_name, cache_prefix, test_ds, train_gens, test_gens in setups:
        logger.info(f"\nEvaluating Setup: {setup_name} (Train={train_gens}, Test={test_gens})")
        all_metrics[setup_name] = {}

        gen_train_tag = "_".join(train_gens)
        gen_test_tag  = "_".join(test_gens)

        for mode in modes:
            logger.info(f"Evaluating Model Mode: '{mode.upper()}'...")

            # Reconstruct the saved model wrapper base path
            experiment_tag = f"{mode}__train_{gen_train_tag}__test_{gen_test_tag}"
            model_base_path = os.path.join(args.output_dir, f"model_{experiment_tag}")

            if not os.path.exists(f"{model_base_path}_config.json"):
                logger.warning(f"Model config not found at {model_base_path}_config.json. Skipping.")
                continue

            try:
                # Load the model wrapper
                model_wrapper = ModelWrapper.load(model_base_path)

                # Run the quality scan (test_ds is already limited and warmed;
                # cache_prefix namespaces feature caches by setup so in-dist and
                # cross-gen features are stored separately)
                res = evaluate_model_robustness(
                    model_wrapper=model_wrapper,
                    test_ds=test_ds,
                    qualities=args.qualities,
                    cache_prefix=cache_prefix,
                )
                all_metrics[setup_name][mode] = res
            except Exception as e:
                logger.error(f"Failed to evaluate model '{mode}' in setup '{setup_name}': {e}")
                import traceback
                traceback.print_exc()

    # 3. Save JSON metrics
    metrics_json_path = os.path.join(args.output_dir, 'jpeg_robustness_metrics.json')
    with open(metrics_json_path, 'w') as f:
        # Convert keys to strings for JSON serializability
        serializable_metrics = {}
        for setup, modes_data in all_metrics.items():
            serializable_metrics[setup] = {}
            for mode, res in modes_data.items():
                serializable_metrics[setup][mode] = {
                    'clean': float(res['clean']),
                    'qualities': {str(k): float(v) for k, v in res['qualities'].items()},
                    'drops': {str(k): float(v) for k, v in res['drops'].items()}
                }
        json.dump(serializable_metrics, f, indent=2)
    logger.info(f"Saved raw metrics to {metrics_json_path}")

    # 4. Generate Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = {'none': '#7f8c8d', 'ga': '#00bcd4', 'random': '#e67e22'}
    markers = {'none': 'o', 'ga': '^', 'random': 's'}
    labels = {'none': 'No Mask (Baseline)', 'ga': 'GA Evolved Mask', 'random': 'Random Mask (Control)'}

    for idx, (setup_name, _, _, _) in enumerate(setups):
        ax = axes[idx]
        modes_data = all_metrics.get(setup_name, {})
        
        # Plot clean baseline as quality=100 or a dashed baseline?
        # We'll plot quality levels on x-axis
        for mode in modes:
            if mode not in modes_data:
                continue
            
            q_data = modes_data[mode]['qualities']
            sorted_qs = sorted(list(q_data.keys()), reverse=True)
            accs = [q_data[q] for q in sorted_qs]
            
            # Plot the line
            ax.plot(sorted_qs, accs, label=labels[mode], color=colors[mode], 
                    marker=markers[mode], linewidth=2.5, markersize=8)

        ax.set_title(f"{setup_name} JPEG Robustness", fontsize=14, fontweight='bold')
        ax.set_xlabel("JPEG Quality Level (Lower = More Compression)", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_xlim(105, 5) # reverse x axis to show degradation from left to right
        ax.set_ylim(0.45, 1.0)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc="lower left", fontsize=10)

    plt.tight_layout()
    plot_path = os.path.join(args.paper_dir, 'jpeg_robustness_comparison.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    logger.info(f"Saved comparative robustness plot to {plot_path}")
    logger.info("="*70)
    logger.info("JPEG ROBUSTNESS EXPERIMENT RUN COMPLETED SUCCESSFULLY")
    logger.info("="*70)

if __name__ == "__main__":
    main()
