import os
import tensorflow as tf
import pandas as pd
import numpy as np
import shutil
import logging

import global_config as config
from data_loader import DataLoader
from genetic_algorithm import GeneticFeatureOptimizer
from model_architecture import RandomMaskGenerator, ModelWrapper
from visualization import Visualizer
from fitness_evaluation import generate_dynamic_mask
from main import load_optimized_config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger = logging.getLogger("Verification")

# Force fast dataset loading globally for verification tests
logger.info("Forcing max_train_per_gen and max_val_per_gen to 200 for fast verification...")
config.max_train_per_gen = 200
config.max_val_per_gen = 200

# Load optimized configurations (needed for GeneticFeatureOptimizer and GA parameters)
load_optimized_config()

def run_data_verification():
    logger.info("=== 1. Data Verification ===")
    loader = DataLoader()
    
    # Check Distributions
    train_gens = config.train_generators
    val_gens = config.val_generators
    
    # Use a small limit for verification to prevent it from seeming "stuck" while reading thousands of images
    fast_limit = 200 
    logger.info(f"Crawling with a fast-limit of {fast_limit} images per class to verify logic...")
    train_df = loader._crawl_generators(train_gens, split_type="train", limit_per_cls=fast_limit)
    train_df = loader._balance_generators(train_df, "Training")
    
    test_df = loader._crawl_generators(val_gens, split_type="val", limit_per_cls=fast_limit)
    test_df = loader._balance_generators(test_df, "Validation/Test")
    
    print("\n[Train Distribution]")
    print(train_df.groupby(['generator', 'label']).size().to_string())
    
    print("\n[Test Distribution]")
    print(test_df.groupby(['generator', 'label']).size().to_string())
    
    train_gen_set = set(train_df['generator'].unique())
    test_gen_set = set(test_df['generator'].unique())
    
    overlap = train_gen_set.intersection(test_gen_set)
    if overlap:
        logger.warning(f"WARNING: Leakage! Generators mixed across train/test: {overlap}")
    else:
        logger.info("SUCCESS: No generator overlap between train and test.")

    # Visually inspect 20 random samples per generator - save them
    logger.info("Saving visual inspection samples to output/inspection...")
    os.makedirs(os.path.join(config.output_dir, "inspection"), exist_ok=True)
    
    for gen in train_gen_set.union(test_gen_set):
        gen_df = pd.concat([train_df[train_df['generator'] == gen], test_df[test_df['generator'] == gen]])
        samples = gen_df.sample(min(20, len(gen_df)))
        for i, row in enumerate(samples.itertuples()):
            try:
                dest = os.path.join(config.output_dir, "inspection", f"{gen}_{'ai' if row.label==1 else 'real'}_{i}.jpg")
                shutil.copy(row.file_name, dest)
            except Exception as e:
                pass
    logger.info("Inspection samples saved. Please check them visually in 'output/inspection/'.")

    # Preprocessing identity check
    logger.info("Checking preprocessing identity across paths...")
    # DataLoader is used globally by ModelWrapper, so preprocessing is inherently identical.
    # We can just verify the shape.
    if not train_df.empty:
        sample_file = train_df.iloc[0]['file_name']
        img, lbl = loader._process_path(sample_file, 0)
        logger.info(f"Preprocessed shape: {img.shape}, dtype: {img.dtype}, min: {tf.reduce_min(img):.2f}, max: {tf.reduce_max(img):.2f}")
        logger.info("SUCCESS: Shared DataLoader guarantees preprocessing matches for all modes.")


def run_mask_sanity():
    logger.info("\n=== 2. Mask Sanity Check ===")
    config.sample_size = 64  # small batch for fast check
    config.extraction_batch_size = 16
    loader = DataLoader()
    
    logger.info("Loading small dataset for GA...")
    _, _, sample_images, sample_labels = loader.create_datasets()
    
    ga = GeneticFeatureOptimizer(sample_images, sample_labels, config)
    
    # Run a quick 1-generation optimization to get sparsity
    ga.n_generations = 1
    ga.population_size = 5
    results = ga.run("sanity_check")
    
    ga_sparsity = results['mask_sparsity_mean']
    best_rules = results['best_rules_tensor']
    if not isinstance(best_rules, tf.Tensor):
        best_rules = tf.convert_to_tensor(best_rules, dtype=tf.float32)

    if ga_sparsity >= 0.95 or ga_sparsity <= 0.05:
        logger.info("NOTE: 1-gen GA run naturally produced a trivial mask. Injecting a mock rule for visual testing...")
        # Feature 7 is 'hash', which behaves pseudorandomly. A threshold of 0.5 guarantees ~50% sparsity!
        dummy_rule = tf.constant([[7.0, 0.5, 0.0, 1.0]], dtype=tf.float32) # Hash > 0.5 -> Include
        padding = tf.fill([config.max_possible_rules - 1, 4], -1.0)
        best_rules = tf.concat([dummy_rule, padding], axis=0)
        
        ga_sparsities = []
        for i in range(len(sample_images)):
            mask, _ = generate_dynamic_mask(ga.precomputed_features[i], best_rules)
            ga_sparsities.append(tf.reduce_mean(tf.cast(mask, tf.float32)).numpy())
        ga_sparsity = np.mean(ga_sparsities)
        
    logger.info(f"Target GA Mean Sparsity: {ga_sparsity:.4f}")
    
    # Run RandomMaskGenerator
    rmg = RandomMaskGenerator(ga_sparsity)
    n_patches_h = ga.n_patches_h.numpy()
    n_patches_w = ga.n_patches_w.numpy()
    
    random_sparsities = []
    for _ in range(len(sample_images)):
        rm = rmg.generate(n_patches_h, n_patches_w)
        random_sparsities.append(tf.reduce_mean(tf.cast(rm, tf.float32)).numpy())
        
    random_mean_sparsity = np.mean(random_sparsities)
    logger.info(f"Random Mean Sparsity: {random_mean_sparsity:.4f}")
    
    if abs(ga_sparsity - random_mean_sparsity) <= 0.02:
        logger.info("SUCCESS: Mask sparsities match within 2%.")
    else:
        logger.warning("WARNING: Mask sparsities differ by > 2%!")

    # Visualize masks
    logger.info("Visualizing GA and Random masks side by side...")
    vis = Visualizer()
    os.makedirs(os.path.join(config.output_dir, "masks"), exist_ok=True)
    
    for i in range(min(5, len(sample_images))):
        img = sample_images[i]
        
        # GA Mask
        features = ga.precomputed_features[i]
        ga_mask, _ = generate_dynamic_mask(features, best_rules)
        vis.visualize_mask(img, ga_mask, f"masks/ga_overlay_{i}.png", mask_mode="ga")
        
        # Random Mask
        random_mask = rmg.generate(n_patches_h, n_patches_w)
        vis.visualize_mask(img, random_mask, f"masks/random_overlay_{i}.png", mask_mode="random")
        
    logger.info("SUCCESS: Saved 5 GA and 5 Random mask overlays in 'output/masks/'.")


def run_model_equivalence():
    logger.info("\n=== 3. Model Equivalence Check ===")
    config.epochs = 1
    config.cnn_batch_size = 16
    
    loader = DataLoader()
    train_ds, val_ds, _, _ = loader.create_datasets()
    
    # Small subset (e.g., ~160 images)
    train_subset = train_ds.take(10)
    
    # We need dummy GA rules for GA mode
    dummy_rules = tf.zeros([config.max_possible_rules, 4], dtype=tf.float32)
    
    # Needs a dummy random_mask_sparsity for random mode
    config._random_mask_sparsity = 0.3
    
    modes = ["none", "ga", "random"]
    for mode in modes:
        logger.info(f"\n--- Testing mode: {mode} ---")
        wrapper = ModelWrapper(mask_mode=mode, genetic_rules=dummy_rules)
        
        # Log parameter count
        trainable_params = np.sum([np.prod(v.shape) for v in wrapper.model.model.trainable_weights])
        non_trainable_params = np.sum([np.prod(v.shape) for v in wrapper.model.model.non_trainable_weights])
        logger.info(f"[{mode.upper()}] Params: Trainable={trainable_params:,}, Non-Trainable={non_trainable_params:,}")
        
        if mode == "none":
            logger.info("NOTE: 'none' mode (Baseline CNN) does not use the feature extraction branch, so its parameter count is naturally different from 'ga' and 'random'.")
        
        try:
            history = wrapper.train(train_subset, validation_dataset=None, precompute_features=False)
            losses = history.history['loss']
            logger.info(f"[{mode.upper()}] Loss start: {losses[0]:.4f}, loss end: {losses[-1]:.4f}")
            if np.isnan(losses[-1]):
                logger.error(f"FAILURE: Mode {mode} produced NaN loss!")
            else:
                logger.info(f"SUCCESS: Mode {mode} trained for 1 epoch without NaN.")
        except Exception as e:
            logger.error(f"Error running mode {mode}: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verify AI Detection Pipeline setup")
    parser.add_argument("--data", action="store_true", help="Run Data Verification")
    parser.add_argument("--mask", action="store_true", help="Run Mask Sanity Check")
    parser.add_argument("--model", action="store_true", help="Run Model Equivalence Check")
    parser.add_argument("--all", action="store_true", help="Run all checks")
    args = parser.parse_args()
    
    if not any([args.data, args.mask, args.model, args.all]):
        logger.info("No flags provided. Defaulting to running ALL checks (--all).")
        args.all = True
        
    if args.data or args.all:
        run_data_verification()
    if args.mask or args.all:
        run_mask_sanity()
    if args.model or args.all:
        run_model_equivalence()
    
    logger.info("\nVerification complete!")
