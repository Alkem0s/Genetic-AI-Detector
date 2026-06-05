import argparse
import sys
import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
# Setup GPU memory growth before any other module imports tensorflow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        pass

from hyperparameter_optimizer import HyperparameterOptimizer
import optuna_config as config
import global_config

logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.WARNING)

def main():
    """Run the hyperparameter optimization process."""
    parser = argparse.ArgumentParser(description="Genetic AI Detector Hyperparameter Optimization")
    parser.add_argument('--phase', type=str, choices=['weights', 'ga', 'cnn'], required=True,
                        help='Optimization phase to run (weights, ga, or cnn)')
    parser.add_argument('--mask-mode', type=str, choices=['ga', 'random', 'none', 'all'], default='all',
                        help='Mask mode to tune CNN for (only used for cnn phase: ga, random, none, or all)')
    args = parser.parse_args()

    print("="*60)
    print("GENETIC ALGORITHM HYPERPARAMETER OPTIMIZATION")
    print(f"Phase: {args.phase.upper()}")
    if args.phase == 'cnn':
        print(f"Mask Mode: {args.mask_mode.upper()}")
    print("="*60)
    print(f"Feature Weight Trials: {config.feature_weight_trials}")
    print(f"GA Configuration Trials: {config.ga_config_trials}")
    print(f"CNN Trials: {getattr(config, 'cnn_trials', 20)}")
    print(f"Max Train Samples: {global_config.max_train_samples}")
    print(f"Max Val Samples: {global_config.max_val_samples}")
    print(f"Sample Size: {global_config.sample_size}")
    print(f"Random Seed: {config.optimization_seed}")
    print("="*60)
    
    try:
        # Create optimizer
        optimizer = HyperparameterOptimizer()
        
        # Run Phase 1: Feature Weights
        if args.phase == 'weights':
            print("\nRUNNING PHASE 1: Feature Weight Optimization...")
            best_weights = optimizer.optimize_feature_weights()
            print(f"\nFeature weight optimization completed successfully! Saved to {config.feature_weights_output_file}")
        
        # Run Phase 2: GA Configuration
        elif args.phase == 'ga':
            print("\nRUNNING PHASE 2: GA Configuration Optimization...")
            best_ga_config = optimizer.optimize_ga_config()
            print(f"\nGA configuration optimization completed successfully! Saved to {config.ga_config_output_file}")
            
        # Run Phase 3: CNN Hyperparameters
        elif args.phase == 'cnn':
            mask_modes = ['ga', 'none'] if args.mask_mode == 'all' else [args.mask_mode]
            for mode in mask_modes:
                print(f"\nRUNNING PHASE 3: CNN Hyperparameter Optimization for mask_mode '{mode}'...")
                optimizer.optimize_cnn(mode)
                output_file = config.cnn_config_output_file_template.format(mask_mode=mode)
                print(f"Saved optimized CNN config for '{mode}' to {output_file}")
            print("\nCNN hyperparameter optimization completed successfully!")
        
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nOptimization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()