import argparse
import sys
import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from hyperparameter_optimizer import HyperparameterOptimizer
import optuna_config as config
import global_config

logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.WARNING)

def main():
    """Run the hyperparameter optimization process."""
    parser = argparse.ArgumentParser(description="Genetic AI Detector Hyperparameter Optimization")
    parser.add_argument('--phase', type=str, choices=['weights', 'ga', 'both'], default='both',
                        help='Optimization phase to run (weights, ga, or both)')
    args = parser.parse_args()

    print("="*60)
    print("GENETIC ALGORITHM HYPERPARAMETER OPTIMIZATION")
    print(f"Phase: {args.phase.upper()}")
    print("="*60)
    print(f"Feature Weight Trials: {config.feature_weight_trials}")
    print(f"GA Configuration Trials: {config.ga_config_trials}")
    print(f"Max Images: {global_config.max_images}")
    print(f"Sample Size: {global_config.sample_size}")
    print(f"Random Seed: {config.optimization_seed}")
    print("="*60)
    
    try:
        # Create optimizer
        optimizer = HyperparameterOptimizer()
        
        # Run Phase 1: Feature Weights
        if args.phase in ['weights', 'both']:
            print("\nRUNNING PHASE 1: Feature Weight Optimization...")
            best_weights = optimizer.optimize_feature_weights()
        
        # Run Phase 2: GA Configuration
        if args.phase in ['ga', 'both']:
            print("\nRUNNING PHASE 2: GA Configuration Optimization...")
            best_ga_config = optimizer.optimize_ga_config()
        
        print("\nOptimization process completed successfully!")
        print(f"Results saved to {config.feature_weights_output_file} and {config.ga_config_output_file}")
        
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