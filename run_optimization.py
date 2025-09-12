
from hyperparameter_optimizer import HyperparameterOptimizer
import optuna_config as config

def main():
    """Run the hyperparameter optimization process."""
    print("="*60)
    print("GENETIC ALGORITHM HYPERPARAMETER OPTIMIZATION")
    print("="*60)
    print(f"Feature Weight Trials: {config.feature_weight_trials}")
    print(f"GA Configuration Trials: {config.ga_config_trials}")
    print(f"Sample Size: {config.sample_size}")
    print(f"Random Seed: {config.optimization_seed}")
    print("="*60)
    
    # Create and run optimizer
    optimizer = HyperparameterOptimizer()
    best_weights, best_ga_config = optimizer.run_optimization()
    
    print("\nOptimization completed successfully!")
    print(f"Best fitness achieved: {optimizer.best_fitness:.4f}")
    print(f"Results saved to JSON files")

if __name__ == "__main__":
    main()