import optuna # type: ignore
import json
import logging
import time
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Any, Tuple

# Import your project modules
from data_loader import DataLoader
from genetic_algorithm import GeneticFeatureOptimizer
import optuna_config as config

# Configure logging
logging.basicConfig(
    level=logging.INFO if config.verbosity >= 1 else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """
    Handles hyperparameter optimization for the genetic algorithm feature optimizer.
    Separates optimization into two phases: feature weights, then GA configuration.
    Reuses a single GeneticFeatureOptimizer instance with precomputed features for efficiency.
    """
    
    def __init__(self):
        """Initialize the optimizer with data loading and create reusable GA instance."""
        logger.info("Initializing HyperparameterOptimizer...")
        
        # Load data once at the beginning
        self.data_loader = DataLoader()
        self.train_ds, self.test_ds, self.sample_images, self.sample_labels = self.data_loader.create_datasets()
        logger.info(f"Loaded dataset with {len(self.sample_images)} samples")
        
        # Create a single GeneticFeatureOptimizer instance with default config
        # Features will be precomputed once during initialization
        base_config = self.create_base_config()
        logger.info("Creating GeneticFeatureOptimizer instance (features will be precomputed once)...")
        
        self.genetic_optimizer = GeneticFeatureOptimizer(
            images=self.sample_images,
            labels=self.sample_labels,
            config=base_config
        )
        
        logger.info("GeneticFeatureOptimizer created with precomputed features - ready for optimization!")
        
        # Store best results
        self.best_feature_weights = None
        self.best_ga_config = None
        self.best_fitness = -float('inf')
        
        # Set random seed if specified
        if config.optimization_seed is not None:
            optuna.samplers.RandomSampler(seed=config.optimization_seed)
            np.random.seed(config.optimization_seed)
    
    def create_base_config(self, feature_weights: Dict[str, float] = None, 
                          ga_params: Dict[str, Any] = None) -> Any:
        """
        Create a configuration object with the given parameters.
        
        Args:
            feature_weights: Dictionary of feature weights
            ga_params: Dictionary of GA parameters
            
        Returns:
            Configuration object compatible with GeneticFeatureOptimizer
        """
        class Config:
            pass
        
        cfg = Config()
        
        # Base configuration
        cfg.image_size = config.image_size
        cfg.patch_size = config.patch_size
        cfg.sample_size = config.sample_size
        cfg.extraction_batch_size = config.extraction_batch_size
        cfg.use_feature_extraction = config.use_feature_extraction
        cfg.fitness_weights = config.fitness_weights.copy()
        
        # Feature weights
        if feature_weights is not None:
            cfg.feature_weights = feature_weights.copy()
        else:
            cfg.feature_weights = config.default_feature_weights.copy()
        
        # GA parameters
        if ga_params is not None:
            for key, value in ga_params.items():
                setattr(cfg, key, value)
        else:
            for key, value in config.default_ga_config.items():
                setattr(cfg, key, value)
        
        return cfg
    
    def update_optimizer_config(self, feature_weights: Dict[str, float] = None, 
                               ga_params: Dict[str, Any] = None):
        """
        Update the configuration of the existing genetic optimizer without recreating it.
        This is much more efficient than creating new instances.
        
        Args:
            feature_weights: New feature weights to use
            ga_params: New GA parameters to use
        """
        # Update feature weights if provided
        if feature_weights is not None:
            # Update feature weights in the optimizer
            self.genetic_optimizer.config.feature_weights = feature_weights.copy()
            # Update the tensor version used in evaluation
            import tensorflow as tf
            self.genetic_optimizer.feature_weights = tf.convert_to_tensor(
                list(feature_weights.values()), dtype=tf.float32
            )
            
        # Update GA parameters if provided
        if ga_params is not None:
            for key, value in ga_params.items():
                if hasattr(self.genetic_optimizer.config, key):
                    setattr(self.genetic_optimizer.config, key, value)
                # Also update the optimizer's direct attributes for commonly used parameters
                if hasattr(self.genetic_optimizer, key):
                    setattr(self.genetic_optimizer, key, value)
            
            # Re-setup DEAP if population or selection parameters changed
            population_changed = any(key in ga_params for key in [
                'population_size', 'tournament_size', 'num_elites'
            ])
            if population_changed:
                self.genetic_optimizer._setup_deap()
    
    def normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize weights to sum to 1.0.
        
        Args:
            weights: Dictionary of unnormalized weights
            
        Returns:
            Dictionary of normalized weights
        """
        total = sum(weights.values())
        if total == 0:
            # Fallback to equal weights
            return {key: 1.0 / len(weights) for key in weights}
        return {key: value / total for key, value in weights.items()}
    
    def suggest_feature_weights(self, trial: optuna.Trial) -> Dict[str, float]:
        """
        Suggest feature weights for optimization trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested feature weights (normalized to sum to 1.0)
        """
        weights = {}
        for feature, (min_val, max_val) in config.feature_weight_ranges.items():
            weights[feature] = trial.suggest_float(
                f"weight_{feature}", 
                min_val, 
                max_val
            )
        
        # Normalize to sum to 1.0
        return self.normalize_weights(weights)
    
    def suggest_ga_config(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest GA configuration parameters for optimization trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested GA parameters
        """
        ga_config = {}
        
        ga_config['population_size'] = trial.suggest_int(
            'population_size', 
            *config.population_size_range
        )
        
        ga_config['n_generations'] = trial.suggest_int(
            'n_generations', 
            *config.n_generations_range
        )
        
        ga_config['rules_per_individual'] = trial.suggest_int(
            'rules_per_individual', 
            *config.rules_per_individual_range
        )
        
        ga_config['max_possible_rules'] = trial.suggest_int(
            'max_possible_rules', 
            *config.max_possible_rules_range
        )
        
        ga_config['crossover_prob'] = trial.suggest_float(
            'crossover_prob', 
            *config.crossover_prob_range
        )
        
        ga_config['mutation_prob'] = trial.suggest_float(
            'mutation_prob', 
            *config.mutation_prob_range
        )
        
        ga_config['tournament_size'] = trial.suggest_int(
            'tournament_size', 
            *config.tournament_size_range
        )
        
        ga_config['num_elites'] = trial.suggest_int(
            'num_elites', 
            *config.num_elites_range
        )
        
        # Fixed parameters
        ga_config['random_seed'] = config.optimization_seed
        ga_config['verbose'] = False  # Keep quiet during optimization
        
        # Ensure max_possible_rules >= rules_per_individual
        if ga_config['max_possible_rules'] < ga_config['rules_per_individual']:
            ga_config['max_possible_rules'] = ga_config['rules_per_individual']
        
        return ga_config
    
    def objective_feature_weights(self, trial: optuna.Trial) -> float:
        """
        Objective function for optimizing feature weights.
        Reuses the same GeneticFeatureOptimizer instance with updated weights.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Fitness score (higher is better)
        """
        try:
            # Suggest feature weights
            feature_weights = self.suggest_feature_weights(trial)
            
            # Update the existing optimizer with new feature weights
            self.update_optimizer_config(
                feature_weights=feature_weights,
                ga_params=config.default_ga_config  # Use default GA config for Phase 1
            )
            
            if config.verbosity >= 2:
                logger.info(f"Trial {trial.number}: Testing feature weights: {feature_weights}")
            
            # Run GA with updated weights (features are already precomputed!)
            best_ind, ga_stats = self.genetic_optimizer.run(run_id=f"fw_trial_{trial.number}")
            fitness = best_ind.fitness.values[0]
            
            # Report intermediate result for pruning
            if config.enable_pruning:
                trial.report(fitness, 0)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            # Early stopping for very poor performance
            if fitness < config.min_fitness_threshold:
                raise optuna.TrialPruned()
            
            if config.verbosity >= 1:
                logger.info(f"Trial {trial.number}: Fitness = {fitness:.4f}")
            
            return fitness
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {str(e)}")
            return -1.0  # Return poor fitness for failed trials
    
    def objective_ga_config(self, trial: optuna.Trial) -> float:
        """
        Objective function for optimizing GA configuration.
        Uses the best feature weights found in the previous phase.
        Reuses the same GeneticFeatureOptimizer instance with updated config.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Fitness score (higher is better)
        """
        try:
            # Suggest GA configuration
            ga_params = self.suggest_ga_config(trial)
            
            # Use best feature weights from previous optimization
            if self.best_feature_weights is None:
                logger.warning("No best feature weights found, using defaults")
                feature_weights = config.default_feature_weights
            else:
                feature_weights = self.best_feature_weights
            
            # Update the existing optimizer with new GA config and best weights
            self.update_optimizer_config(
                feature_weights=feature_weights,
                ga_params=ga_params
            )
            
            if config.verbosity >= 2:
                logger.info(f"Trial {trial.number}: Testing GA config: {ga_params}")
            
            # Run GA with updated configuration (features are already precomputed!)
            best_ind, ga_stats = self.genetic_optimizer.run(run_id=f"ga_trial_{trial.number}")
            fitness = best_ind.fitness.values[0]
            
            # Report intermediate results for pruning
            if config.enable_pruning and 'history' in ga_stats:
                for i, gen_data in enumerate(ga_stats['history']):
                    if i >= config.pruning_warmup_steps and i % config.pruning_interval == 0:
                        trial.report(gen_data['max_fitness'], i)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
            
            # Early stopping for very poor performance
            if fitness < config.min_fitness_threshold:
                raise optuna.TrialPruned()
            
            if config.verbosity >= 1:
                logger.info(f"Trial {trial.number}: Fitness = {fitness:.4f}")
            
            return fitness
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {str(e)}")
            return -1.0  # Return poor fitness for failed trials
    
    def optimize_feature_weights(self) -> Dict[str, float]:
        """
        Optimize feature weights using Optuna.
        
        Returns:
            Dictionary of best feature weights
        """
        logger.info("="*60)
        logger.info("PHASE 1: Optimizing Feature Weights")
        logger.info("="*60)
        logger.info("Using precomputed features - no re-extraction needed!")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            study_name=config.feature_weight_study_name,
            pruner=optuna.pruners.MedianPruner() if config.enable_pruning else None
        )
        
        # Optimize
        start_time = time.time()
        study.optimize(
            self.objective_feature_weights, 
            n_trials=config.feature_weight_trials,
            show_progress_bar=config.show_progress_bar
        )
        end_time = time.time()
        
        # Get best results
        best_trial = study.best_trial
        best_weights = self.suggest_feature_weights(best_trial)
        
        logger.info(f"Feature weight optimization completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Best fitness: {best_trial.value:.4f}")
        logger.info(f"Best feature weights: {best_weights}")
        
        # Save results
        self.best_feature_weights = best_weights
        self.save_json(best_weights, config.feature_weights_output_file)
        
        return best_weights
    
    def optimize_ga_config(self) -> Dict[str, Any]:
        """
        Optimize GA configuration using Optuna.
        Uses the best feature weights from the previous optimization.
        
        Returns:
            Dictionary of best GA configuration
        """
        logger.info("="*60)
        logger.info("PHASE 2: Optimizing GA Configuration")
        logger.info("="*60)
        logger.info("Using precomputed features - no re-extraction needed!")
        
        if self.best_feature_weights is None:
            logger.warning("No feature weights optimized yet, using defaults")
            self.best_feature_weights = config.default_feature_weights
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            study_name=config.ga_config_study_name,
            pruner=optuna.pruners.MedianPruner() if config.enable_pruning else None
        )
        
        # Optimize
        start_time = time.time()
        study.optimize(
            self.objective_ga_config, 
            n_trials=config.ga_config_trials,
            show_progress_bar=config.show_progress_bar
        )
        end_time = time.time()
        
        # Get best results
        best_trial = study.best_trial
        best_ga_config = self.suggest_ga_config(best_trial)
        
        logger.info(f"GA config optimization completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Best fitness: {best_trial.value:.4f}")
        logger.info(f"Best GA config: {best_ga_config}")
        
        # Save results
        self.best_ga_config = best_ga_config
        self.best_fitness = best_trial.value
        self.save_json(best_ga_config, config.ga_config_output_file)
        
        return best_ga_config
    
    def save_json(self, data: Dict, filename: str):
        """
        Save dictionary as JSON file.
        
        Args:
            data: Dictionary to save
            filename: Output filename
        """
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved results to {filename}")
        except Exception as e:
            logger.error(f"Failed to save {filename}: {str(e)}")
    
    def save_combined_config(self):
        """Save combined configuration with both feature weights and GA config."""
        if self.best_feature_weights is None or self.best_ga_config is None:
            logger.warning("Cannot save combined config - missing optimization results")
            return
        
        combined_config = {
            'feature_weights': self.best_feature_weights,
            'ga_config': self.best_ga_config,
            'best_fitness': self.best_fitness,
            'base_config': {
                'image_size': config.image_size,
                'patch_size': config.patch_size,
                'fitness_weights': config.fitness_weights
            }
        }
        
        self.save_json(combined_config, config.combined_config_output_file)
    
    def run_optimization(self) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Run the complete optimization process.
        
        Returns:
            Tuple of (best_feature_weights, best_ga_config)
        """
        logger.info("Starting hyperparameter optimization...")
        logger.info(f"Total trials: {config.feature_weight_trials + config.ga_config_trials}")
        logger.info("EFFICIENCY BOOST: Using single GA instance with precomputed features!")
        
        start_time = time.time()
        
        try:
            # Phase 1: Optimize feature weights
            best_weights = self.optimize_feature_weights()
            
            # Phase 2: Optimize GA configuration
            best_ga_config = self.optimize_ga_config()
            
            # Save combined results
            self.save_combined_config()
            
            end_time = time.time()
            total_time = end_time - start_time
            
            logger.info("="*60)
            logger.info("OPTIMIZATION COMPLETE")
            logger.info("="*60)
            logger.info(f"Total optimization time: {total_time:.2f} seconds")
            logger.info(f"Best overall fitness: {self.best_fitness:.4f}")
            logger.info(f"Results saved to:")
            logger.info(f"  - Feature weights: {config.feature_weights_output_file}")
            logger.info(f"  - GA config: {config.ga_config_output_file}")
            logger.info(f"  - Combined: {config.combined_config_output_file}")
            
            return best_weights, best_ga_config
            
        except KeyboardInterrupt:
            logger.info("Optimization interrupted by user")
            if self.best_feature_weights:
                self.save_json(self.best_feature_weights, config.feature_weights_output_file)
            if self.best_ga_config:
                self.save_json(self.best_ga_config, config.ga_config_output_file)
            raise
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            raise


def main():
    """Main function to run the hyperparameter optimization."""
    try:
        optimizer = HyperparameterOptimizer()
        best_weights, best_ga_config = optimizer.run_optimization()
        
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(f"Best Feature Weights: {best_weights}")
        print(f"Best GA Configuration: {best_ga_config}")
        print(f"Best Fitness Achieved: {optimizer.best_fitness:.4f}")
        
    except KeyboardInterrupt:
        print("\nOptimization stopped by user.")
    except Exception as e:
        print(f"\nOptimization failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()