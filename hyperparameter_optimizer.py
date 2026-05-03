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
import global_config

# Configure logging
logging.basicConfig(
    level=logging.INFO if config.verbosity >= 1 else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
# Ensure the GA logger specifically respects the level to avoid DEBUG spam
logging.getLogger('genetic_algorithm').setLevel(logging.INFO if config.verbosity >= 1 else logging.WARNING)
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
        _, _, all_images, all_labels = self.data_loader.create_datasets()
        
        # Implement 80/20 Train-Test Split for Honesty
        num_samples = len(all_images)
        split_idx = int(num_samples * 0.8)
        
        # Shuffle indices for a fair split
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        self.train_images = all_images[train_indices]
        self.train_labels = all_labels[train_indices]
        self.test_images = all_images[test_indices]
        self.test_labels = all_labels[test_indices]
        
        logger.info(f"Dataset split: {len(self.train_images)} Training, {len(self.test_images)} Validation")
        
        # 1. Load Strict Source-of-Truth JSONs (Optimization Baseline)
        try:
            with open(config.feature_weights_output_file, 'r') as f:
                self.json_weights = json.load(f)
            with open(config.ga_config_output_file, 'r') as f:
                self.json_ga_config = json.load(f)
        except Exception as e:
            logger.error(f"FATAL: Could not load baseline configuration from JSON: {e}")
            logger.error(f"Please ensure {config.feature_weights_output_file} and {config.ga_config_output_file} exist.")
            sys.exit(1)

        # 2. Create a single GeneticFeatureOptimizer instance
        # Features will be precomputed once during initialization
        base_config = self.create_base_config()
        logger.info("Creating GeneticFeatureOptimizer instance (features will be precomputed once)...")
        
        self.genetic_optimizer = GeneticFeatureOptimizer(
            images=self.train_images,
            labels=self.train_labels,
            config=base_config
        )
        
        # Save the auto-calibrated scales for future use/inference
        self.genetic_optimizer.save_feature_scales("feature_scales.json")
        
        logger.info("GeneticFeatureOptimizer created with precomputed features - ready for optimization!")
        
        # Store best results
        self.best_fitness = -float('inf')
        self.best_individual = None
        self.fw_study = None
        self.ga_study = None
        
        # Set random seed if specified
        if config.optimization_seed is not None:
            optuna.samplers.RandomSampler(seed=config.optimization_seed)
            np.random.seed(config.optimization_seed)
    
    def create_base_config(self, feature_weights: Dict[str, float] = None, 
                          ga_params: Dict[str, Any] = None):
        """Create a configuration object combining global settings with overrides."""
        class Config:
            pass
        
        cfg = Config()
        
        # Copy environment attributes from global_config
        for attr in dir(global_config):
            if not attr.startswith("__"):
                setattr(cfg, attr, getattr(global_config, attr))
        
        # Use existing JSON loads if not provided as overrides
        # Set feature weights (Passed weights override JSON weights)
        cfg.feature_weights = feature_weights.copy() if feature_weights is not None else self.json_weights.copy()
        
        # Set GA parameters (Passed params override JSON params)
        base_ga = ga_params if ga_params is not None else self.json_ga_config
        for key, value in base_ga.items():
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
            # Explicitly cast to float to avoid any SQLite/numpy compatibility issues
            weights[feature] = float(trial.suggest_float(
                f"weight_{feature}", 
                float(min_val), 
                float(max_val)
            ))
        
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
        
        ga_config['population_size'] = int(trial.suggest_int(
            'population_size', 
            int(config.population_size_range[0]), 
            int(config.population_size_range[1])
        ))
        
        ga_config['n_generations'] = int(trial.suggest_int(
            'n_generations', 
            int(config.n_generations_range[0]), 
            int(config.n_generations_range[1])
        ))
        
        ga_config['rules_per_individual'] = int(trial.suggest_int(
            'rules_per_individual', 
            int(config.rules_per_individual_range[0]), 
            int(config.rules_per_individual_range[1])
        ))
        
        ga_config['max_possible_rules'] = int(trial.suggest_int(
            'max_possible_rules', 
            int(config.max_possible_rules_range[0]), 
            int(config.max_possible_rules_range[1])
        ))
        
        ga_config['crossover_prob'] = float(trial.suggest_float(
            'crossover_prob', 
            float(config.crossover_prob_range[0]), 
            float(config.crossover_prob_range[1])
        ))
        
        ga_config['mutation_prob'] = float(trial.suggest_float(
            'mutation_prob', 
            float(config.mutation_prob_range[0]), 
            float(config.mutation_prob_range[1])
        ))
        
        ga_config['tournament_size'] = int(trial.suggest_int(
            'tournament_size', 
            int(config.tournament_size_range[0]), 
            int(config.tournament_size_range[1])
        ))
        
        ga_config['num_elites'] = int(trial.suggest_int(
            'num_elites', 
            int(config.num_elites_range[0]), 
            int(config.num_elites_range[1])
        ))
        
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
                ga_params=self.json_ga_config  # Use baseline GA config for Phase 1
            )
            
            if config.verbosity >= 2:
                logger.info(f"Trial {trial.number}: Testing feature weights: {feature_weights}")
            
            # Run GA with updated weights (features are already precomputed!)
            results = self.genetic_optimizer.run(run_id=f"fw_trial_{trial.number}")
            fitness = results['best_fitness']
            
            # Report intermediate result for pruning
            if config.enable_pruning:
                trial.report(fitness, 0)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            # Early stopping for very poor performance
            if fitness < config.min_fitness_threshold:
                raise optuna.TrialPruned()
            
            # Sanitize fitness to avoid SQLite issues with NaN/Inf
            fitness = float(np.nan_to_num(fitness, nan=-1.0, posinf=-1.0, neginf=-1.0))
            
            if config.verbosity >= 1:
                logger.info(f"Trial {trial.number}: Fitness = {fitness:.4f}")
            
            # Update internal best for tracking
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_feature_weights = feature_weights
                self.best_individual = results['best_individual']

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
            
            # Use best feature weights from previous optimization or baseline JSON
            if self.best_feature_weights is None:
                feature_weights = self.json_weights
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
            results = self.genetic_optimizer.run(run_id=f"ga_trial_{trial.number}")
            fitness = results['best_fitness']
            
            # Report intermediate results for pruning
            if config.enable_pruning and 'history' in results:
                for i, gen_data in enumerate(results['history']):
                    if i >= config.pruning_warmup_steps and i % config.pruning_interval == 0:
                        trial.report(gen_data['max_fitness'], i)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
            
            # Early stopping for very poor performance
            if fitness < config.min_fitness_threshold:
                raise optuna.TrialPruned()
            
            # Sanitize fitness to avoid SQLite issues with NaN/Inf
            fitness = float(np.nan_to_num(fitness, nan=-1.0, posinf=-1.0, neginf=-1.0))
            
            if config.verbosity >= 1:
                logger.info(f"Trial {trial.number}: Fitness = {fitness:.4f}")
            
            # Update internal best
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_ga_config = ga_params
                self.best_individual = results['best_individual']

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
        
        # Use robust RDBStorage with WAL mode and high timeout
        storage_url = 'sqlite:///optuna_study.db'
        
        # Ensure we use WAL mode for better concurrency and reliability
        storage = optuna.storages.RDBStorage(
            url=storage_url,
            engine_kwargs={
                "connect_args": {"timeout": 60},
                "pool_pre_ping": True
            }
        )
        
        # Set WAL mode manually via PRAGMA
        import sqlite3
        try:
            conn = sqlite3.connect('optuna_study.db')
            conn.execute("PRAGMA journal_mode=WAL")
            conn.close()
        except Exception as e:
            logger.warning(f"Could not set WAL mode: {e}")

        self.fw_study = optuna.create_study(
            direction='maximize',
            study_name=config.feature_weight_study_name,
            storage=storage,
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner() if config.enable_pruning else None
        )
        study = self.fw_study
        
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
        
        # 3. Honest Validation
        if self.best_individual is not None:
            logger.info("\n" + "="*60)
            logger.info("FINAL HONEST VALIDATION (PHASE 1)")
            logger.info("="*60)
            val_fitness = self.genetic_optimizer.validate(
                self.test_images, self.test_labels, self.best_individual
            )
            logger.info(f"Phase 1 Train Fitness: {best_trial.value:.4f}")
            logger.info(f"Phase 1 Test Fitness:  {val_fitness:.4f}")
            logger.info("="*60 + "\n")
        
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
            self.best_feature_weights = self.json_weights
        
        # Use robust RDBStorage with WAL mode and high timeout
        storage_url = 'sqlite:///optuna_study.db'
        storage = optuna.storages.RDBStorage(
            url=storage_url,
            engine_kwargs={
                "connect_args": {"timeout": 60},
                "pool_pre_ping": True
            }
        )
        
        self.ga_study = optuna.create_study(
            direction='maximize',
            study_name=config.ga_config_study_name,
            storage=storage,
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner() if config.enable_pruning else None
        )
        study = self.ga_study
        
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
        
        # 3. Honest Validation
        if self.best_individual is not None:
            logger.info("\n" + "="*60)
            logger.info("FINAL HONEST VALIDATION (PHASE 2)")
            logger.info("="*60)
            val_fitness = self.genetic_optimizer.validate(
                self.test_images, self.test_labels, self.best_individual
            )
            logger.info(f"Phase 2 Train Fitness: {best_trial.value:.4f}")
            logger.info(f"Phase 2 Test Fitness:  {val_fitness:.4f}")
            logger.info("="*60 + "\n")
            
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
            
            end_time = time.time()
            total_time = end_time - start_time
            
            logger.info("="*60)
            logger.info("OPTIMIZATION COMPLETE")
            logger.info("="*60)
            logger.info(f"Total optimization time: {total_time:.2f} seconds")
            logger.info(f"  - Feature weights: {config.feature_weights_output_file}")
            logger.info(f"  - GA config: {config.ga_config_output_file}")
            
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