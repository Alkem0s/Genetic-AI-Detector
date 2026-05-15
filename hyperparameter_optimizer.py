import optuna # type: ignore
from optuna.storages import JournalStorage, JournalFileStorage
import json
import logging
import argparse
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
        
        # Use the FULL training sample for optimisation.
        # The val_generators (sdv4, vqdm) form a completely separate held-out set
        # loaded lazily by _get_val_sample() for the final validation only.
        self.train_images = all_images
        self.train_labels = all_labels
        
        # Placeholder — loaded on demand from global_config.val_generators
        self._val_images = None
        self._val_labels = None
        
        logger.info(
            f"Training sample: {len(self.train_images)} images "
            f"from train_generators={global_config.train_generators}"
        )
        logger.info(
            f"Validation will use val_generators={global_config.val_generators} "
        )
        
        # 1. Load Strict Source-of-Truth JSONs (Optimization Baseline)
        try:
            with open(config.feature_weights_output_file, 'r') as f:
                raw_weights = json.load(f)
                # Filter out metadata fields like '__fitness__'
                self.json_weights = {k: v for k, v in raw_weights.items() if not k.startswith("__")}
            with open(config.ga_config_output_file, 'r') as f:
                self.json_ga_config = json.load(f)
                
            # Override with Proxy GA config if requested in optuna_config
            if getattr(config, 'use_proxy_ga_config', False):
                logger.info("Overriding baseline GA config with Proxy GA config from optuna_config.py")
                proxy_config = getattr(config, 'proxy_ga_config', {})
                for key, value in proxy_config.items():
                    self.json_ga_config[key] = value
                    
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
        
        # --- CRITICAL MEMORY OPTIMIZATION ---
        # Once features are precomputed, we NO LONGER need the raw images.
        # We clear them NOW before starting the probe extraction to save RAM/VRAM.
        logger.info("Features precomputed. Clearing raw training images to free up RAM...")
        self.train_images = None
        if hasattr(self, 'genetic_optimizer'):
            self.genetic_optimizer.images = None
        import gc
        gc.collect()
        # ------------------------------------
        
        # --- CRITICAL: PRECOMPUTE PROBE FEATURES ---
        # We precompute probe features NOW to avoid OOM crashes later.
        # This ensures heavy extraction happens when GPU memory is fresh.
        logger.info("Precomputing probe features for HPO validation...")
        self.genetic_optimizer.precompute_probe_features()


        
        # Store best results
        self.best_fitness = -float('inf')
        self.best_individual = None
        self.best_feature_weights = None
        self.best_ga_config = None
        self.fw_study = None
        self.ga_study = None
        
        # Set random seed if specified — must be passed to the sampler, not created and discarded
        self._sampler = (
            optuna.samplers.TPESampler(seed=config.optimization_seed)
            if config.optimization_seed is not None
            else optuna.samplers.TPESampler()
        )

    def _get_val_sample(self):
        """
        Lazily load and cache the cross-generator validation sample.

        Sourced exclusively from ``global_config.val_generators`` / ``val`` split,
        respecting ``global_config.max_val_per_gen`` per class per generator.
        This data is NEVER touched during optimisation.
        """
        if self._val_images is None:
            # Sourced from val_generators. 
            # 5000 images is robust for cross-generator validation.
            self._val_images, self._val_labels = \
                self.data_loader.create_val_sample(sample_size=global_config.probe_sample_size)
            logger.info(
                f"Validation sample ready: {len(self._val_images)} images "
                f"from {global_config.val_generators}"
            )
        return self._val_images, self._val_labels

    
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
            
            if 'inactive_weight_penalty' in ga_params:
                import tensorflow as tf
                self.genetic_optimizer.inactive_penalty_tf = tf.constant(
                    float(ga_params['inactive_weight_penalty']), dtype=tf.float32
                )

            if 'target_sparsity' in ga_params:
                import tensorflow as tf
                self.genetic_optimizer.target_sparsity_tf = tf.constant(
                    float(ga_params['target_sparsity']), dtype=tf.float32
                )
            
            if 'sparsity_radius' in ga_params:
                import tensorflow as tf
                self.genetic_optimizer.sparsity_radius_tf = tf.constant(
                    float(ga_params['sparsity_radius']), dtype=tf.float32
                )
            
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
        
        ga_config['max_possible_rules'] = int(config.max_possible_rules_range[1])
        
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
        
        if getattr(config, 'optimize_weight_penalty', False):
            ga_config['inactive_weight_penalty'] = float(trial.suggest_float(
                'inactive_weight_penalty',
                float(config.inactive_weight_penalty_range[0]),
                float(config.inactive_weight_penalty_range[1])
            ))
        else:
            ga_config['inactive_weight_penalty'] = float(getattr(config, 'inactive_weight_penalty', 0.0))
        
        # Fixed parameters
        ga_config['random_seed'] = config.optimization_seed
        ga_config['verbose'] = False  # Keep quiet during optimization
        
        # Ensure max_possible_rules >= rules_per_individual
        if ga_config['max_possible_rules'] < ga_config['rules_per_individual']:
            ga_config['max_possible_rules'] = ga_config['rules_per_individual']
        
        # Mask coverage (Sparsity) targets
        ga_config['target_sparsity'] = float(trial.suggest_float(
            'target_sparsity',
            float(config.target_sparsity_range[0]),
            float(config.target_sparsity_range[1])
        ))
        
        ga_config['sparsity_radius'] = float(trial.suggest_float(
            'sparsity_radius',
            float(config.sparsity_radius_range[0]),
            float(config.sparsity_radius_range[1])
        ))
        
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
            
            # --- BLENDED PRUNING CALLBACK ---
            def pruning_callback(gen_num, best_ind):
                if not config.enable_pruning or gen_num < config.pruning_warmup_steps:
                    return
                if gen_num % config.pruning_interval != 0:
                    return
                
                # Evaluate on probe and blend
                # Evaluate with penalty to ensure pruning respects the "Honest Weights" policy
                t_fit, _, _, _, _, _, _, _ = self.genetic_optimizer.get_fitness_breakdown(best_ind)
                # p_fit (probe) - use penalized probe for pruning in Phase 1
                p_fit = self.genetic_optimizer.eval_on_probe(best_ind, penalized=True)
                blended = 0.7 * t_fit + 0.3 * p_fit
                
                trial.report(blended, gen_num)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            # -------------------------------

            # Clear history to avoid memory accumulation across trials
            self.genetic_optimizer.all_run_histories.clear()
            
            num_runs = getattr(config, 'num_ga_runs_per_trial', 1)
            use_seeding = getattr(config, 'deterministic_trial_seeding', False)
            base_seed = getattr(config, 'optimization_seed', 42)
            if base_seed is None:
                base_seed = 42
                
            run_fitnesses = []
            best_ind_overall = None
            best_fit_overall = -1.0
            
            for run_idx in range(num_runs):
                if use_seeding:
                    run_seed = base_seed + trial.number * 100 + run_idx
                    if hasattr(self.genetic_optimizer, 'set_random_seed'):
                        self.genetic_optimizer.set_random_seed(run_seed)
                        
                # Run GA with updated configuration
                results = self.genetic_optimizer.run(
                    run_id=f"fw_trial_{trial.number}_run_{run_idx}",
                    on_generation_callback=pruning_callback if run_idx == 0 else None
                )
                
                train_fitness = results['best_fitness']
                best_ind = results['best_individual']
                
                # Penalize probe as well to maintain consistency in Phase 1
                probe_fitness = self.genetic_optimizer.eval_on_probe(best_ind, penalized=True)
                
                run_fit = 0.7 * train_fitness + 0.3 * probe_fitness
                run_fitnesses.append(run_fit)
                
                if config.verbosity >= 1:
                    logger.info(
                        f"Trial {trial.number} Run {run_idx}: train={train_fitness:.4f}  "
                        f"probe={probe_fitness:.4f}  blended={run_fit:.4f}"
                    )
                
                if run_fit > best_fit_overall:
                    best_fit_overall = run_fit
                    best_ind_overall = best_ind
                    
                # FAST FAIL MECHANISM (unpenalized probe evaluation)
                if run_idx == 0 and num_runs > 1:
                    if probe_fitness < config.min_fitness_threshold:
                        if config.verbosity >= 1:
                            logger.info(f"Trial {trial.number}: Fast failing after Run 0 (probe {probe_fitness:.4f} < {config.min_fitness_threshold})")
                        raise optuna.TrialPruned()
                    
                    if self.best_fitness > 0 and run_fit < self.best_fitness * 0.85:
                        if config.verbosity >= 1:
                            logger.info(f"Trial {trial.number}: Fast failing after Run 0 (probe {probe_fitness:.4f} << best {self.best_fitness:.4f})")
                        raise optuna.TrialPruned()
            
            fitness = float(np.mean(run_fitnesses))
            best_ind = best_ind_overall
            # -----------------------------------------------------------
            
            # Early stopping for very poor performance
            if fitness < config.min_fitness_threshold:
                raise optuna.TrialPruned()
            
            # Sanitize fitness to avoid SQLite issues with NaN/Inf
            fitness = float(np.nan_to_num(fitness, nan=-1.0, posinf=-1.0, neginf=-1.0))
            
            if config.verbosity >= 1:
                logger.info(f"Trial {trial.number}: Averaged Fitness = {fitness:.4f}")
            
            # Update internal best for tracking
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_feature_weights = feature_weights
                self.best_individual = best_ind

            # Force garbage collection to free memory
            import gc
            del results
            gc.collect()
            
            # Clear TensorFlow session to free GPU/RAM
            import tensorflow as tf
            tf.keras.backend.clear_session()

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
            
            # --- BLENDED PRUNING CALLBACK ---
            def pruning_callback(gen_num, best_ind):
                if not config.enable_pruning or gen_num < config.pruning_warmup_steps:
                    return
                if gen_num % config.pruning_interval != 0:
                    return
                
                # Evaluate on probe and blend (Unpenalized for Phase 2)
                t_fit = self.genetic_optimizer.get_unpenalized_fitness(best_ind)
                p_fit = self.genetic_optimizer.eval_on_probe(best_ind, penalized=False)
                
                if getattr(config, 'optimize_weight_penalty', False):
                    blended = p_fit
                else:
                    blended = 0.7 * t_fit + 0.3 * p_fit
                
                trial.report(blended, gen_num)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            # -------------------------------

            # Clear history to avoid memory accumulation across trials
            self.genetic_optimizer.all_run_histories.clear()
            
            num_runs = getattr(config, 'num_ga_runs_per_trial', 1)
            use_seeding = getattr(config, 'deterministic_trial_seeding', False)
            base_seed = getattr(config, 'optimization_seed', 42)
            if base_seed is None:
                base_seed = 42
                
            run_fitnesses = []
            best_ind_overall = None
            best_fit_overall = -1.0
            
            for run_idx in range(num_runs):
                if use_seeding:
                    run_seed = base_seed + trial.number * 100 + run_idx
                    if hasattr(self.genetic_optimizer, 'set_random_seed'):
                        self.genetic_optimizer.set_random_seed(run_seed)
                        
                # Run GA with updated configuration
                results = self.genetic_optimizer.run(
                    run_id=f"ga_trial_{trial.number}_run_{run_idx}",
                    on_generation_callback=pruning_callback if run_idx == 0 else None
                )
                
                train_fitness = results['best_true_fitness']
                best_ind = results['best_individual']
                # Maximize raw signal on probe in Phase 2
                probe_fitness = self.genetic_optimizer.eval_on_probe(best_ind, penalized=False)
                
                # If optimizing the inactive weight penalty, use pure unpenalized target to avoid zero collapse
                if getattr(config, 'optimize_weight_penalty', False):
                    run_fit = probe_fitness
                else:
                    run_fit = 0.7 * train_fitness + 0.3 * probe_fitness
                    
                run_fitnesses.append(run_fit)
                
                if config.verbosity >= 1:
                    logger.info(
                        f"Trial {trial.number} Run {run_idx}: train={train_fitness:.4f}  "
                        f"probe={probe_fitness:.4f}  target={run_fit:.4f}"
                    )
                                
                if run_fit > best_fit_overall:
                    best_fit_overall = run_fit
                    best_ind_overall = best_ind
                    
                # FAST FAIL MECHANISM (unpenalized probe evaluation)
                if run_idx == 0 and num_runs > 1:
                    if probe_fitness < config.min_fitness_threshold:
                        if config.verbosity >= 1:
                            logger.info(f"Trial {trial.number}: Fast failing after Run 0 (probe {probe_fitness:.4f} < {config.min_fitness_threshold})")
                        raise optuna.TrialPruned()
                    
                    if self.best_fitness > 0 and run_fit < self.best_fitness * 0.85:
                        if config.verbosity >= 1:
                            logger.info(f"Trial {trial.number}: Fast failing after Run 0 (probe {probe_fitness:.4f} << best {self.best_fitness:.4f})")
                        raise optuna.TrialPruned()
            
            fitness = float(np.mean(run_fitnesses))
            best_ind = best_ind_overall
            
            # Calculate compute effort penalty to discourage bloated runs
            pop_size = ga_params.get('population_size', 100)
            n_gens = ga_params.get('n_generations', 100)
            compute_effort = pop_size * n_gens
            compute_penalty = compute_effort * 1e-6  # 0.01 penalty per 10,000 extra evaluations
            
            final_score = fitness - compute_penalty
            
            # Early stopping for very poor performance
            if final_score < config.min_fitness_threshold:
                raise optuna.TrialPruned()
            
            # Sanitize fitness to avoid SQLite issues with NaN/Inf
            final_score = float(np.nan_to_num(final_score, nan=-1.0, posinf=-1.0, neginf=-1.0))
            
            if config.verbosity >= 1:
                logger.info(f"Trial {trial.number}: Raw Fit={fitness:.4f}, Effort={compute_effort}, Penalty=-{compute_penalty:.4f} -> Final={final_score:.4f}")
            
            # Update internal best
            if final_score > self.best_fitness:
                self.best_fitness = final_score
                self.best_ga_config = ga_params
                self.best_individual = best_ind

            # Force garbage collection to free memory
            import gc
            del results
            gc.collect()
            
            # Clear TensorFlow session to free GPU/RAM
            import tensorflow as tf
            tf.keras.backend.clear_session()

            return final_score
            
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
        
        # Use JournalStorage instead of RDBStorage to avoid SQLite locking issues on Windows/WSL
        storage = JournalStorage(JournalFileStorage("optuna_journal.log"))
        
        self.fw_study = optuna.create_study(
            direction='maximize',
            study_name=config.feature_weight_study_name,
            storage=storage,
            load_if_exists=True,
            sampler=self._sampler,
            pruner=optuna.pruners.MedianPruner() if config.enable_pruning else None
        )
        study = self.fw_study
        
        # Optimize with a loop to release file locks between trials
        from tqdm import tqdm
        start_time = time.time()
        for _ in tqdm(range(config.feature_weight_trials), desc="Optimizing Weights"):
            study.optimize(
                self.objective_feature_weights, 
                n_trials=1,
                show_progress_bar=False
            )
        end_time = time.time()
        
        # Reconstruct best weights from stored trial params (deterministic — no re-sampling)
        best_trial = study.best_trial
        raw_weights = {feat: best_trial.params[f"weight_{feat}"]
                       for feat in config.feature_weight_ranges}
        best_weights = self.normalize_weights(raw_weights)
        
        logger.info(f"Feature weight optimization completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Best fitness: {best_trial.value:.4f}")
        logger.info(f"Best feature weights: {best_weights}")
        
        # Save results with fitness
        self.best_feature_weights = best_weights
        output_weights = best_weights.copy()
        output_weights['__fitness__'] = best_trial.value
        self.save_json(output_weights, config.feature_weights_output_file)
        
        # 3. Validation on CROSS-GENERATOR held-out set
        if self.best_individual is not None:
            logger.info("\n" + "="*60)
            logger.info("FINAL VALIDATION (PHASE 1)")
            logger.info(f"  Source: val_generators = {global_config.val_generators}")
            logger.info(f"  Split : val (never seen during optimisation)")
            logger.info("="*60)
            val_images, val_labels = self._get_val_sample()
            val_fitness = self.genetic_optimizer.validate(
                val_images, val_labels, self.best_individual
            )
            logger.info(f"Phase 1 Train Fitness (in-dist):  {best_trial.value:.4f}")
            logger.info(f"Phase 1 Val Fitness  (cross-gen): {val_fitness:.4f}")
            gap = best_trial.value - val_fitness
            logger.info(f"Generalisation gap: {gap:+.4f} ({'overfitting' if gap > 0.05 else 'good'})")
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
        
        # Use JournalStorage instead of RDBStorage to avoid SQLite locking issues on Windows/WSL
        storage = JournalStorage(JournalFileStorage("optuna_journal.log"))
        
        self.ga_study = optuna.create_study(
            direction='maximize',
            study_name=config.ga_config_study_name,
            storage=storage,
            load_if_exists=True,
            sampler=self._sampler,
            pruner=optuna.pruners.MedianPruner() if config.enable_pruning else None
        )
        study = self.ga_study
        
        # Optimize with a loop to release file locks between trials
        from tqdm import tqdm
        start_time = time.time()
        for _ in tqdm(range(config.ga_config_trials), desc="Optimizing GA Config"):
            study.optimize(
                self.objective_ga_config, 
                n_trials=1,
                show_progress_bar=False
            )
        end_time = time.time()
        
        # Reconstruct best GA config from stored trial params
        best_trial = study.best_trial
        best_ga_config = {
            'population_size':      int(best_trial.params['population_size']),
            'n_generations':        int(best_trial.params['n_generations']),
            'rules_per_individual': int(best_trial.params['rules_per_individual']),
            'max_possible_rules':   int(best_trial.params.get('max_possible_rules', config.max_possible_rules_range[1])),
            'crossover_prob':       float(best_trial.params['crossover_prob']),
            'mutation_prob':        float(best_trial.params['mutation_prob']),
            'tournament_size':      int(best_trial.params['tournament_size']),
            'num_elites':           int(best_trial.params['num_elites']),
            'inactive_weight_penalty': float(best_trial.params.get('inactive_weight_penalty', getattr(config, 'inactive_weight_penalty', 0.0))),
        }
        
        # Ensure max_possible_rules >= rules_per_individual
        best_ga_config['max_possible_rules'] = max(
            best_ga_config['max_possible_rules'], 
            best_ga_config['rules_per_individual']
        )
        
        logger.info(f"GA config optimization completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Best fitness: {best_trial.value:.4f}")
        logger.info(f"Best GA config: {best_ga_config}")
        
        # Save results with fitness
        self.best_ga_config = best_ga_config
        self.best_fitness = best_trial.value
        output_config = best_ga_config.copy()
        output_config['fitness'] = best_trial.value
        self.save_json(output_config, config.ga_config_output_file)
        
        # 3. Validation on CROSS-GENERATOR held-out set
        if self.best_individual is not None:
            logger.info("\n" + "="*60)
            logger.info("FINAL VALIDATION (PHASE 2)")
            logger.info(f"  Source: val_generators = {global_config.val_generators}")
            logger.info(f"  Split : val (never seen during optimisation)")
            logger.info("="*60)
            val_images, val_labels = self._get_val_sample()
            val_fitness = self.genetic_optimizer.validate(
                val_images, val_labels, self.best_individual
            )
            logger.info(f"Phase 2 Train Fitness (in-dist):  {best_trial.value:.4f}")
            logger.info(f"Phase 2 Val Fitness  (cross-gen): {val_fitness:.4f}")
            gap = best_trial.value - val_fitness
            logger.info(f"Generalisation gap: {gap:+.4f} ({'overfitting' if gap > 0.05 else 'good'})")
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
    
    
    def run_optimization(self, phase: str = 'all') -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Run the optimization process.
        
        Args:
            phase: 'all', 'weights', or 'ga'
        
        Returns:
            Tuple of (best_feature_weights, best_ga_config)
        """
        logger.info(f"Starting hyperparameter optimization (phase: {phase})...")
        
        trials_str = ""
        if phase in ['all', 'weights']:
            trials_str += f"Weights: {config.feature_weight_trials} "
        if phase in ['all', 'ga']:
            trials_str += f"GA: {config.ga_config_trials}"
        logger.info(f"Total trials to run: {trials_str}")
        
        start_time = time.time()
        
        best_weights = None
        best_ga_config = None
        
        try:
            # Phase 1: Optimize feature weights
            if phase in ['all', 'weights']:
                best_weights = self.optimize_feature_weights()
            
            # Phase 2: Optimize GA configuration
            if phase in ['all', 'ga']:
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
    parser = argparse.ArgumentParser(description="Run hyperparameter optimization for the genetic algorithm.")
    parser.add_argument(
        '--phase', 
        type=str, 
        choices=['all', 'weights', 'ga'], 
        default='all',
        help="Optimization phase to run: 'weights' (Phase 1), 'ga' (Phase 2), or 'all' (both)."
    )
    args = parser.parse_args()
    
    try:
        optimizer = HyperparameterOptimizer()
        best_weights, best_ga_config = optimizer.run_optimization(phase=args.phase)
        
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        if best_weights is not None:
            print(f"Best Feature Weights: {best_weights}")
        if best_ga_config is not None:
            print(f"Best GA Configuration: {best_ga_config}")
        print(f"Best Fitness Achieved: {optimizer.best_fitness:.4f}")
        
    except KeyboardInterrupt:
        print("\nOptimization stopped by user.")
    except Exception as e:
        print(f"\nOptimization failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()