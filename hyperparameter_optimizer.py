import optuna # type: ignore
from optuna.storages import JournalStorage, JournalFileStorage
import os
import json
import logging
import argparse
import time
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Any, Tuple

import tensorflow as tf
# Setup GPU memory growth before any other module imports tensorflow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        pass

# Import your project modules
from data_loader import DataLoader
from genetic_algorithm import GeneticFeatureOptimizer
import optuna_config as config
import global_config

def _run_single_trial_subprocess(study_name, mask_mode, genetic_rules_np, random_mask_sparsity):
    """
    Subprocess entrypoint to run a single HPO trial.
    Spawning a new process for each trial guarantees that TensorFlow completely
    releases all GPU memory back to the OS when the trial completes, preventing OOM leaks.
    """
    import optuna
    from optuna.storages import JournalStorage, JournalFileStorage
    import tensorflow as tf
    import gc
    
    # 1. Enable memory growth in the subprocess
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    # 2. Initialize optimizer and load datasets
    from hyperparameter_optimizer import HyperparameterOptimizer
    optimizer = HyperparameterOptimizer()
    optimizer._init_cnn_datasets(mask_mode)
    
    # 3. Convert rules back to tensor if needed
    genetic_rules = None
    if genetic_rules_np is not None:
        genetic_rules = tf.convert_to_tensor(genetic_rules_np, dtype=tf.float32)
        
    # 4. Load the study with a dynamic sampler seed to prevent duplicate trials in process isolation
    import optuna_config as config
    storage = JournalStorage(JournalFileStorage("optuna_journal.log"))
    
    # Load first with a temporary sampler to count completed trials
    temp_study = optuna.load_study(study_name=study_name, storage=storage)
    completed_trials = len([t for t in temp_study.trials if t.state.is_finished()])
    
    # Dynamically adjust the seed based on the number of completed trials
    base_seed = getattr(config, 'optimization_seed', 42)
    sampler_seed = (base_seed + completed_trials) if base_seed is not None else None
    
    study = optuna.load_study(
        study_name=study_name,
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=sampler_seed)
    )
    
    # 5. Define local objective
    objective = lambda t: optimizer.objective_cnn(
        trial=t,
        mask_mode=mask_mode,
        genetic_rules=genetic_rules,
        random_mask_sparsity=random_mask_sparsity
    )
    
    # 6. Run exactly 1 trial
    study.optimize(objective, n_trials=1, show_progress_bar=False)
    
    # 7. Clean up
    tf.keras.backend.clear_session()
    gc.collect()


def _run_cnn_prep_subprocess(mask_mode, json_ga_config, json_weights, output_queue):
    """
    Runs HPO CNN preparation (GA evolution and raw feature precomputation)
    in a spawned subprocess to prevent GPU VRAM leaks in the parent process.
    """
    import tensorflow as tf
    import gc
    import logging
    from model_architecture import ModelWrapper
    
    # 1. Enable memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    # 2. Initialize optimizer and load datasets
    from hyperparameter_optimizer import HyperparameterOptimizer
    optimizer = HyperparameterOptimizer()
    optimizer.json_ga_config = json_ga_config
    optimizer.json_weights = json_weights
    
    sample_images, sample_labels = optimizer._init_cnn_datasets(mask_mode)
    
    genetic_rules = None
    genetic_rules_np = None
    random_mask_sparsity = None
    
    if mask_mode == 'ga':
        from genetic_algorithm import GeneticFeatureOptimizer
        base_config = optimizer.create_base_config()
        optimizer.genetic_optimizer = GeneticFeatureOptimizer(
            images=sample_images,
            labels=sample_labels,
            config=base_config
        )
        optimizer.update_optimizer_config(ga_params=json_ga_config)
        
        import global_config
        num_prep_runs = getattr(global_config, 'num_ga_prep_runs', 3)
        best_overall_ruleset = None
        best_overall_fitness = -float('inf')
        
        sub_logger = logging.getLogger(__name__)
        sub_logger.info(f"Running {num_prep_runs} GA runs to find the best ruleset for Phase 3 CNN prep...")
        for i in range(num_prep_runs):
            run_seed = (global_config.random_seed or 42) + i * 100
            optimizer.genetic_optimizer.set_random_seed(run_seed)
            ga_results = optimizer.genetic_optimizer.run(run_id=f"cnn_hpo_ga_prep_run_{i}")
            
            fitness = ga_results['best_fitness']
            if fitness > best_overall_fitness:
                best_overall_fitness = fitness
                best_overall_ruleset = ga_results['best_individual'].rules_tensor.numpy()
                
        genetic_rules_np = best_overall_ruleset
        genetic_rules = tf.convert_to_tensor(genetic_rules_np, dtype=tf.float32)
    elif mask_mode == 'random':
        random_mask_sparsity = json_ga_config.get('target_sparsity', 0.45)
        
    # 4. Precompute raw features once (ModelWrapper uses features)
    model_wrapper = ModelWrapper(
        genetic_rules=genetic_rules,
        mask_mode=mask_mode,
        random_mask_sparsity=random_mask_sparsity
    )
    
    sub_logger = logging.getLogger(__name__)
    sub_logger.info(f"Precomputing raw features inside CNN prep subprocess for mode '{mask_mode}'...")
    model_wrapper.precompute_features(optimizer.train_ds, "train")
    model_wrapper.precompute_features(optimizer.test_ds, "val")
    model_wrapper.precompute_features(optimizer.test_ds, "test")
    
    # Put results into queue
    output_queue.put((genetic_rules_np, random_mask_sparsity))
    
    # Clean up
    tf.keras.backend.clear_session()
    gc.collect()


def _run_single_evaluation_subprocess(mask_mode, params, genetic_rules_np, random_mask_sparsity, output_queue):
    """
    Runs a single training/evaluation run for a given mask mode and set of hyperparameters,
    and returns the validation accuracy. Runs in a subprocess to isolate GPU memory.
    """
    import tensorflow as tf
    import gc
    import os
    import numpy as np
    import global_config
    import optuna_config as config
    from model_architecture import ModelWrapper
    
    # 1. Enable memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    # 2. Initialize optimizer and load datasets
    from hyperparameter_optimizer import HyperparameterOptimizer
    optimizer = HyperparameterOptimizer()
    optimizer._init_cnn_datasets(mask_mode)
    
    # 3. Setup global config dynamically
    for key, val in params.items():
        setattr(global_config, f"cnn_{key}", val)
        
    global_config.epochs = getattr(config, 'cnn_epochs', 15)
    global_config.use_feature_cache = mask_mode in ('ga', 'random')
    
    # Run seed
    run_seed = getattr(config, 'optimization_seed', 42)
    if run_seed is None:
        run_seed = 42
    global_config.random_seed = run_seed
    np.random.seed(run_seed)
    tf.random.set_seed(run_seed)
    
    genetic_rules = None
    if genetic_rules_np is not None:
        genetic_rules = tf.convert_to_tensor(genetic_rules_np, dtype=tf.float32)
        
    model_wrapper = ModelWrapper(
        genetic_rules=genetic_rules,
        mask_mode=mask_mode,
        random_mask_sparsity=random_mask_sparsity
    )
    
    temp_model_path = f"temp_eval_{mask_mode}.keras"
    try:
        model_wrapper.train(
            train_dataset=optimizer.train_ds,
            validation_dataset=optimizer.test_ds,
            model_path=temp_model_path,
            precompute_features=model_wrapper.use_features
        )
        if os.path.exists(temp_model_path):
            model_wrapper.model.load(temp_model_path)
            
        eval_results = model_wrapper.evaluate(optimizer.test_ds)
        val_acc = float(eval_results[1])
        output_queue.put(val_acc)
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Single evaluation failed: {e}")
        output_queue.put(0.0)
    finally:
        if os.path.exists(temp_model_path):
            try:
                os.remove(temp_model_path)
            except:
                pass
        tf.keras.backend.clear_session()
        gc.collect()


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
        """Initialize the optimizer with data loading configuration and weights."""
        logger.info("Initializing HyperparameterOptimizer...")
        
        # Seed all random number generators at the absolute top of initialization
        base_seed = getattr(config, 'optimization_seed', 42)
        if base_seed is not None:
            import random
            import numpy as np
            import tensorflow as tf
            random.seed(base_seed)
            np.random.seed(base_seed)
            tf.random.set_seed(base_seed)
            global_config.random_seed = base_seed
            logger.info(f"Global random, numpy, tensorflow, and global_config seeds set to {base_seed}")
            
        self.data_loader = DataLoader()
        
        # Dataset placeholders — loaded lazily depending on HPO phase
        self.train_ds = None
        self.test_ds = None
        self.train_images = None
        self.train_labels = None
        
        # Placeholder — loaded on demand from global_config.val_generators
        self._val_images = None
        self._val_labels = None
        
        # 1. Load Strict Source-of-Truth JSONs (Optimization Baseline)
        try:
            with open(config.feature_weights_output_file, 'r') as f:
                raw_weights = json.load(f)
                # Filter out metadata fields like '__fitness__'
                self.json_weights = {k: v for k, v in raw_weights.items() if not k.startswith("__")}
            with open(config.ga_config_output_file, 'r') as f:
                self.json_ga_config = json.load(f)
                
            # Probe GA config overrides are now applied on-the-fly in Phase 1 (objective_feature_weights) 
            # to prevent polluting the base configuration for Phase 2.
                    
        except Exception as e:
            logger.error(f"FATAL: Could not load baseline configuration from JSON: {e}")
            logger.error(f"Please ensure {config.feature_weights_output_file} and {config.ga_config_output_file} exist.")
            sys.exit(1)

        # Genetic Feature Optimizer placeholder — loaded lazily
        self.genetic_optimizer = None
        
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

    def _init_ga_optimizer(self):
        """Lazily initialize the GeneticFeatureOptimizer and precompute feature arrays for GA optimization (Phase 1 & 2)."""
        if self.genetic_optimizer is not None:
            return
            
        logger.info("Initializing GA Datasets and GeneticFeatureOptimizer for Phase 1/2...")
        # Load datasets and GA sample
        self.train_ds, self.test_ds, all_images, all_labels = self.data_loader.create_datasets(create_sample=True)
        self.train_images = all_images
        self.train_labels = all_labels
        
        logger.info(
            f"Training sample: {len(self.train_images)} images "
            f"from train_generators={global_config.train_generators}"
        )
        logger.info(
            f"Validation will use val_generators={global_config.val_generators}"
        )
        
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
        logger.info("Features precomputed. Clearing raw training images to free up RAM...")
        self.train_images = None
        if hasattr(self, 'genetic_optimizer'):
            self.genetic_optimizer.images = None
        import gc
        gc.collect()
        # ------------------------------------
        
        # --- CRITICAL: PRECOMPUTE PROBE FEATURES ---
        logger.info("Precomputing probe features for HPO validation...")
        self.genetic_optimizer.precompute_probe_features()

    def _init_cnn_datasets(self, mask_mode: str):
        """Initialize and configure datasets specifically for CNN optimization (Phase 3)."""
        logger.info("Initializing CNN Datasets for Phase 3...")
        
        # Apply temporary limits for HPO
        global_config.max_train_samples = getattr(config, 'cnn_max_train_samples', 3000)
        global_config.max_val_samples = getattr(config, 'cnn_max_val_samples', 1000)
        global_config.mask_mode = mask_mode
        global_config.feature_weights = self.json_weights
        
        logger.info(f"Re-loading data with limits: max_train={global_config.max_train_samples}, max_val={global_config.max_val_samples}")
        # Only create the sample dataset if we actually need it (i.e. mask_mode is 'ga')
        self.train_ds, self.test_ds, sample_images, sample_labels = self.data_loader.create_datasets(
            create_sample=(mask_mode == 'ga')
        )
        return sample_images, sample_labels

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
            # Explicitly order the weights array to match self.genetic_optimizer.feature_names
            ordered_weights = [feature_weights[name] for name in self.genetic_optimizer.feature_names]
            self.genetic_optimizer.feature_weights = tf.convert_to_tensor(
                ordered_weights, dtype=tf.float32
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
        # Ensure we iterate in the exact order of the GA optimizer's feature_names to prevent index mismatches!
        feature_order = self.genetic_optimizer.feature_names
        
        for feature in feature_order:
            if feature in config.feature_weight_ranges:
                min_val, max_val = config.feature_weight_ranges[feature]
            else:
                logger.warning(f"Feature '{feature}' not found in feature_weight_ranges. Using default (0.0, 1.0).")
                min_val, max_val = 0.0, 1.0
                
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

        When ``config.sparsity_only_phase2`` is True, all structural GA params are pinned
        to the current best_ga_config.json values and only target_sparsity and
        sparsity_radius are optimised.  This is used after the soft-mask change where the
        semantic meaning of target_sparsity changed but the rest of the GA search space
        remains valid.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of suggested GA parameters
        """
        sparsity_only = getattr(config, 'sparsity_only_phase2', False)

        if sparsity_only:
            # --- Sparsity-only mode: pin everything from best known config ---
            ga_config = {k: v for k, v in self.json_ga_config.items()
                         if k not in ('fitness',)}
            # Ensure required numeric types are preserved for pinned params
            for int_key in ('population_size', 'n_generations', 'rules_per_individual',
                            'max_possible_rules'):
                if int_key in ga_config:
                    ga_config[int_key] = int(ga_config[int_key])
            for float_key in ('crossover_prob', 'mutation_prob',
                              'inactive_weight_penalty'):
                if float_key in ga_config:
                    ga_config[float_key] = float(ga_config[float_key])

            # Search: sparsity targets + selection pressure params
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
            # Keep fixed overrides
            ga_config['random_seed'] = config.optimization_seed
            ga_config['verbose'] = False
            return ga_config

        # --- Full Phase 2 mode (original behaviour) ---
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
            
            # Use baseline GA config, overridden by Probe config if requested
            ga_config = self.json_ga_config.copy()
            if getattr(config, 'use_probe_ga_config', False):
                probe_config = getattr(config, 'probe_ga_config', {})
                for key, value in probe_config.items():
                    ga_config[key] = value

            # Update the existing optimizer with new feature weights
            self.update_optimizer_config(
                feature_weights=feature_weights,
                ga_params=ga_config
            )
            
            if config.verbosity >= 2:
                logger.info(f"Trial {trial.number}: Testing feature weights: {feature_weights}")
            
            # --- BLENDED PRUNING CALLBACK ---
            best_blended_so_far = [-float('inf')]

            def pruning_callback(gen_num, best_ind):
                # Evaluate on probe and blend
                # Evaluate with penalty to ensure pruning respects the "Honest Weights" policy
                t_fit, _, _, _, _, _, _, _ = self.genetic_optimizer.get_fitness_breakdown(best_ind)
                # p_fit (probe) - use penalized probe for pruning in Phase 1
                p_fit = self.genetic_optimizer.eval_on_probe(best_ind, penalized=True)
                blended = 0.3 * t_fit + 0.7 * p_fit
                
                if blended > best_blended_so_far[0]:
                    best_blended_so_far[0] = blended
                
                # Check for pruning at exactly generation 40 and 80
                if config.enable_pruning:
                    if gen_num == 40:
                        trial.report(best_blended_so_far[0], step=40)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                    elif gen_num == 80:
                        trial.report(best_blended_so_far[0], step=80)
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
                
                run_fit = 0.3 * train_fitness + 0.7 * probe_fitness
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
                    
                    # Compute average fitness of all completed trials so far as the baseline reference
                    completed_values = [t.value for t in trial.study.trials if t.value is not None]
                    reference_fitness = float(np.mean(completed_values)) if completed_values else 0.0
                    
                    if reference_fitness > 0 and run_fit < reference_fitness * getattr(config, 'fast_fail_tolerance', 0.85):
                        if config.verbosity >= 1:
                            logger.info(f"Trial {trial.number}: Fast failing after Run 0 (fit {run_fit:.4f} << avg {reference_fitness:.4f})")
                        raise optuna.TrialPruned()
            
            # Prioritize the best run's fitness (max) instead of the mean to match the training behavior
            fitness = float(best_fit_overall)
            best_ind = best_ind_overall
            # -----------------------------------------------------------
            
            # Early stopping for very poor performance
            if fitness < config.min_fitness_threshold:
                raise optuna.TrialPruned()
            
            # Sanitize fitness to avoid SQLite issues with NaN/Inf
            fitness = float(np.nan_to_num(fitness, nan=-1.0, posinf=-1.0, neginf=-1.0))
            
            if config.verbosity >= 1:
                logger.info(f"Trial {trial.number}: Best Run Fitness = {fitness:.4f}")
            
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
            best_blended_so_far = [-float('inf')]

            def pruning_callback(gen_num, best_ind):
                # Evaluate on probe and blend (Unpenalized for Phase 2)
                t_fit = self.genetic_optimizer.get_unpenalized_fitness(best_ind)
                p_fit = self.genetic_optimizer.eval_on_probe(best_ind, penalized=False)
                
                if getattr(config, 'optimize_weight_penalty', False):
                    blended = p_fit
                else:
                    blended = 0.3 * t_fit + 0.7 * p_fit
                
                if blended > best_blended_so_far[0]:
                    best_blended_so_far[0] = blended
                
                # Check for pruning at exactly generation 40 and 80
                if config.enable_pruning:
                    if gen_num == 40:
                        trial.report(best_blended_so_far[0], step=40)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                    elif gen_num == 80:
                        trial.report(best_blended_so_far[0], step=80)
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
                    run_fit = 0.3 * train_fitness + 0.7 * probe_fitness
                    
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
                    
                    # Compute average fitness of all completed trials so far as the baseline reference
                    completed_values = [t.value for t in trial.study.trials if t.value is not None]
                    reference_fitness = float(np.mean(completed_values)) if completed_values else 0.0
                    
                    if reference_fitness > 0 and run_fit < reference_fitness * getattr(config, 'fast_fail_tolerance', 0.85):
                        if config.verbosity >= 1:
                            logger.info(f"Trial {trial.number}: Fast failing after Run 0 (fit {run_fit:.4f} << avg {reference_fitness:.4f})")
                        raise optuna.TrialPruned()
            
            # Prioritize the best run's fitness (max) instead of the mean to match the training behavior
            fitness = float(best_fit_overall)
            best_ind = best_ind_overall
            
            # Calculate compute effort penalty to discourage bloated runs
            pop_size = ga_params.get('population_size', 100)
            n_gens = ga_params.get('n_generations', 100)
            compute_effort = pop_size * n_gens
            penalty_coeff = getattr(config, 'compute_penalty_coefficient', 5e-7)
            compute_penalty = compute_effort * penalty_coeff  # Discourage bloated evaluations
            
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
        # Lazily load data and precompute features for Phase 1 GA
        self._init_ga_optimizer()
        
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
            pruner=optuna.pruners.PercentilePruner(percentile=config.prune_percentile) if config.enable_pruning else None
        )
        study = self.fw_study
        
        # Optimize with a loop to release file locks between trials
        from tqdm import tqdm
        start_time = time.time()
        completed_trials = len(study.trials)
        remaining_trials = config.feature_weight_trials - completed_trials
        
        if remaining_trials <= 0:
            logger.info(f"Feature weight optimization already has {completed_trials} trials (target: {config.feature_weight_trials}). Skipping optimization.")
        else:
            logger.info(f"Feature weight optimization starting with {completed_trials} existing trials. Running {remaining_trials} remaining trials.")
            for _ in tqdm(range(remaining_trials), desc="Optimizing Weights"):
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
        # Lazily load data and precompute features for Phase 2 GA
        self._init_ga_optimizer()
        
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
            pruner=optuna.pruners.PercentilePruner(percentile=config.prune_percentile) if config.enable_pruning else None
        )
        study = self.ga_study
        
        # Optimize with a loop to release file locks between trials
        from tqdm import tqdm
        start_time = time.time()
        completed_trials = len(study.trials)
        sparsity_only = getattr(config, 'sparsity_only_phase2', False)
        target_trials = (
            getattr(config, 'ga_config_sparsity_only_trials', 50)
            if sparsity_only else config.ga_config_trials
        )
        remaining_trials = target_trials - completed_trials

        if remaining_trials <= 0:
            logger.info(
                f"GA config optimization already has {completed_trials} trials "
                f"(target: {target_trials}{'  [sparsity-only]' if sparsity_only else ''}). "
                "Skipping optimization."
            )
        else:
            mode_tag = '  [sparsity-only: target_sparsity + sparsity_radius only]' if sparsity_only else ''
            logger.info(
                f"GA config optimization starting with {completed_trials} existing trials. "
                f"Running {remaining_trials} remaining trials.{mode_tag}"
            )
            for _ in tqdm(range(remaining_trials), desc="Optimizing GA Config"):
                study.optimize(
                    self.objective_ga_config, 
                    n_trials=1,
                    show_progress_bar=False
                )
        end_time = time.time()
        
        # Reconstruct best GA config from stored trial params.
        # In sparsity-only mode, only target_sparsity and sparsity_radius are trial params;
        # the rest were pinned from json_ga_config — merge them back in here.
        sparsity_only = getattr(config, 'sparsity_only_phase2', False)
        best_trial = study.best_trial

        if sparsity_only:
            best_ga_config = {k: v for k, v in self.json_ga_config.items()
                              if k not in ('fitness',)}
            for int_key in ('population_size', 'n_generations', 'rules_per_individual',
                            'max_possible_rules'):
                if int_key in best_ga_config:
                    best_ga_config[int_key] = int(best_ga_config[int_key])
            for float_key in ('crossover_prob', 'mutation_prob',
                              'inactive_weight_penalty'):
                if float_key in best_ga_config:
                    best_ga_config[float_key] = float(best_ga_config[float_key])
            best_ga_config['target_sparsity'] = float(best_trial.params['target_sparsity'])
            best_ga_config['sparsity_radius']  = float(best_trial.params['sparsity_radius'])
            best_ga_config['tournament_size']  = int(best_trial.params['tournament_size'])
            best_ga_config['num_elites']        = int(best_trial.params['num_elites'])
        else:
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
                'target_sparsity':      float(best_trial.params.get('target_sparsity', getattr(config, 'target_sparsity', 0.4))),
                'sparsity_radius':      float(best_trial.params.get('sparsity_radius', getattr(config, 'sparsity_radius', 0.2))),
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

    def suggest_cnn_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest CNN hyperparameters for optimization trial."""
        lr_min, lr_max = getattr(config, 'cnn_learning_rate_range', (1e-5, 1e-2))
        do1_min, do1_max = getattr(config, 'cnn_dropout_1_range', (0.0, 0.6))
        do2_min, do2_max = getattr(config, 'cnn_dropout_2_range', (0.0, 0.6))
        du_min, du_max = getattr(config, 'cnn_dense_units_range', (64, 512))
        l2_min, l2_max = getattr(config, 'cnn_l2_reg_range', (1e-6, 1e-2))
        optimizers = getattr(config, 'cnn_optimizers', ['adam', 'rmsprop', 'sgd'])
        
        params = {
            'learning_rate': float(trial.suggest_float('learning_rate', float(lr_min), float(lr_max), log=True)),
            'dropout_1': float(trial.suggest_float('dropout_1', float(do1_min), float(do1_max))),
            'dropout_2': float(trial.suggest_float('dropout_2', float(do2_min), float(do2_max))),
            'dense_units': int(trial.suggest_int('dense_units', int(du_min), int(du_max), step=64)),
            'l2_reg': float(trial.suggest_float('l2_reg', float(l2_min), float(l2_max), log=True)),
            'optimizer': str(trial.suggest_categorical('optimizer', optimizers))
        }
        return params

    def objective_cnn(self, trial: optuna.Trial, mask_mode: str, 
                      genetic_rules=None, random_mask_sparsity: float = None) -> float:
        """Objective function for optimizing CNN hyperparameters."""
        import tensorflow as tf
        import gc
        import os
        from model_architecture import ModelWrapper
        
        # 1. Suggest CNN parameters
        cnn_params = self.suggest_cnn_params(trial)
        
        # 2. Set them on global_config dynamically
        for key, value in cnn_params.items():
            setattr(global_config, f"cnn_{key}", value)
            
        # Also set the training settings for fast tuning
        original_epochs = global_config.epochs
        original_use_cache = global_config.use_feature_cache
        original_seed = global_config.random_seed
        
        global_config.epochs = getattr(config, 'cnn_epochs', 15)
        # Use feature cache for speed if use_features is True
        use_features = mask_mode in ('ga', 'random')
        global_config.use_feature_cache = use_features
        
        num_runs = getattr(config, 'num_cnn_runs_per_trial', 1)
        base_seed = getattr(config, 'optimization_seed', 42)
        if base_seed is None:
            base_seed = 42
            
        logger.info(f"Trial {trial.number}: Testing CNN params on mode '{mask_mode}' with {num_runs} runs: {cnn_params}")
        
        run_accuracies = []
        
        try:
            for run_idx in range(num_runs):
                import numpy as np
                run_seed = base_seed + trial.number * 100 + run_idx
                global_config.random_seed = run_seed
                np.random.seed(run_seed)
                tf.random.set_seed(run_seed)
                
                logger.info(f"  Trial {trial.number} Run {run_idx}/{num_runs} using seed {run_seed}")
                
                temp_model_path = os.path.join(global_config.output_dir, f"temp_cnn_{mask_mode}_trial_{trial.number}_run_{run_idx}.keras")
                
                try:
                    # 3. Create ModelWrapper
                    model_wrapper = ModelWrapper(
                        genetic_rules=genetic_rules,
                        mask_mode=mask_mode,
                        random_mask_sparsity=random_mask_sparsity
                    )
                    
                    # 4. Train the model
                    model_wrapper.train(
                        train_dataset=self.train_ds,
                        validation_dataset=self.test_ds,
                        model_path=temp_model_path,
                        precompute_features=use_features
                    )
                    
                    # Evaluate using best checkpoint weights
                    if os.path.exists(temp_model_path):
                        model_wrapper.model.load(temp_model_path)
                    
                    eval_results = model_wrapper.evaluate(self.test_ds)
                    val_acc = eval_results[1]
                    run_accuracies.append(val_acc)
                    
                    logger.info(f"  Trial {trial.number} Run {run_idx} finished. Val Accuracy: {val_acc:.4f}")
                    
                finally:
                    # Clean up temporary files for this run
                    if os.path.exists(temp_model_path):
                        try:
                            os.remove(temp_model_path)
                        except Exception as e:
                            logger.warning(f"Could not remove temp model file {temp_model_path}: {e}")
            
            avg_acc = float(np.mean(run_accuracies))
            logger.info(f"Trial {trial.number} finished. Averaged Validation Accuracy ({num_runs} runs): {avg_acc:.4f}")
            return avg_acc
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
            
        finally:
            # Clean up and reset config
            global_config.epochs = original_epochs
            global_config.use_feature_cache = original_use_cache
            global_config.random_seed = original_seed
            
            # Clear TensorFlow and garbage collect
            tf.keras.backend.clear_session()
            gc.collect()

    def optimize_cnn(self, mask_mode: str) -> Dict[str, Any]:
        """
        Optimize CNN hyperparameters for a specific mask mode using Optuna.
        """
        logger.info("="*60)
        logger.info(f"PHASE 3: Optimizing CNN Hyperparameters (Mode: {mask_mode.upper()})")
        logger.info("="*60)
        
        # Clear existing cache directories to avoid stale/stray files
        import shutil
        cache_dir = os.path.join(global_config.output_dir, global_config.feature_cache_dir)
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(cache_dir)
                logger.info(f"Cleared feature cache directory at {cache_dir} to start Phase 3 freshly.")
            except Exception as e:
                logger.warning(f"Could not clear feature cache directory: {e}")

        # Save original config
        orig_max_train = global_config.max_train_samples
        orig_max_val = global_config.max_val_samples
        orig_use_cache = global_config.use_feature_cache
        orig_mask_mode = global_config.mask_mode
        
        genetic_rules = None
        random_mask_sparsity = None
        
        # Setup specific properties based on mask mode
        if mask_mode in ('ga', 'random'):
            logger.info(f"Running HPO initialization and feature precomputation for '{mask_mode}' in a subprocess...")
            import multiprocessing
            ctx = multiprocessing.get_context('spawn')
            q = ctx.Queue()
            p = ctx.Process(
                target=_run_cnn_prep_subprocess,
                args=(mask_mode, self.json_ga_config, self.json_weights, q)
            )
            p.start()
            genetic_rules_np, random_mask_sparsity = q.get()
            p.join()
            
            import tensorflow as tf
            if genetic_rules_np is not None:
                genetic_rules = tf.convert_to_tensor(genetic_rules_np, dtype=tf.float32)
                
            self._init_cnn_datasets(mask_mode)
            
        else:
            logger.info("Setting up None mode for CNN HPO (no feature extraction)...")
            self._init_cnn_datasets(mask_mode)
            
        # Create Optuna study using JournalStorage to avoid locking
        study_name = config.cnn_study_name_template.format(mask_mode=mask_mode)
        storage = JournalStorage(JournalFileStorage("optuna_journal.log"))
        
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            sampler=self._sampler
        )
        
        # Optimize
        from tqdm import tqdm
        start_time = time.time()
        
        num_trials = getattr(config, 'cnn_trials', 20)
        completed_trials = len(study.trials)
        remaining_trials = num_trials - completed_trials
        
        if remaining_trials <= 0:
            logger.info(f"CNN optimization ({mask_mode}) already has {completed_trials} trials (target: {num_trials}). Skipping optimization.")
        else:
            logger.info(f"CNN optimization ({mask_mode}) starting with {completed_trials} existing trials. Running {remaining_trials} remaining trials.")
            
            import multiprocessing
            ctx = multiprocessing.get_context('spawn')
            genetic_rules_np = genetic_rules.numpy() if genetic_rules is not None else None
            
            for _ in tqdm(range(remaining_trials), desc=f"Optimizing CNN ({mask_mode})"):
                p = ctx.Process(
                    target=_run_single_trial_subprocess,
                    args=(study_name, mask_mode, genetic_rules_np, random_mask_sparsity)
                )
                p.start()
                p.join()
                
                if p.exitcode != 0:
                    raise RuntimeError(f"Subprocess for HPO trial exited with non-zero code {p.exitcode}. Aborting optimization.")
            
        end_time = time.time()
        
        # Reload the study to retrieve the best trial computed by the subprocesses
        study = optuna.load_study(
            study_name=study_name,
            storage=storage,
            sampler=self._sampler
        )
        best_trial = study.best_trial
        
        logger.info(f"CNN optimization ({mask_mode}) completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Best validation accuracy: {best_trial.value:.4f}")
        logger.info(f"Best CNN params: {best_trial.params}")
        
        # Restore original configs
        global_config.max_train_samples = orig_max_train
        global_config.max_val_samples = orig_max_val
        global_config.use_feature_cache = orig_use_cache
        global_config.mask_mode = orig_mask_mode
        
        # Save results
        output_file = config.cnn_config_output_file_template.format(mask_mode=mask_mode)
        output_data = best_trial.params.copy()
        output_data['validation_accuracy'] = best_trial.value
        self.save_json(output_data, output_file)
        
        if mask_mode == 'ga':
            random_output_file = config.cnn_config_output_file_template.format(mask_mode='random')
            
            # Retrieve GA params (excluding validation_accuracy metadata)
            ga_params = best_trial.params.copy()
            
            # Evaluate random mask mode on the validation set using these GA parameters
            logger.info("\nEvaluating Random mask mode with GA parameters to obtain accurate validation accuracy...")
            import multiprocessing
            ctx = multiprocessing.get_context('spawn')
            eval_q = ctx.Queue()
            # Retrieve random sparsity target
            target_sparsity = self.json_ga_config.get('target_sparsity', 0.45)
            
            p_eval = ctx.Process(
                target=_run_single_evaluation_subprocess,
                args=('random', ga_params, None, target_sparsity, eval_q)
            )
            p_eval.start()
            random_val_acc = eval_q.get()
            p_eval.join()
            
            logger.info(f"Actual validation accuracy for Random mask mode: {random_val_acc:.4f}")
            
            # Save random config with actual validation accuracy
            random_output_data = ga_params.copy()
            random_output_data['validation_accuracy'] = random_val_acc
            self.save_json(random_output_data, random_output_file)
            logger.info(f"Saved optimized CNN config for 'random' to {random_output_file}")
        
        # Clear TensorFlow and garbage collect
        import tensorflow as tf
        tf.keras.backend.clear_session()
        import gc
        gc.collect()
        
        return best_trial.params


def main():
    """Main function to run the hyperparameter optimization."""
    parser = argparse.ArgumentParser(description="Run hyperparameter optimization.")
    parser.add_argument(
        '--phase', 
        type=str, 
        choices=['weights', 'ga', 'cnn'], 
        required=True,
        help="Optimization phase to run: 'weights' (Phase 1), 'ga' (Phase 2), or 'cnn' (Phase 3)."
    )
    parser.add_argument(
        '--mask-mode',
        type=str,
        choices=['ga', 'random', 'none', 'all'],
        default='all',
        help="Mask mode to tune CNN for (only used for cnn phase: ga, random, none, or all)"
    )
    args = parser.parse_args()
    
    try:
        optimizer = HyperparameterOptimizer()
        
        if args.phase == 'weights':
            best_weights = optimizer.optimize_feature_weights()
            print(f"\nBest Feature Weights: {best_weights}")
        elif args.phase == 'ga':
            best_ga_config = optimizer.optimize_ga_config()
            print(f"\nBest GA Configuration: {best_ga_config}")
        elif args.phase == 'cnn':
            mask_modes = ['ga', 'random', 'none'] if args.mask_mode == 'all' else [args.mask_mode]
            for mode in mask_modes:
                best_cnn = optimizer.optimize_cnn(mode)
                print(f"\nBest CNN Params for '{mode}': {best_cnn}")
                
    except KeyboardInterrupt:
        print("\nOptimization stopped by user.")
    except Exception as e:
        print(f"\nOptimization failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()