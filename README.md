# Genetic-AI-Detector

This project is a hybrid machine learning pipeline that detects AI-generated images by combining a **Genetic Algorithm (GA)** for adaptive feature selection with a **Convolutional Neural Network (CNN)** for classification. The system analyzes images by dividing them into patches and evaluating handcrafted visual features — such as gradient perfection, symmetry, noise distribution, and texture — to identify artifacts characteristic of AI-generated content. A genetic algorithm evolves a set of conditional rules that determine which image patches are most informative, and those selected patches feed into the CNN alongside the raw image to produce a final human-vs-AI prediction.

---

## Architecture Overview

### Data Pipeline (`data_loader.py`)
Images and labels are loaded from a CSV file. The loader performs stratified subsampling to preserve class balance, applies center-crop preprocessing, and builds efficient `tf.data` pipelines with optional augmentation (flips, brightness, contrast) for training.

### Feature Extraction (`feature_extractor.py`, `structural_features.py`, `texture_features.py`)
Each image is divided into fixed-size patches (`patch_size=16` by default). For every patch, eight features are computed:
- **Gradient** – detects unnaturally smooth or perfect gradients
- **Pattern** – uses FFT to identify repetitive spectral artifacts
- **Edge** – applies a TensorFlow-native Canny detector for edge coherence
- **Symmetry** – measures horizontal and vertical symmetry anomalies
- **Noise** – analyzes noise uniformity via Laplacian filtering
- **Texture** – extracts Local Binary Patterns (LBP)
- **Color** – detects saturation distribution anomalies in HSV space
- **Hash** – computes perceptual hash scores (pHash/dHash)

### Genetic Algorithm (`genetic_algorithm.py`, `fitness_evaluation.py`)
The GA (implemented with [DEAP](https://deap.readthedocs.io/)) evolves a population of rule sets. Each individual is a tensor of up to `max_possible_rules` conditional rules of the form `if feature_X [>|<] threshold → include/exclude patch`. Fitness is evaluated by applying each rule set to precomputed patch features, generating a dynamic binary patch mask per image, and computing a weighted score across balanced accuracy, F1, precision, recall, patch efficiency, mask connectivity, and rule simplicity. Features are precomputed once at initialization and reused across all generations and trials for efficiency.

### CNN Model (`model_architecture.py`)
The `AIDetectorModel` is a multi-input CNN. When feature extraction is enabled, a dedicated branch processes the GA-selected feature maps, which are then fused with the main image branch via a cross-attention mechanism. The combined representation is passed through dense layers to a sigmoid output. Training uses binary cross-entropy with early stopping and learning rate reduction callbacks.

### Hyperparameter Optimization (`hyperparameter_optimizer.py`, `optuna_config.py`)
[Optuna](https://optuna.org/) is used for two-phase optimization: first tuning the eight feature weights, then tuning GA parameters (population size, generation count, crossover/mutation probabilities, etc.). The same GA instance and precomputed features are reused across all Optuna trials to avoid redundant computation.
