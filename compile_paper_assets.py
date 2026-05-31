# compile_paper_assets.py
import os
import json
import logging
import argparse
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('PaperAssets')

def parse_args():
    parser = argparse.ArgumentParser(description="Compile Case Study Results and Evolved Hyperparameters into CSV Assets")
    parser.add_argument('--output-dir', type=str, default='output',
                        help="Directory where results JSONs are stored")
    parser.add_argument('--paper-dir', type=str, default='paper',
                        help="Directory where paper-ready compiled CSV assets will be written")
    return parser.parse_args()

def load_all_metrics(output_dir):
    """Load results from case studies and JPEG robustness scans."""
    results = {
        'In-Distribution': {'none': None, 'ga': None, 'random': None},
        'Cross-Generator': {'none': None, 'ga': None, 'random': None}
    }
    
    # 1. Load 6 Case Study JSONs
    if os.path.exists(output_dir):
        files = [f for f in os.listdir(output_dir) if f.startswith('results_') and f.endswith('.json')]
        logger.info(f"Found {len(files)} results JSON files: {files}")
        for file in files:
            path = os.path.join(output_dir, file)
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                
                mask_mode = data.get('mask_mode')
                train_gens = data.get('generator_train', [])
                test_gens = data.get('generator_test', [])
                
                is_in_dist = sorted(train_gens) == sorted(test_gens)
                setup = 'In-Distribution' if is_in_dist else 'Cross-Generator'
                
                if mask_mode in results[setup]:
                    results[setup][mask_mode] = data
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
                
    # 2. Load JPEG Robustness metrics if available
    jpeg_metrics = {}
    jpeg_path = os.path.join(output_dir, 'jpeg_robustness_metrics.json')
    if os.path.exists(jpeg_path):
        try:
            with open(jpeg_path, 'r') as f:
                jpeg_metrics = json.load(f)
            logger.info("Loaded JPEG robustness metrics from file.")
        except Exception as e:
            logger.warning(f"Could not load JPEG robustness metrics: {e}")
            
    return results, jpeg_metrics

def generate_results_csv(results, jpeg_metrics, paper_dir):
    """Compile case studies and JPEG robustness results into a comparison CSV file."""
    rows = []
    setups = ['In-Distribution', 'Cross-Generator']
    modes = ['none', 'ga', 'random']
    
    for setup in setups:
        for mode in modes:
            data = results[setup][mode]
            row = {
                'Setup': setup,
                'Mask Mode': mode.upper(),
                'Train Generators': "N/A",
                'Test Generators': "N/A",
                'Sparsity': 1.0,
                'Accuracy': np.nan,
                'Balanced Accuracy': np.nan,
                'F1 Score': np.nan,
                'Loss': np.nan,
                'JPEG_Q75_Accuracy': np.nan,
                'JPEG_Q50_Accuracy': np.nan,
                'JPEG_Q30_Accuracy': np.nan
            }
            
            if data is not None:
                metrics = data.get('metrics', {})
                row['Train Generators'] = ",".join(data.get('generator_train', []))
                row['Test Generators'] = ",".join(data.get('generator_test', []))
                
                sparsity_mean = data.get('mask_sparsity_mean')
                row['Sparsity'] = sparsity_mean if sparsity_mean is not None else 1.0
                
                row['Accuracy'] = metrics.get('accuracy')
                row['Balanced Accuracy'] = metrics.get('balanced_accuracy')
                row['F1 Score'] = metrics.get('f1')
                row['Loss'] = metrics.get('loss')
                
            # Add JPEG robustness values if they exist
            if setup in jpeg_metrics and mode in jpeg_metrics[setup]:
                q_data = jpeg_metrics[setup][mode].get('qualities', {})
                row['JPEG_Q75_Accuracy'] = q_data.get('75')
                row['JPEG_Q50_Accuracy'] = q_data.get('50')
                row['JPEG_Q30_Accuracy'] = q_data.get('30')
                
            rows.append(row)
            
    df = pd.DataFrame(rows)
    csv_path = os.path.join(paper_dir, 'experiments_comparison.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved compiled paper results to CSV -> {csv_path}")
    return df

def generate_feature_weights_csv(paper_dir):
    """Compile optimized feature weights from best_feature_weights.json into a CSV."""
    weights_path = "best_feature_weights.json"
    if not os.path.exists(weights_path):
        logger.warning("best_feature_weights.json not found. Skipping feature weights CSV.")
        return
        
    try:
        with open(weights_path, 'r') as f:
            weights = json.load(f)
            
        rows = []
        for key, val in weights.items():
            if key == '__fitness__':
                rows.append({'Feature': 'Fusion Fitness Score', 'Weight': val})
            else:
                rows.append({'Feature': key, 'Weight': val})
                
        df = pd.DataFrame(rows)
        csv_path = os.path.join(paper_dir, 'feature_weights.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved feature weights to CSV -> {csv_path}")
    except Exception as e:
        logger.error(f"Error compiling feature weights CSV: {e}")

def generate_ga_config_csv(paper_dir):
    """Compile optimized Genetic Algorithm configuration from best_ga_config.json into a CSV."""
    ga_path = "best_ga_config.json"
    if not os.path.exists(ga_path):
        logger.warning("best_ga_config.json not found. Skipping GA config CSV.")
        return
        
    try:
        with open(ga_path, 'r') as f:
            ga_config = json.load(f)
            
        rows = []
        for key, val in ga_config.items():
            rows.append({'Hyperparameter': key, 'Value': val})
            
        df = pd.DataFrame(rows)
        csv_path = os.path.join(paper_dir, 'ga_hyperparameters.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved GA hyperparameters to CSV -> {csv_path}")
    except Exception as e:
        logger.error(f"Error compiling GA config CSV: {e}")

def generate_cnn_config_csv(paper_dir):
    """Compile optimized CNN configurations for all mask modes side-by-side into a CSV."""
    configs = {}
    modes = [('none', 'No Mask (Baseline)'), ('ga', 'GA Evolved Mask'), ('random', 'Random Mask (Control)')]
    for mode, label in modes:
        file_path = f"best_cnn_config_{mode}.json"
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    configs[mode] = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
                
    if not configs:
        logger.warning("No best_cnn_config_*.json files found. Skipping CNN config CSV.")
        return
        
    try:
        all_keys = set()
        for cfg in configs.values():
            all_keys.update(cfg.keys())
            
        # Determine canonical order of hyperparameters
        ordered_keys = [
            'learning_rate', 'dropout_1', 'dropout_2', 
            'dense_units', 'l2_reg', 'optimizer', 'validation_accuracy'
        ]
        for k in sorted(all_keys):
            if k not in ordered_keys:
                ordered_keys.append(k)
                
        rows = []
        for k in ordered_keys:
            if any(k in cfg for cfg in configs.values()):
                row = {'Hyperparameter': k}
                row['No Mask (Baseline)'] = configs.get('none', {}).get(k, np.nan)
                row['GA Evolved Mask'] = configs.get('ga', {}).get(k, np.nan)
                row['Random Mask (Control)'] = configs.get('random', {}).get(k, np.nan)
                rows.append(row)
                
        df = pd.DataFrame(rows)
        csv_path = os.path.join(paper_dir, 'cnn_hyperparameters.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved CNN hyperparameters to CSV -> {csv_path}")
    except Exception as e:
        logger.error(f"Error compiling CNN config CSV: {e}")

def main():
    args = parse_args()
    
    # Ensure paper directory exists
    os.makedirs(args.paper_dir, exist_ok=True)
    
    logger.info(f"Starting CSV paper assets compilation. Output directory: {args.output_dir}, Paper directory: {args.paper_dir}")
    
    # 1. Load case studies and jpeg robustness metrics
    results, jpeg_metrics = load_all_metrics(args.output_dir)
    
    # 2. Export Case Studies & Robustness CSV
    generate_results_csv(results, jpeg_metrics, args.paper_dir)
    
    # 3. Export Feature Weights CSV
    generate_feature_weights_csv(args.paper_dir)
    
    # 4. Export GA Config CSV
    generate_ga_config_csv(args.paper_dir)
    
    # 5. Export CNN Hyperparameters CSV
    generate_cnn_config_csv(args.paper_dir)
    
    logger.info("CSV paper assets compilation completed successfully!")

if __name__ == '__main__':
    main()
