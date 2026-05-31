# plot_experiments_summary.py
import os
import json
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ExperimentsSummary')

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Comparative Visualizations for the 6 Case Studies")
    parser.add_argument('--output-dir', type=str, default='output',
                        help="Directory where results JSONs are stored and where comparison plot will be saved")
    return parser.parse_args()

def load_results(output_dir):
    """
    Search for results_*.json files in the output directory and group them by setup and mode.
    """
    results = {
        'In-Distribution': {'none': None, 'ga': None, 'random': None},
        'Cross-Generator': {'none': None, 'ga': None, 'random': None}
    }
    
    if not os.path.exists(output_dir):
        logger.error(f"Output directory '{output_dir}' does not exist.")
        return results

    files = [f for f in os.listdir(output_dir) if f.startswith('results_') and f.endswith('.json')]
    logger.info(f"Found {len(files)} result JSON files in {output_dir}.")

    for file in files:
        path = os.path.join(output_dir, file)
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            mask_mode = data.get('mask_mode')
            train_gens = data.get('generator_train', [])
            test_gens = data.get('generator_test', [])
            
            # Determine if in-distribution (train == test) or cross-generator
            is_in_dist = sorted(train_gens) == sorted(test_gens)
            setup = 'In-Distribution' if is_in_dist else 'Cross-Generator'
            
            if mask_mode in results[setup]:
                results[setup][mask_mode] = data
                logger.info(f"Loaded {setup} | Mode: {mask_mode} from {file}")
            
        except Exception as e:
            logger.error(f"Error reading results file {file}: {e}")

    return results

def generate_summary_table(results):
    """
    Generate and print a styled markdown summary table of the 6 case studies.
    """
    print("\n" + "="*80)
    print("                      6 CASE STUDIES SUMMARY TABLE")
    print("="*80)
    
    headers = [
        "Setup", "Mode", "Train Gens", "Test Gens", 
        "Accuracy", "Balanced Acc", "F1 Score", "Loss", "Sparsity"
    ]
    
    col_widths = [18, 8, 18, 18, 10, 14, 10, 8, 10]
    
    # Print Markdown Headers
    header_str = " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    print(f"| {header_str} |")
    
    separator_str = " | ".join("-" * w for w in col_widths)
    print(f"| {separator_str} |")
    
    # Print Rows
    setups = ['In-Distribution', 'Cross-Generator']
    modes = ['none', 'ga', 'random']
    
    for setup in setups:
        for mode in modes:
            data = results[setup][mode]
            if data is None:
                row = [
                    setup, mode.upper(), "N/A", "N/A", 
                    "N/A", "N/A", "N/A", "N/A", "N/A"
                ]
            else:
                metrics = data.get('metrics', {})
                train = ",".join(data.get('generator_train', []))
                test = ",".join(data.get('generator_test', []))
                
                acc = f"{metrics.get('accuracy', 0.0):.4%}"
                bal_acc = f"{metrics.get('balanced_accuracy', 0.0):.4%}"
                f1 = f"{metrics.get('f1', 0.0):.4%}"
                loss = f"{metrics.get('loss', 0.0):.4f}"
                
                sparsity_mean = data.get('mask_sparsity_mean')
                sparsity = f"{sparsity_mean:.1%}" if sparsity_mean is not None else "100.0%"
                
                row = [
                    setup, mode.upper(), train, test, 
                    acc, bal_acc, f1, loss, sparsity
                ]
                
            row_str = " | ".join(f"{str(val):<{w}}" for val, w in zip(row, col_widths))
            print(f"| {row_str} |")
            
    print("="*80 + "\n")

def plot_comparisons(results, output_dir):
    """
    Generate grouped bar charts comparing accuracy and F1 score across the 6 case studies.
    """
    setups = ['In-Distribution', 'Cross-Generator']
    modes = ['none', 'ga', 'random']
    
    colors = {
        'none': '#7f8c8d',     # Grey
        'ga': '#00bcd4',       # Cyan
        'random': '#e67e22'    # Orange
    }
    
    labels = {
        'none': 'No Mask (Baseline)',
        'ga': 'GA Evolved Mask',
        'random': 'Random Mask (Control)'
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(len(setups))
    width = 0.25

    # 1. Plot Test Accuracy
    ax_acc = axes[0]
    for i, mode in enumerate(modes):
        values = []
        for setup in setups:
            data = results[setup][mode]
            acc = data.get('metrics', {}).get('accuracy', 0.0) if data is not None else 0.0
            values.append(acc)
            
        offset = (i - 1) * width
        rects = ax_acc.bar(x + offset, values, width, label=labels[mode], color=colors[mode], edgecolor='black', alpha=0.9)
        
        # Add labels on top of bars
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax_acc.annotate(f'{height:.2%}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax_acc.set_title("Test Accuracy Comparison", fontsize=14, fontweight='bold', pad=15)
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(setups, fontsize=12)
    ax_acc.set_ylabel("Accuracy", fontsize=12)
    ax_acc.set_ylim(0.0, 1.05)
    ax_acc.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax_acc.legend(loc="lower left", fontsize=10)

    # 2. Plot F1 Score
    ax_f1 = axes[1]
    for i, mode in enumerate(modes):
        values = []
        for setup in setups:
            data = results[setup][mode]
            f1 = data.get('metrics', {}).get('f1', 0.0) if data is not None else 0.0
            values.append(f1)
            
        offset = (i - 1) * width
        rects = ax_f1.bar(x + offset, values, width, label=labels[mode], color=colors[mode], edgecolor='black', alpha=0.9)
        
        # Add labels on top of bars
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax_f1.annotate(f'{height:.2%}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax_f1.set_title("F1 Score Comparison", fontsize=14, fontweight='bold', pad=15)
    ax_f1.set_xticks(x)
    ax_f1.set_xticklabels(setups, fontsize=12)
    ax_f1.set_ylabel("F1 Score", fontsize=12)
    ax_f1.set_ylim(0.0, 1.05)
    ax_f1.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax_f1.legend(loc="lower left", fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'experiments_summary_comparison.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    logger.info(f"Saved comparative summary chart to {save_path}")

def main():
    args = parse_args()
    
    logger.info("="*70)
    logger.info("COMPILING 6 CASE STUDIES VISUALIZATION SUMMARY")
    logger.info("="*70)
    
    results = load_results(args.output_dir)
    
    # 1. Print Summary Table
    generate_summary_table(results)
    
    # 2. Plot Grouped Comparisons
    plot_comparisons(results, args.output_dir)
    
    logger.info("Summary compilation completed successfully!")

if __name__ == '__main__':
    main()
