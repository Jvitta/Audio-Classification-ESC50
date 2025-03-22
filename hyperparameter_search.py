import os
import numpy as np
import itertools
import json
import torch
import time
from datetime import datetime
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
from train import cross_validation
import argparse

def grid_search(data_path, param_grid, num_folds=4, test_fold=5, epochs_per_config=10):
    """
    Perform grid search for hyperparameter optimization
    
    Args:
        data_path: Path to preprocessed data
        param_grid: Dictionary of parameter names to lists of values to try
        num_folds: Number of folds to use for cross-validation
        test_fold: Fold to reserve for final testing
        epochs_per_config: Number of epochs to train for each configuration
    
    Returns:
        best_config: Configuration that achieved best validation performance
        all_results: List of all configurations and their results
    """
    print("Starting hyperparameter grid search...")
    
    # Generate all combinations of hyperparameters
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    print(f"Total configurations to evaluate: {len(param_combinations)}")
    
    all_results = []
    best_mean_val_acc = 0
    best_config = None
    
    # Set up MLflow experiment
    experiment_name = "esc50_hyperparameter_search"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    
    mlflow.set_experiment(experiment_name)
    
    # Evaluate each hyperparameter configuration
    for i, param_combination in enumerate(param_combinations):
        config = {param_names[j]: param_combination[j] for j in range(len(param_names))}
        # Add fixed parameters
        config['num_epochs'] = epochs_per_config
        
        print(f"\n===== Configuration {i+1}/{len(param_combinations)} =====")
        print(json.dumps(config, indent=2))
        
        # Track this configuration with MLflow
        with mlflow.start_run(run_name=f"config_{i+1}"):
            # Log parameters
            for key, value in config.items():
                mlflow.log_param(key, value)
            
            # Evaluate this configuration with cross-validation
            _, cv_results = cross_validation(config, data_path, num_folds, test_fold)
            
            # Store results
            result = {
                'config': config,
                'mean_val_acc': cv_results['mean_val_acc'],
                'std_val_acc': cv_results['std_val_acc'],
                'fold_val_accs': cv_results['fold_val_accs']
            }
            all_results.append(result)
            
            # Log to MLflow
            mlflow.log_metric("mean_val_acc", cv_results['mean_val_acc'])
            mlflow.log_metric("std_val_acc", cv_results['std_val_acc'])
            
            # Update best configuration if needed
            if cv_results['mean_val_acc'] > best_mean_val_acc:
                best_mean_val_acc = cv_results['mean_val_acc']
                best_config = config.copy()
                print(f"New best configuration found! Mean validation accuracy: {best_mean_val_acc:.4f}")
    
    # Sort results by mean validation accuracy (descending)
    all_results.sort(key=lambda x: x['mean_val_acc'], reverse=True)
    
    print("\n===== Grid Search Results =====")
    print(f"Best configuration: {json.dumps(best_config, indent=2)}")
    print(f"Best mean validation accuracy: {best_mean_val_acc:.4f}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f'results/hyperparameter_search_{timestamp}.json'
    
    with open(results_path, 'w') as f:
        json.dump({
            'best_config': best_config,
            'all_results': all_results[:5]  # Save top 5 configurations
        }, f, indent=2)
    
    print(f"Results saved to {results_path}")
    
    # Plot comparison of top configurations
    plot_top_configs(all_results[:5])
    
    return best_config, all_results

def plot_top_configs(top_results):
    """
    Plot comparison of top hyperparameter configurations
    
    Args:
        top_results: List of top configurations and their results
    """
    plt.figure(figsize=(12, 6))
    
    # Bar chart of mean validation accuracies
    plt.subplot(1, 2, 1)
    accuracies = [result['mean_val_acc'] for result in top_results]
    errors = [result['std_val_acc'] for result in top_results]
    config_labels = [f"Config {i+1}" for i in range(len(top_results))]
    
    plt.bar(config_labels, accuracies, yerr=errors, capsize=10)
    plt.ylim(max(0, min(accuracies) - 0.1), min(1.0, max(accuracies) + 0.1))
    plt.title('Validation Accuracy by Configuration')
    plt.ylabel('Mean Validation Accuracy')
    plt.xlabel('Configuration')
    
    # Add configuration details as text
    plt.subplot(1, 2, 2)
    plt.axis('off')
    config_text = "Top Configurations:\n\n"
    
    for i, result in enumerate(top_results):
        config_text += f"Config {i+1}:\n"
        for key, value in result['config'].items():
            if key != 'num_epochs':  # Skip epochs as it's fixed for hyperparameter search
                config_text += f"  {key}: {value}\n"
        config_text += f"  Accuracy: {result['mean_val_acc']:.4f} ± {result['std_val_acc']:.4f}\n\n"
    
    plt.text(0, 0.5, config_text, fontsize=9, verticalalignment='center')
    
    plt.tight_layout()
    os.makedirs('visualizations', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'visualizations/hyperparameter_comparison_{timestamp}.png')
    
    # Log to MLflow if in active run
    if mlflow.active_run():
        mlflow.log_artifact(f'visualizations/hyperparameter_comparison_{timestamp}.png')

def save_baseline_configs():
    """
    Save baseline configurations to the configs directory.
    These can be useful starting points for manual tuning.
    """
    os.makedirs('configs', exist_ok=True)
    
    # Baseline configuration for quick testing
    quick_config = {
        'batch_size': 32,
        'num_epochs': 1,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'use_lr_scheduler': True,
        'save_fold_models': False
    }
    
    # Baseline configuration for development
    baseline_config = {
        'batch_size': 32,
        'num_epochs': 5,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'use_lr_scheduler': True,
        'save_fold_models': False
    }
    
    # Baseline configuration for production
    production_config = {
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'use_lr_scheduler': True,
        'save_fold_models': True
    }
    
    # Save configurations
    with open(os.path.join('configs', 'quick_config.json'), 'w') as f:
        json.dump(quick_config, f, indent=2)
    
    with open(os.path.join('configs', 'baseline_config.json'), 'w') as f:
        json.dump(baseline_config, f, indent=2)
    
    with open(os.path.join('configs', 'production_config.json'), 'w') as f:
        json.dump(production_config, f, indent=2)
    
    print("Saved baseline configurations to configs directory:")
    print("  - configs/quick_config.json (1 epoch, for testing)")
    print("  - configs/baseline_config.json (5 epochs, for development)")
    print("  - configs/production_config.json (50 epochs, for final training)")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Hyperparameter search for audio classification')
    parser.add_argument('--create-baseline-configs', action='store_true',
                      help='Create baseline configuration files without running search')
    args = parser.parse_args()
    
    # Create baseline configs if requested
    if args.create_baseline_configs:
        save_baseline_configs()
        return
    
    # Data path
    data_path = 'data/preprocessed/esc50_preprocessed.npz'
    
    # Define parameter grid to search
    param_grid = {
        'batch_size': [16, 32, 64],
        'learning_rate': [0.01, 0.001, 0.0001],
        'weight_decay': [1e-4, 1e-5, 0],
        'use_lr_scheduler': [True, False]
    }
    
    # Number of epochs for each configuration
    epochs_per_config = 5
    
    # Run hyperparameter search
    start_time = time.time()
    best_config, all_results = grid_search(
        data_path, 
        param_grid, 
        epochs_per_config=epochs_per_config
    )
    elapsed_time = time.time() - start_time
    
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
    print(f"Average time per configuration: {elapsed_time / (len(all_results)):.2f} seconds")
    
    # Create configs directory if it doesn't exist
    os.makedirs('configs', exist_ok=True)
    
    # Save best configuration for later use
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_filename = f"best_config_{timestamp}.json"
    config_path = os.path.join('configs', config_filename)
    
    with open(config_path, 'w') as f:
        # For final training, we want to use more epochs
        final_config = best_config.copy()
        final_config['num_epochs'] = 50  # Use more epochs for final training
        json.dump(final_config, f, indent=2)
    
    # Also save a generic best_config.json for easy reference
    with open(os.path.join('configs', 'best_config.json'), 'w') as f:
        final_config = best_config.copy()
        final_config['num_epochs'] = 50  # Use more epochs for final training
        json.dump(final_config, f, indent=2)
    
    # Optionally, you can now train the final model with the best configuration
    print("\nTo train with the best configuration, run:")
    print(f"python train.py --config configs/{config_filename} --mode final")
    print("Or using the generic config:")
    print("python train.py --config configs/best_config.json --mode final")
    
    print(f"Best configuration saved to configs/{config_filename} and configs/best_config.json")

if __name__ == "__main__":
    main()
