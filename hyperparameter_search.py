import os
import numpy as np
import itertools
import json
import torch
import torch.nn as nn
import torch.optim as optim
import time
from datetime import datetime
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
from train import cross_validation
import train
import argparse
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from models.resnet import create_resnet18
from torch_lr_finder import LRFinder

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

def get_optimal_lr(lr_finder):
    """
    Get the optimal learning rate from the lr_finder history.
    This is a replacement for the non-existent suggestion() method.
    
    The approach finds the point with the steepest downward slope
    in the loss vs. learning rate curve.
    
    Args:
        lr_finder: The LRFinder object after running range_test
        
    Returns:
        float: The suggested optimal learning rate
    """
    lrs = lr_finder.history["lr"]
    losses = lr_finder.history["loss"]
    
    # Handle empty history
    if not lrs or not losses:
        print("Warning: Empty learning rate history. Using default learning rate.")
        return 1e-3
    
    # Skip the beginning and end of the curve for more stable results
    skip_start = min(10, len(lrs) // 10)
    skip_end = min(5, len(lrs) // 20)
    
    if skip_start >= len(lrs) or skip_end >= len(lrs) or skip_start + skip_end >= len(lrs):
        # If not enough data points, use a simpler approach
        if len(lrs) > 3:
            min_loss_idx = losses.index(min(losses))
            # Return the learning rate at minimum loss or slightly before
            return lrs[max(0, min_loss_idx - 1)]
        return lrs[0] if lrs else 1e-3  # Default if no data
    
    # Calculate gradients with safeguards for division by zero
    gradients = []
    for i in range(skip_start, len(lrs) - skip_end - 1):
        lr_diff = lrs[i + 1] - lrs[i]
        if abs(lr_diff) < 1e-10:  # Avoid division by near-zero
            continue
        gradients.append((losses[i + 1] - losses[i]) / lr_diff)
    
    # Check if we have valid gradients
    if not gradients:
        print("Warning: Could not calculate valid gradients. Using median learning rate.")
        return lrs[len(lrs) // 2]
    
    # Find the point with the steepest negative gradient
    # (use smoothed gradient to avoid noise)
    smooth_window = min(5, len(gradients) // 5)
    if smooth_window > 0 and len(gradients) > smooth_window:
        smoothed_gradients = []
        for i in range(len(gradients) - smooth_window + 1):
            smoothed_gradients.append(sum(gradients[i:i+smooth_window]) / smooth_window)
        
        if not smoothed_gradients:
            # If we somehow ended up with no smoothed gradients
            steepest_idx = gradients.index(min(gradients))
        else:
            steepest_idx = smoothed_gradients.index(min(smoothed_gradients))
    else:
        steepest_idx = gradients.index(min(gradients))
    
    # Return the learning rate at the steepest point
    suggested_lr = lrs[skip_start + steepest_idx]
    
    return suggested_lr

def find_learning_rate(data_path, batch_size=32, test_fold=5, min_lr=1e-7, max_lr=10, num_iter=100):
    """
    Use the LR Finder to discover the optimal learning rate for the model.
    Uses k-fold cross-validation to get a more robust estimate.
    
    Args:
        data_path: Path to preprocessed data
        batch_size: Batch size to use for training
        test_fold: Fold to reserve for final testing
        min_lr: Minimum learning rate to explore
        max_lr: Maximum learning rate to explore
        num_iter: Number of iterations for the LR finder
    
    Returns:
        suggested_lr: The suggested learning rate based on the finder
    """
    print("Starting Learning Rate Finder with cross-validation...")
    
    # Set up MLflow if available
    try:
        experiment_name = "esc50_learning_rate_finder"
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except:
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        
        mlflow.set_experiment(experiment_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = mlflow.start_run(run_name=f"lr_finder_cv_{timestamp}").info.run_id
    except Exception as e:
        print(f"Warning: Could not set up MLflow: {e}")
        run_id = None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Now we'll create a temporary wrapper that will override the train_one_cv_split 
    # function from train module with our LR finder version
    original_train_one_cv_split = getattr(train, 'train_one_cv_split')
    
    # Track results across folds
    suggested_lrs = []
    fold_ranges = []
    
    # Define the custom training function that will replace train_one_cv_split
    def lr_finder_train_split(model, train_loader, val_loader, config, fold=None):
        """
        Override the regular training with LR finder
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Initialize optimizer and loss
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
        
        # Initialize LR Finder
        lr_finder = LRFinder(model, optimizer, criterion, device=device)
        
        # Run LR Finder
        print(f"Running LR Finder for fold {fold} from {min_lr} to {max_lr} over {num_iter} iterations...")
        lr_finder.range_test(train_loader, val_loader=val_loader, end_lr=max_lr, num_iter=num_iter, step_mode="exp")
        
        # Get suggestion for this fold
        fold_suggested_lr = get_optimal_lr(lr_finder)
        suggested_lrs.append(fold_suggested_lr)
        print(f"Fold {fold} suggested learning rate: {fold_suggested_lr:.8f}")
        
        # Store min/max boundaries of the good LR range for this fold
        losses = lr_finder.history["loss"]
        lrs = lr_finder.history["lr"]
        
        # Find the range where loss is decreasing
        smooth_f = 0.05  # Smoothing factor for loss curve
        smooth_losses = []
        avg_loss = 0
        for loss in losses:
            avg_loss = smooth_f * loss + (1 - smooth_f) * avg_loss
            smooth_losses.append(avg_loss)
        
        # Find the point of steepest decline
        derivatives = []
        valid_range_found = False
        
        try:
            for i in range(1, len(lrs)):
                if abs(lrs[i] - lrs[i-1]) < 1e-10:  # Avoid division by near-zero
                    continue
                derivatives.append((smooth_losses[i] - smooth_losses[i-1]) / (lrs[i] - lrs[i-1]))
            
            if derivatives:
                # Find the min LR where loss starts decreasing significantly
                min_idx = 10  # Skip first few points
                if min_idx < len(derivatives) - 5:
                    for i in range(min_idx, len(derivatives) - 5):
                        if derivatives[i] < -0.5:  # Threshold for significant decrease
                            min_idx = i
                            break
                    
                    # Find the max LR where loss starts increasing again
                    max_idx = len(derivatives) - 5
                    for i in range(min_idx, len(derivatives) - 5):
                        if derivatives[i] > 0:  # Loss starts increasing
                            max_idx = i
                            break
                    
                    fold_range = (lrs[min_idx], lrs[max_idx])
                    fold_ranges.append(fold_range)
                    valid_range_found = True
        except Exception as e:
            print(f"Warning: Error calculating learning rate range: {e}")
        
        # If we couldn't determine a valid range, use a default range around the suggested LR
        if not valid_range_found:
            print(f"Could not determine a valid learning rate range for fold {fold}. Using default range.")
            fold_range = (fold_suggested_lr / 10, fold_suggested_lr * 10)
            fold_ranges.append(fold_range)
        
        # Plot for this fold
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            lr_finder.plot(ax=ax, skip_start=10, skip_end=5)
            ax.set_title(f'Fold {fold} Learning Rate Finder (Suggested LR: {fold_suggested_lr:.8f})')
            ax.set_xlabel('Learning Rate')
            ax.set_ylabel('Loss')
            ax.axvline(x=fold_suggested_lr, color='r', linestyle='--', alpha=0.7)
            ax.axvline(x=fold_range[0], color='g', linestyle='--', alpha=0.5)
            ax.axvline(x=fold_range[1], color='y', linestyle='--', alpha=0.5)
            
            # Save the fold plot
            os.makedirs('visualizations', exist_ok=True)
            fold_lr_plot_path = f'visualizations/lr_finder_fold_{fold}_{timestamp}.png'
            plt.savefig(fold_lr_plot_path)
            plt.close()
            
            # Log to MLflow
            if run_id:
                try:
                    mlflow.log_metric(f"fold_{fold}_suggested_lr", fold_suggested_lr)
                    mlflow.log_artifact(fold_lr_plot_path)
                except Exception as e:
                    print(f"Warning: Could not log to MLflow: {e}")
        except Exception as e:
            print(f"Warning: Error creating learning rate plot: {e}")
        
        # Instead of returning a trained model, we return a dummy model
        # This is OK since we're not using the CV results for actual training
        return model, {}, 0.0  # Return dummy history and acc - we don't use them
    
    # Temporarily replace the training function with our LR finder version
    setattr(train, 'train_one_cv_split', lr_finder_train_split)
    
    # Create a config that matches what cross_validation expects
    config = {
        'batch_size': batch_size,
        'num_epochs': 1,  # Not used for LR finding
        'learning_rate': 1e-5,  # Not used for LR finding
        'weight_decay': 1e-5,  # Will be used in the Adam optimizer
        'use_lr_scheduler': False  # Not needed for LR finding
    }
    
    # Run cross-validation which will use our modified training function
    try:
        # Suppress normal CV output
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        
        _, _ = train.cross_validation(config, data_path, num_folds=4, test_fold=test_fold)
        
        # Restore stdout
        sys.stdout = old_stdout
    finally:
        # Restore the original training function no matter what
        setattr(train, 'train_one_cv_split', original_train_one_cv_split)
    
    # Compute final suggested learning rate (geometric mean of fold suggestions)
    suggested_lr = np.exp(np.mean(np.log(suggested_lrs)))
    print(f"\nFinal suggested learning rate (geometric mean): {suggested_lr:.8f}")
    
    # Also calculate range intersection
    if fold_ranges:
        min_lr_range = max([r[0] for r in fold_ranges])
        max_lr_range = min([r[1] for r in fold_ranges])
        
        # Check if the range is valid
        if min_lr_range <= max_lr_range:
            print(f"Common learning rate range across folds: {min_lr_range:.8f} to {max_lr_range:.8f}")
        else:
            print("No common learning rate range found across folds. Using suggested learning rate only.")
            min_lr_range = suggested_lr / 10
            max_lr_range = suggested_lr * 10
    else:
        print("No valid learning rate ranges found. Using suggested learning rate and default range.")
        min_lr_range = suggested_lr / 10
        max_lr_range = suggested_lr * 10
    
    # Create a summary plot of all fold suggestions
    plt.figure(figsize=(10, 6))
    for i, lr in enumerate(suggested_lrs):
        plt.axvline(x=lr, color=f'C{i}', linestyle='--', alpha=0.7, label=f'Fold {i+1}: {lr:.8f}')
    
    plt.axvline(x=suggested_lr, color='r', linestyle='-', linewidth=2, label=f'Final: {suggested_lr:.8f}')
    plt.axvspan(min_lr_range, max_lr_range, alpha=0.1, color='green', label='Common range')
    
    plt.xscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlabel('Learning Rate')
    plt.title('Learning Rate Suggestions Across Folds')
    plt.legend()
    
    summary_plot_path = f'visualizations/lr_finder_summary_{timestamp}.png'
    plt.savefig(summary_plot_path)
    plt.close()
    
    # Save the suggested learning rate to a config file
    lr_config = {
        'batch_size': batch_size,
        'learning_rate': suggested_lr,
        'weight_decay': 1e-5,
        'use_lr_scheduler': True,
        'num_epochs': 50
    }
    
    os.makedirs('configs', exist_ok=True)
    config_path = os.path.join('configs', f'lr_finder_config_{timestamp}.json')
    with open(config_path, 'w') as f:
        json.dump(lr_config, f, indent=2)
    
    print(f"Configuration with suggested learning rate saved to {config_path}")
    
    # Also save to a standard name for easy reference
    standard_config_path = os.path.join('configs', 'lr_finder_config.json')
    with open(standard_config_path, 'w') as f:
        json.dump(lr_config, f, indent=2)
    
    print(f"Configuration also saved to {standard_config_path}")
    
    # Log to MLflow if available
    if run_id:
        try:
            mlflow.log_metric("suggested_lr", suggested_lr)
            mlflow.log_metric("min_lr_range", min_lr_range)
            mlflow.log_metric("max_lr_range", max_lr_range)
            mlflow.log_artifact(summary_plot_path)
            mlflow.log_artifact(config_path)
        except Exception as e:
            print(f"Warning: Could not log to MLflow: {e}")
            
        try:
            mlflow.end_run()
        except:
            pass
    
    return suggested_lr

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Hyperparameter search for audio classification')
    parser.add_argument('--create-baseline-configs', action='store_true',
                      help='Create baseline configuration files without running search')
    parser.add_argument('--find-lr', action='store_true',
                      help='Run learning rate finder to determine optimal learning rate range')
    parser.add_argument('--min-lr', type=float, default=1e-7,
                      help='Minimum learning rate to explore (default: 1e-7)')
    parser.add_argument('--max-lr', type=float, default=10.0,
                      help='Maximum learning rate to explore (default: 10.0)')
    parser.add_argument('--num-iter', type=int, default=100,
                      help='Number of iterations for LR finder (default: 100, higher values give smoother curves)')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for LR finder (default: 32)')
    args = parser.parse_args()
    
    # Create baseline configs if requested
    if args.create_baseline_configs:
        save_baseline_configs()
        return
    
    # Run learning rate finder if requested
    if args.find_lr:
        print(f"Starting learning rate finder with batch size {args.batch_size}")
        print(f"Learning rate range: {args.min_lr} to {args.max_lr} over {args.num_iter} iterations")
        try:
            suggested_lr = find_learning_rate(
                'data/preprocessed/esc50_preprocessed.npz',
                batch_size=args.batch_size,
                min_lr=args.min_lr,
                max_lr=args.max_lr,
                num_iter=args.num_iter
            )
            print(f"Learning rate finder completed successfully!")
            print(f"Suggested learning rate: {suggested_lr:.8f}")
            print(f"Configuration saved to configs/lr_finder_config.json")
        except Exception as e:
            print(f"Error running learning rate finder: {e}")
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
