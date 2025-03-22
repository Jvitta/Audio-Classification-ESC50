import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from models.resnet import create_resnet18
import json
import time
from datetime import datetime
import mlflow
import mlflow.pytorch
import argparse
import random

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """
    Validate the model
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            
            # Store predictions and targets for metrics
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    # Calculate metrics
    val_loss = running_loss / len(all_targets)
    val_acc = accuracy_score(all_targets, all_predictions)
    
    return val_loss, val_acc, all_targets, all_predictions

def train_one_cv_split(model, train_loader, val_loader, config, fold=None):
    """
    Train a model on one cross-validation split (multiple training folds, one validation fold)
    
    Args:
        model: PyTorch model
        train_loader: Training data loader (contains multiple folds)
        val_loader: Validation data loader (contains one fold)
        config: Dictionary with training configuration
        fold: Fold number used for validation (for logging)
    
    Returns:
        trained model, training history, best validation accuracy
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Extract training parameters
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    weight_decay = config.get('weight_decay', 0)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    if config.get('use_lr_scheduler', False):
        # Get scheduler parameters from config if available
        scheduler_patience = config.get('scheduler_patience', 5)
        scheduler_factor = config.get('scheduler_factor', 0.5)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=scheduler_factor, 
            patience=scheduler_patience,
            verbose=True
        )
        # Store initial learning rate for comparison
        prev_lr = [group['lr'] for group in optimizer.param_groups]
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
    }
    
    best_val_acc = 0.0
    best_model_state = None
    
    fold_str = f" (Fold {fold})" if fold is not None else ""
    mlflow_fold_str = f"_fold_{fold}" if fold is not None else ""
    print(f"Starting training{fold_str}...")
    
    for epoch in range(num_epochs):
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation phase
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        # Update learning rate if using scheduler
        if config.get('use_lr_scheduler', False):
            scheduler.step(val_loss)
            
            # Check if learning rate changed
            current_lr = [group['lr'] for group in optimizer.param_groups]
            if current_lr != prev_lr:
                print(f"Learning rate adjusted from {prev_lr[0]:.6f} to {current_lr[0]:.6f}")
                prev_lr = current_lr
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}{fold_str} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Log metrics to MLflow
        if mlflow.active_run():
            metrics = {
                f"train_loss{mlflow_fold_str}": train_loss,
                f"train_acc{mlflow_fold_str}": train_acc,
                f"val_loss{mlflow_fold_str}": val_loss,
                f"val_acc{mlflow_fold_str}": val_acc
            }
            mlflow.log_metrics(metrics, step=epoch)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, history, best_val_acc

class AugmentedDataset(TensorDataset):
    """
    Dataset wrapper that applies augmentation to spectrograms
    """
    def __init__(self, spectrograms, labels, use_augmentation=False, aug_strength=0.3):
        super(AugmentedDataset, self).__init__(spectrograms, labels)
        self.use_augmentation = use_augmentation
        self.aug_strength = aug_strength
    
    def __getitem__(self, index):
        spectrogram, label = super(AugmentedDataset, self).__getitem__(index)
        
        if self.use_augmentation and random.random() < 0.5:  # 50% chance to apply augmentation
            # Apply augmentation to the spectrogram
            spectrogram = self._augment_spectrogram(spectrogram)
        
        return spectrogram, label
    
    def _augment_spectrogram(self, spectrogram):
        """Apply augmentation to a single spectrogram tensor"""
        # Make a copy to avoid modifying the original
        aug_spec = spectrogram.clone()
        
        # Get dimensions (1, height, width)
        C, H, W = aug_spec.shape
        
        # Time shift: move the spectrogram left or right by a small amount
        if random.random() < 0.5:
            shift_amount = int(W * self.aug_strength * random.uniform(0.1, 1.0))
            if random.random() < 0.5:  # Shift left
                aug_spec[:, :, shift_amount:] = aug_spec[:, :, :-shift_amount]
                aug_spec[:, :, :shift_amount] = 0
            else:  # Shift right
                aug_spec[:, :, :-shift_amount] = aug_spec[:, :, shift_amount:]
                aug_spec[:, :, -shift_amount:] = 0
        
        # Frequency masking: mask some frequency bands
        if random.random() < 0.5:
            mask_height = int(H * self.aug_strength * random.uniform(0.1, 0.5))
            mask_start = random.randint(0, H - mask_height)
            aug_spec[:, mask_start:mask_start+mask_height, :] = 0
        
        # Time masking: mask some time steps
        if random.random() < 0.5:
            mask_width = int(W * self.aug_strength * random.uniform(0.1, 0.3))
            mask_start = random.randint(0, W - mask_width)
            aug_spec[:, :, mask_start:mask_start+mask_width] = 0
        
        # Magnitude perturbation: randomly adjust the magnitudes
        if random.random() < 0.5:
            magnitude_factor = 1.0 + random.uniform(-self.aug_strength, self.aug_strength)
            aug_spec = aug_spec * magnitude_factor
        
        return aug_spec

def cross_validation(config, data_path, num_folds=4, test_fold=5):
    """
    Perform cross-validation training
    
    Args:
        config: Dictionary with training configuration
        data_path: Path to preprocessed data
        num_folds: Number of folds to use for cross-validation
        test_fold: Fold to reserve for final testing
    
    Returns:
        best_config: Configuration that achieved best validation results
        cv_results: Cross-validation results
    """
    # Load data
    data = np.load(data_path)
    spectrograms = data['spectrograms']
    labels = data['labels']
    folds = data['folds']
    
    # Convert to torch tensors
    spectrograms_tensor = torch.tensor(spectrograms, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    # Get augmentation settings from config
    use_augmentation = config.get('use_augmentation', False)
    aug_strength = config.get('aug_strength', 0.3)
    
    # Prepare dataset with optional augmentation
    dataset = AugmentedDataset(
        spectrograms_tensor, 
        labels_tensor,
        use_augmentation=use_augmentation,
        aug_strength=aug_strength
    )
    
    # Track results across folds
    fold_val_accs = []
    fold_histories = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Log dataset info to MLflow
    if mlflow.active_run():
        mlflow.log_param("dataset_path", data_path)
        mlflow.log_param("num_samples", len(spectrograms))
        mlflow.log_param("num_classes", len(np.unique(labels)))
        mlflow.log_param("input_shape", spectrograms.shape[1:])
        if use_augmentation:
            mlflow.log_param("augmentation", "enabled")
            mlflow.log_param("aug_strength", aug_strength)
        else:
            mlflow.log_param("augmentation", "disabled")
    
    # Cross-validation loop
    for fold in range(1, num_folds + 1):
        print(f"\n--- Fold {fold} ---")
        
        # Create train/validation split based on folds
        # The test_fold (fold 5) is completely excluded
        val_indices = np.where(folds == fold)[0]
        train_indices = np.where((folds != fold) & (folds != test_fold))[0]
        
        # Create samplers
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        
        # Create data loaders
        train_loader = DataLoader(
            dataset, batch_size=config['batch_size'], sampler=train_sampler
        )
        val_loader = DataLoader(
            dataset, batch_size=config['batch_size'], sampler=val_sampler
        )
        
        # Create model, with optional dropout support
        dropout_rate = config.get('dropout_rate', 0.0)  # Default to 0 (no dropout)
        model = create_resnet18(num_classes=len(np.unique(labels)), dropout_rate=dropout_rate)
        
        # Train on this fold
        model, history, val_acc = train_one_cv_split(
            model, train_loader, val_loader, config, fold=fold
        )
        
        # Save results
        fold_val_accs.append(val_acc)
        fold_histories.append(history)
        
        # Log fold results to MLflow
        if mlflow.active_run():
            mlflow.log_metric(f"fold_{fold}_val_acc", val_acc)
        
        # Save fold model if specified
        if config.get('save_fold_models', False):
            os.makedirs('models/saved', exist_ok=True)
            model_path = f'models/saved/model_fold_{fold}.pth'
            torch.save(model.state_dict(), model_path)
            
            # Log model to MLflow
            if mlflow.active_run():
                mlflow.pytorch.log_model(model, f"model_fold_{fold}")
    
    # Compute overall CV performance
    mean_val_acc = np.mean(fold_val_accs)
    std_val_acc = np.std(fold_val_accs)
    
    print(f"\nCross-validation results:")
    print(f"Mean validation accuracy: {mean_val_acc:.4f} ± {std_val_acc:.4f}")
    print(f"Individual fold accuracies: {fold_val_accs}")
    
    # Log overall CV results to MLflow
    if mlflow.active_run():
        mlflow.log_metric("mean_val_acc", mean_val_acc)
        mlflow.log_metric("std_val_acc", std_val_acc)
    
    cv_results = {
        'mean_val_acc': mean_val_acc,
        'std_val_acc': std_val_acc,
        'fold_val_accs': fold_val_accs,
        'fold_histories': fold_histories
    }
    
    return config, cv_results

def train_final_model(config, data_path, test_fold=5):
    """
    Train the final model on all training data (excluding test fold)
    
    Args:
        config: Dictionary with training configuration
        data_path: Path to preprocessed data
        test_fold: Fold to reserve for final testing
    
    Returns:
        model: Trained model
        history: Training history
        test_results: Results on test set
    """
    # Load data
    data = np.load(data_path)
    spectrograms = data['spectrograms']
    labels = data['labels']
    folds = data['folds']
    
    # Convert to torch tensors
    spectrograms_tensor = torch.tensor(spectrograms, dtype=torch.float32).unsqueeze(1)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    # Get augmentation settings from config
    use_augmentation = config.get('use_augmentation', False)
    aug_strength = config.get('aug_strength', 0.3)
    
    # Prepare dataset with optional augmentation
    dataset = AugmentedDataset(
        spectrograms_tensor, 
        labels_tensor,
        use_augmentation=use_augmentation,
        aug_strength=aug_strength
    )
    
    # Create train/test split
    train_indices = np.where(folds != test_fold)[0]
    test_indices = np.where(folds == test_fold)[0]
    
    # Create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        dataset, batch_size=config['batch_size'], sampler=train_sampler
    )
    test_loader = DataLoader(
        dataset, batch_size=config['batch_size'], sampler=test_sampler
    )
    
    # Create model with optional dropout
    dropout_rate = config.get('dropout_rate', 0.0)  # Default to 0 (no dropout)
    model = create_resnet18(num_classes=len(np.unique(labels)), dropout_rate=dropout_rate)
    
    # Train the model on all training data
    print("\n--- Training Final Model ---")
    model, history, _ = train_one_cv_split(model, train_loader, test_loader, config)
    
    # Evaluate on test set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, true_labels, predictions = validate(model, test_loader, criterion, device)
    
    print(f"\nFinal Test Results:")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Log test results to MLflow
    if mlflow.active_run():
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_loss", test_loss)
    
    # Generate classification report
    try:
        # Load label mapping if available
        label_mapping_path = os.path.join(os.path.dirname(data_path), 'label_mapping.json')
        if os.path.exists(label_mapping_path):
            with open(label_mapping_path, 'r') as f:
                label_mapping = json.load(f)
            
            # Convert numeric labels to category names for the report
            target_names = [label_mapping[str(i)] for i in range(len(np.unique(labels)))]
        else:
            target_names = None
            
        # Print classification report
        report = classification_report(true_labels, predictions, target_names=target_names, output_dict=True)
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions, target_names=target_names))
        
        # Log classification report to MLflow
        if mlflow.active_run():
            # Convert the report dict to metrics that MLflow can log
            for class_label, metrics in report.items():
                if isinstance(metrics, dict):  # Skip the averages
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(f"{class_label}_{metric_name}", value)
            
            # Log average metrics
            for avg_type in ['macro avg', 'weighted avg']:
                if avg_type in report:
                    for metric_name, value in report[avg_type].items():
                        mlflow.log_metric(f"{avg_type.replace(' ', '_')}_{metric_name}", value)
    except Exception as e:
        print(f"Error generating classification report: {e}")
    
    # Save the model
    os.makedirs('models/saved', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'models/saved/final_model_{timestamp}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Log model to MLflow
    if mlflow.active_run():
        mlflow.pytorch.log_model(model, "final_model")
        mlflow.log_artifact(model_path)
    
    # Plot confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.tight_layout()
    
    os.makedirs('visualizations', exist_ok=True)
    cm_path = f'visualizations/confusion_matrix_{timestamp}.png'
    plt.savefig(cm_path)
    
    # Log confusion matrix to MLflow
    if mlflow.active_run():
        mlflow.log_artifact(cm_path)
    
    # Save test results
    test_results = {
        'test_acc': test_acc,
        'test_loss': test_loss,
        'true_labels': true_labels,
        'predictions': predictions,
    }
    
    return model, history, test_results

def main():
    """
    Main function to run the training process
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train audio classification model')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--mode', type=str, default='cv', choices=['cv', 'final'],
                        help='Training mode: cv (cross-validation) or final (final model)')
    args = parser.parse_args()
    
    # Default configuration
    config = {
        'batch_size': 32,
        'num_epochs': 5,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'use_lr_scheduler': True,
        'save_fold_models': False
    }
    
    # Load configuration from file if provided
    if args.config:
        # Check if this is a relative path without directory prefix
        config_path = args.config
        if not os.path.dirname(args.config) and not os.path.isfile(args.config):
            # Try to find it in the configs directory
            configs_path = os.path.join('configs', args.config)
            if os.path.isfile(configs_path):
                config_path = configs_path
        
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                config.update(loaded_config)
                print(f"Loaded configuration from {config_path}")
                print(json.dumps(config, indent=2))
        except Exception as e:
            print(f"Error loading configuration: {e}")
    
    # Data path
    data_path = 'data/preprocessed/esc50_preprocessed.npz'
    
    # Choose the mode
    mode = args.mode
    
    # Set up MLflow with more specific experiment naming
    if args.mode == 'cv':
        experiment_name = "esc50_cross_validation"
    elif args.mode == 'final':
        experiment_name = "esc50_final_model"
    else:
        experiment_name = "esc50_audio_classification"

    # You could also include configuration details in the experiment name
    if args.config and os.path.exists(args.config):
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        experiment_name = f"{experiment_name}_{config_name}"

    # Create or get the experiment
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    
    mlflow.set_experiment(experiment_name)
    
    # Start a new MLflow run
    with mlflow.start_run(run_name=f"{mode}_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        for key, value in config.items():
            mlflow.log_param(key, value)
        
        mlflow.log_param("mode", mode)
        mlflow.log_param("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        if mode == 'cv':
            # Perform cross-validation
            _, cv_results = cross_validation(config, data_path)
            
            # Plot learning curves from cross-validation
            plt.figure(figsize=(15, 12))
            
            # Create a grid of subplots - 2 metrics (loss, acc) × 4 folds
            for i, history in enumerate(cv_results['fold_histories']):
                # Loss plot for this fold
                plt.subplot(4, 2, i*2+1)
                plt.plot(history['train_loss'], label=f'Training Loss')
                plt.plot(history['val_loss'], label=f'Validation Loss')
                plt.title(f'Fold {i+1} Loss Curves')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                
                # Accuracy plot for this fold
                plt.subplot(4, 2, i*2+2)
                plt.plot(history['train_acc'], label=f'Training Accuracy')
                plt.plot(history['val_acc'], label=f'Validation Accuracy')
                plt.title(f'Fold {i+1} Accuracy Curves')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
            
            plt.tight_layout()
            os.makedirs('visualizations', exist_ok=True)
            curves_path = 'visualizations/cv_learning_curves.png'
            plt.savefig(curves_path)
            
            # Log the learning curves to MLflow
            mlflow.log_artifact(curves_path)
            
        elif mode == 'final':
            # Train final model
            model, history, test_results = train_final_model(config, data_path)
            
            # Plot learning curves from final training
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(history['train_loss'], label='Train Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.title('Loss Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history['train_acc'], label='Train Accuracy')
            plt.plot(history['val_acc'], label='Validation Accuracy')
            plt.title('Accuracy Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.tight_layout()
            os.makedirs('visualizations', exist_ok=True)
            curves_path = 'visualizations/final_learning_curves.png'
            plt.savefig(curves_path)
            
            # Log the learning curves to MLflow
            mlflow.log_artifact(curves_path)

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
