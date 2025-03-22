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

def train_single_fold(model, train_loader, val_loader, config, fold=None):
    """
    Train a model on a single fold
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Dictionary with training configuration
        fold: Fold number (for logging)
    
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
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
    }
    
    best_val_acc = 0.0
    best_model_state = None
    
    fold_str = f" (Fold {fold})" if fold is not None else ""
    print(f"Starting training{fold_str}...")
    
    for epoch in range(num_epochs):
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation phase
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        # Update learning rate if using scheduler
        if config.get('use_lr_scheduler', False):
            scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}{fold_str} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, history, best_val_acc

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
    
    # Prepare dataset
    dataset = TensorDataset(spectrograms_tensor, labels_tensor)
    
    # Track results across folds
    fold_val_accs = []
    fold_histories = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Cross-validation loop
    for fold in range(1, num_folds + 1):
        print(f"\n--- Fold {fold} ---")
        
        # Create train/validation split based on folds
        # The test_fold (typically fold 5) is completely excluded
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
        
        # Create model
        model = create_resnet18(num_classes=len(np.unique(labels)))
        
        # Train on this fold
        model, history, val_acc = train_single_fold(
            model, train_loader, val_loader, config, fold=fold
        )
        
        # Save results
        fold_val_accs.append(val_acc)
        fold_histories.append(history)
        
        # Save fold model if specified
        if config.get('save_fold_models', False):
            os.makedirs('models/saved', exist_ok=True)
            torch.save(model.state_dict(), f'models/saved/model_fold_{fold}.pth')
    
    # Compute overall CV performance
    mean_val_acc = np.mean(fold_val_accs)
    std_val_acc = np.std(fold_val_accs)
    
    print(f"\nCross-validation results:")
    print(f"Mean validation accuracy: {mean_val_acc:.4f} ± {std_val_acc:.4f}")
    print(f"Individual fold accuracies: {fold_val_accs}")
    
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
    
    # Prepare dataset
    dataset = TensorDataset(spectrograms_tensor, labels_tensor)
    
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
    
    # Create model
    model = create_resnet18(num_classes=len(np.unique(labels)))
    
    # Train the model on all training data
    print("\n--- Training Final Model ---")
    model, history, _ = train_single_fold(model, train_loader, test_loader, config)
    
    # Evaluate on test set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, true_labels, predictions = validate(model, test_loader, criterion, device)
    
    print(f"\nFinal Test Results:")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
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
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions, target_names=target_names))
    except Exception as e:
        print(f"Error generating classification report: {e}")
    
    # Save the model
    os.makedirs('models/saved', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'models/saved/final_model_{timestamp}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Plot confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.tight_layout()
    
    os.makedirs('visualizations', exist_ok=True)
    plt.savefig(f'visualizations/confusion_matrix_{timestamp}.png')
    
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
    # Configuration
    config = {
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'use_lr_scheduler': True,
        'save_fold_models': False
    }
    
    # Data path
    data_path = 'data/preprocessed/esc50_preprocessed.npz'
    
    # Choose the mode
    mode = 'cv'  # 'cv' for cross-validation, 'final' for final model training
    
    if mode == 'cv':
        # Perform cross-validation
        _, cv_results = cross_validation(config, data_path)
        
        # Plot learning curves from cross-validation
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        for i, history in enumerate(cv_results['fold_histories']):
            plt.plot(history['train_loss'], label=f'Fold {i+1} Train')
            plt.plot(history['val_loss'], label=f'Fold {i+1} Val')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        for i, history in enumerate(cv_results['fold_histories']):
            plt.plot(history['train_acc'], label=f'Fold {i+1} Train')
            plt.plot(history['val_acc'], label=f'Fold {i+1} Val')
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        os.makedirs('visualizations', exist_ok=True)
        plt.savefig('visualizations/cv_learning_curves.png')
        
    elif mode == 'final':
        # Train final model
        model, history, test_results = train_final_model(config, data_path)
        
        # Plot learning curves from final training
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Test Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Accuracy')
        plt.plot(history['val_acc'], label='Test Accuracy')
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        os.makedirs('visualizations', exist_ok=True)
        plt.savefig('visualizations/final_learning_curves.png')

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
