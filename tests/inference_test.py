import torch
import numpy as np
from models.resnet import ResNet18ForAudio as ResNet18Large
from models.resnet_small import ResNet18SmallForAudio as ResNet18Small
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
import os
import matplotlib.pyplot as plt

class FeatureStatistics:
    def __init__(self):
        self.layer_stats = {
            'means_min': [],
            'means_max': [],
            'stds_min': [],
            'stds_max': [],
            'active_neurons': [],
            'layer_names': []
        }
    
    def add_stats(self, name, means, stds, active):
        self.layer_stats['layer_names'].append(name)
        self.layer_stats['means_min'].append(means.min().item())
        self.layer_stats['means_max'].append(means.max().item())
        self.layer_stats['stds_min'].append(stds.min().item())
        self.layer_stats['stds_max'].append(stds.max().item())
        self.layer_stats['active_neurons'].append(active.mean().item())

def test_model_features(model, dataloader, model_name, num_samples=5):
    """Test model feature statistics on a few validation samples."""
    model.eval()
    model.log_features = True
    device = next(model.parameters()).device
    
    print(f"\nTesting {model_name} architecture:")
    print("=" * 50)
    
    all_sample_stats = []
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= num_samples:
                break
            
            inputs = inputs.to(device)
            print(f"\nSample {i+1}:")
            print("-" * 30)
            
            # Create statistics collector for this sample
            sample_stats = FeatureStatistics()
            
            # Add logging function to the model instance
            def log_feature_stats(name, tensor):
                if model.log_features and not model.training:
                    if name == "Pre-FC":
                        # For Pre-FC layer, don't calculate std across spatial dimensions
                        means = tensor.mean(dim=0)
                        stds = tensor.std(dim=0)
                        active = (tensor > 0).float().mean(dim=0)
                    else:
                        means = tensor.mean(dim=(0,2,3))
                        stds = tensor.std(dim=(0,2,3))
                        active = (tensor > 0).float().mean(dim=(0,2,3))
                    
                    print(f"\n{name} stats:")
                    print(f"Mean activation range: {means.min():.3f} to {means.max():.3f}")
                    print(f"Std range: {stds.min():.3f} to {stds.max():.3f}")
                    print(f"Active neurons: {active.mean():.1%}")
                    sample_stats.add_stats(name, means, stds, active)
            
            # Attach the logging function to the model instance
            model.log_feature_stats = log_feature_stats
            
            _ = model(inputs)
            
            all_sample_stats.append(sample_stats)
    
    return all_sample_stats

def plot_feature_statistics(large_stats, small_stats, save_path='visualizations'):
    """Plot comparative feature statistics between models."""
    os.makedirs(save_path, exist_ok=True)
    
    # Average stats across samples
    def average_stats(all_stats):
        avg_stats = FeatureStatistics()
        n_samples = len(all_stats)
        first_sample = all_stats[0]
        
        for key in ['means_min', 'means_max', 'stds_min', 'stds_max', 'active_neurons']:
            values = np.array([s.layer_stats[key] for s in all_stats])
            avg_stats.layer_stats[key] = np.mean(values, axis=0)
        
        avg_stats.layer_stats['layer_names'] = first_sample.layer_stats['layer_names']
        return avg_stats
    
    large_avg = average_stats(large_stats)
    small_avg = average_stats(small_stats)
    
    # Create plots
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot mean activation ranges
    ax = axes[0]
    x = range(len(large_avg.layer_stats['layer_names']))
    ax.fill_between(x, large_avg.layer_stats['means_min'], large_avg.layer_stats['means_max'], 
                   alpha=0.3, label='Large Model Range')
    ax.fill_between(x, small_avg.layer_stats['means_min'], small_avg.layer_stats['means_max'], 
                   alpha=0.3, label='Small Model Range')
    ax.set_xticks(x)
    ax.set_xticklabels(large_avg.layer_stats['layer_names'], rotation=45)
    ax.set_title('Mean Activation Ranges Across Layers')
    ax.set_ylabel('Activation Value')
    ax.legend()
    ax.grid(True)
    
    # Plot std ranges
    ax = axes[1]
    ax.fill_between(x, large_avg.layer_stats['stds_min'], large_avg.layer_stats['stds_max'], 
                   alpha=0.3, label='Large Model Range')
    ax.fill_between(x, small_avg.layer_stats['stds_min'], small_avg.layer_stats['stds_max'], 
                   alpha=0.3, label='Small Model Range')
    ax.set_xticks(x)
    ax.set_xticklabels(large_avg.layer_stats['layer_names'], rotation=45)
    ax.set_title('Standard Deviation Ranges Across Layers')
    ax.set_ylabel('Standard Deviation')
    ax.legend()
    ax.grid(True)
    
    # Plot active neurons
    ax = axes[2]
    ax.plot(x, large_avg.layer_stats['active_neurons'], 'o-', label='Large Model')
    ax.plot(x, small_avg.layer_stats['active_neurons'], 'o-', label='Small Model')
    ax.set_xticks(x)
    ax.set_xticklabels(large_avg.layer_stats['layer_names'], rotation=45)
    ax.set_title('Proportion of Active Neurons Across Layers')
    ax.set_ylabel('Proportion Active')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'feature_statistics_comparison.png'))
    plt.close()

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load preprocessed data
    data_path = 'data/preprocessed/esc50_preprocessed.npz'
    data = np.load(data_path)
    spectrograms = data['spectrograms']
    labels = data['labels']
    folds = data['folds']

    # Convert to torch tensors
    spectrograms_tensor = torch.tensor(spectrograms, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Create dataset
    dataset = TensorDataset(spectrograms_tensor, labels_tensor)

    # Use fold 4 for validation (same as in train.py)
    validation_fold = 4
    val_indices = np.where(folds == validation_fold)[0]
    val_sampler = SubsetRandomSampler(val_indices)

    # Create validation dataloader
    val_loader = DataLoader(
        dataset, 
        batch_size=1,  # Use batch size 1 to analyze individual samples
        sampler=val_sampler
    )
    
    # Initialize both models
    large_model = ResNet18Large(num_classes=len(np.unique(labels)))
    small_model = ResNet18Small(num_classes=len(np.unique(labels)))
    
    # Load trained weights if available
    model_path = "models/best_model.pth"  # adjust path as needed
    if os.path.exists(model_path):
        print("Loading trained weights...")
        state_dict = torch.load(model_path, map_location=device)
        large_model.load_state_dict(state_dict)
    
    # Move models to device
    large_model = large_model.to(device)
    small_model = small_model.to(device)
    
    # Test both architectures and collect statistics
    print("\nComparing feature statistics between architectures...")
    large_stats = test_model_features(large_model, val_loader, "Original ResNet18 (Large)")
    print("\n" + "=" * 80 + "\n")  # Separator
    small_stats = test_model_features(small_model, val_loader, "Reduced ResNet18 (Small)")
    
    # Create visualizations
    print("\nGenerating visualization plots...")
    plot_feature_statistics(large_stats, small_stats)
    print("Plots saved in visualizations/feature_statistics_comparison.png")

if __name__ == "__main__":
    main() 