import librosa
import librosa.display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def analyze_raw_audio(audio_path, metadata_path):
    """Analyze raw audio files from the ESC-50 dataset"""
    # Make sure visualization directory exists
    os.makedirs('visualizations', exist_ok=True)
    
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    
    # Load an example audio file
    y, sr = librosa.load(audio_path, sr=44100)
    
    # Create and display regular spectrogram
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, y_axis='linear', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear Spectrogram')
    
    # Create and display mel spectrogram with our preprocessing parameters
    plt.subplot(3, 1, 2)
    # Calculate window size and hop length in samples
    win_length = int(0.025 * sr)  # 25ms window
    hop_length = int(0.010 * sr)  # 10ms hop
    n_fft = win_length
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64,
                                       n_fft=n_fft, win_length=win_length,
                                       hop_length=hop_length, fmax=sr/2)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram (Our Parameters)')
    
    # Display waveform
    plt.subplot(3, 1, 3)
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    plt.tight_layout()
    plt.savefig('visualizations/raw_audio_analysis.png')
    plt.close()
    
    # Display dataset statistics
    print("\n--- Dataset Statistics ---")
    print("Number of classes:", len(metadata['category'].unique()))
    print("Number of samples:", len(metadata))
    print("Class distribution:\n", metadata['category'].value_counts())
    
    # Print audio file information
    filename = os.path.basename(audio_path)
    file_info = metadata[metadata['filename'] == filename]
    if not file_info.empty:
        print(f"\n--- Audio File Information ---")
        print(f"Filename: {filename}")
        print(f"Category: {file_info['category'].values[0]}")
        print(f"Target label: {file_info['target'].values[0]}")
        print(f"Fold: {file_info['fold'].values[0]}")
    
    print(f"\n--- Audio Properties ---")
    print(f"Duration: {len(y)/sr:.2f} seconds")
    print(f"Sample rate: {sr} Hz")
    print(f"Total samples: {len(y)}")
    print(f"Window size: {win_length} samples ({win_length/sr*1000:.1f} ms)")
    print(f"Hop length: {hop_length} samples ({hop_length/sr*1000:.1f} ms)")
    print(f"Spectrogram shape: {S.shape} (mel bins × time frames)")


def analyze_processed_data(preprocessed_path, label_mapping_path):
    """Analyze the preprocessed spectrograms"""
    # Load preprocessed data
    data = np.load(preprocessed_path)
    spectrograms = data['spectrograms']
    labels = data['labels']
    folds = data['folds']
    
    # Load label mapping
    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)
    
    # Convert numeric labels to category names
    label_names = [label_mapping[str(label)] for label in labels]
    
    # Display basic stats
    print("\n--- Preprocessed Data Statistics ---")
    print(f"Number of samples: {len(spectrograms)}")
    print(f"Spectrogram shape: {spectrograms[0].shape}")
    print(f"Number of unique labels: {len(np.unique(labels))}")
    print(f"Number of folds: {len(np.unique(folds))}")
    
    # Display example spectrograms from different categories
    plt.figure(figsize=(15, 10))
    categories_to_show = min(5, len(np.unique(labels)))
    
    for i in range(categories_to_show):
        # Get first sample of this category
        category_samples = np.where(labels == i)[0]
        if len(category_samples) > 0:
            sample_idx = category_samples[0]
            
            plt.subplot(categories_to_show, 1, i+1)
            plt.imshow(spectrograms[sample_idx], aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(format='%.2f')
            plt.title(f'Category: {label_mapping[str(i)]} (Label {i})')
            plt.ylabel('Mel Frequency Bin')
            plt.xlabel('Time Frame')
    
    plt.tight_layout()
    plt.savefig('visualizations/processed_spectrograms.png')
    plt.close()
    
    # Check value ranges of spectrograms
    min_val = np.min([np.min(spec) for spec in spectrograms])
    max_val = np.max([np.max(spec) for spec in spectrograms])
    mean_val = np.mean([np.mean(spec) for spec in spectrograms])
    std_val = np.std([np.std(spec) for spec in spectrograms])
    
    print("\n--- Spectrogram Value Statistics ---")
    print(f"Min value: {min_val}")
    print(f"Max value: {max_val}")
    print(f"Mean value: {mean_val}")
    print(f"Std deviation: {std_val}")
    
    # Verify normalization (should be 0-1 for all spectrograms)
    print(f"All spectrograms normalized to [0,1] range: {min_val >= 0 and max_val <= 1}")
    
    # Dimension check
    print("\n--- Dimension Check ---")
    spectrogram_shape = spectrograms[0].shape
    print(f"Spectrogram shape: {spectrogram_shape} (mel bins × time frames)")
    all_same_shape = all(spec.shape == spectrogram_shape for spec in spectrograms)
    print(f"All spectrograms have the same shape: {all_same_shape}")
    if not all_same_shape:
        shapes = set(spec.shape for spec in spectrograms)
        print(f"Different shapes found: {shapes}")
    
    # Plot class distribution
    plt.figure(figsize=(15, 6))
    class_counts = pd.Series(label_names).value_counts().sort_index()
    class_counts.plot(kind='bar')
    plt.title('Class Distribution in Preprocessed Dataset')
    plt.xlabel('Sound Category')
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('visualizations/class_distribution.png')
    plt.close()
    
    # Fold distribution
    plt.figure(figsize=(10, 5))
    fold_counts = pd.Series(folds).value_counts().sort_index()
    fold_counts.plot(kind='bar')
    plt.title('Fold Distribution')
    plt.xlabel('Fold Number')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('visualizations/fold_distribution.png')
    plt.close()


def main():
    # File paths
    audio_example = 'data/audio/1-137-A-32.wav'  # Example audio file
    metadata_path = 'data/meta/esc50.csv'
    preprocessed_path = 'data/preprocessed/esc50_preprocessed.npz'
    label_mapping_path = 'data/preprocessed/label_mapping.json'
    
    # Analyze raw audio
    print("Analyzing raw audio...")
    analyze_raw_audio(audio_example, metadata_path)
    
    # Check if preprocessed data exists
    if os.path.exists(preprocessed_path) and os.path.exists(label_mapping_path):
        print("\nAnalyzing preprocessed data...")
        analyze_processed_data(preprocessed_path, label_mapping_path)
    else:
        print("\nPreprocessed data not found. Please run audio_preprocessing.py first.")


if __name__ == "__main__":
    main()

