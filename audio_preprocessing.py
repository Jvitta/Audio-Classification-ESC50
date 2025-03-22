import librosa
import librosa.display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

def extract_spectrogram(audio_data, sr, n_mels=64):
    # Calculate window size and hop length in samples
    win_length = int(0.025 * sr)  # 25ms window
    hop_length = int(0.010 * sr)  # 10ms hop
    n_fft = win_length 
    
    # Create mel spectrogram
    mel_spect = librosa.feature.melspectrogram(
        y=audio_data,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        fmax=sr/2
    )
    
    # Convert to log scale
    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
    return mel_spect_db

def load_audio_signal(file_path, sr=44100):
    # Load audio file
    audio_data, sr = librosa.load(file_path, sr=sr)
    return audio_data, sr

def pad_audio(audio_data, target_length):
    # If audio is shorter than target length, pad with zeros
    if len(audio_data) < target_length:
        padding = target_length - len(audio_data)
        audio_data = np.pad(audio_data, (0, padding), mode='constant')
    # If audio is longer, truncate
    else:
        audio_data = audio_data[:target_length]
    return audio_data

def normalize_spectrogram(spectrogram):
    # Min-max normalization
    min_val = spectrogram.min()
    max_val = spectrogram.max()
    normalized = (spectrogram - min_val) / (max_val - min_val)
    return normalized

def preprocess_dataset(data_dir, metadata_path, save_dir='data/preprocessed'):
    """
    Prepare and save the ESC-50 dataset for training.
    
    Args:
        data_dir (str): Directory containing the audio files
        metadata_path (str): Path to the esc50.csv metadata file
        save_dir (str): Directory to save preprocessed data
    
    Returns:
        spectrograms (np.array): Array of normalized mel spectrograms
        labels (np.array): Array of numeric labels [0-49]
        folds (np.array): Array of fold numbers for cross-validation
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    
    # Create label mapping based on actual target-category pairs
    label_mapping = metadata[['target', 'category']].drop_duplicates().set_index('target')['category'].to_dict()
    with open(os.path.join(save_dir, 'label_mapping.json'), 'w') as f:
        json.dump(label_mapping, f, indent=4)
    
    spectrograms = []
    labels = []
    folds = []
    
    for idx, row in metadata.iterrows():
        try:
            # Load audio
            audio_path = f"{data_dir}/{row['filename']}"
            audio_data, sr = load_audio_signal(audio_path)
            
            # Pad audio - ESC-50 files are 5 seconds at 44.1kHz
            target_length = int(5 * sr)  # 5 seconds * 44100 Hz
            padded_audio = pad_audio(audio_data, target_length)
            
            # Extract and normalize spectrogram
            spec = extract_spectrogram(padded_audio, sr)
            norm_spec = normalize_spectrogram(spec)
            
            spectrograms.append(norm_spec)
            labels.append(row['target'])
            folds.append(row['fold'])
            
        except Exception as e:
            print(f"Error processing file {row['filename']}: {str(e)}")
            continue
    
    # Convert lists to numpy arrays
    spectrograms = np.array(spectrograms)
    labels = np.array(labels)
    folds = np.array(folds)
    
    # Save preprocessed data
    np.savez(
        os.path.join(save_dir, 'esc50_preprocessed.npz'),
        spectrograms=spectrograms,
        labels=labels,
        folds=folds
    )
    
    # Save metadata about the preprocessing
    preprocessing_info = {
        'num_samples': len(spectrograms),
        'spectrogram_shape': spectrograms[0].shape,
        'num_classes': len(np.unique(labels)),
        'class_distribution': {label_mapping[k]: v for k, v in pd.Series(labels).value_counts().to_dict().items()},
        'preprocessing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(save_dir, 'preprocessing_info.json'), 'w') as f:
        json.dump(preprocessing_info, f, indent=4)
    
    return spectrograms, labels, folds

if __name__ == "__main__":
    data_dir = "data/audio"
    metadata_path = "data/meta/esc50.csv"
    save_dir = "data/preprocessed"
    
    spectrograms, labels, folds = preprocess_dataset(
        data_dir, metadata_path, save_dir
    )
    print(f"Preprocessed data saved to {save_dir}")
    print(f"Spectrogram shape: {spectrograms.shape}")
    


