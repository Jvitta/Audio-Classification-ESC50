import librosa
import librosa.display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def perform_eda(audio_path, metadata_path):
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    
    # Load an example audio file
    y, sr = librosa.load(audio_path)
    
    # Create and display spectrogram
    plt.figure(figsize=(12, 8))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.show()
    
    # Display dataset statistics
    print("Number of classes:", len(metadata['category'].unique()))
    print("Number of samples:", len(metadata))
    print("Class distribution:\n", metadata['category'].value_counts())

# Example usage
perform_eda('data/audio/1-137-A-32.wav', 'data/metadata/esc50.csv')

