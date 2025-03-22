import librosa
import librosa.display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def extract_spectrogram(audio_data, sr, n_mels=128):
    # Create mel spectrogram
    mel_spect = librosa.feature.melspectrogram(
        y=audio_data,
        sr=sr,
        n_mels=n_mels,
        fmax=sr/2
    )
    
    # Convert to log scale
    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
    return mel_spect_db

def load_audio_signal(file_path, sr=22050):
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

def extract_features(audio_data, sr):
    features = {}
    
    # Mel-frequency cepstral coefficients
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    features['mfcc'] = np.mean(mfccs.T, axis=0)
    
    # Spectral centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
    features['spectral_centroid'] = np.mean(spectral_centroids)
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
    features['zcr'] = np.mean(zcr)
    
    return features