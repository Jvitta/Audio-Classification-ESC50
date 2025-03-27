# ESC-50 Audio Classification with ResNet18

## Project Overview
This project implements an environmental sound classification system using a modified ResNet18 architecture. It processes the ESC-50 dataset, which contains 2000 environmental audio recordings across 50 classes, to create a robust audio classification model. Our model achieved a test accuracy of 73% on the held-out test set, approaching human-level performance (81.3%) on this challenging dataset.

## Dataset
The ESC-50 dataset consists of:
- 2000 environmental recordings (5-second clips)
- 50 semantic classes
- 40 examples per class
- 5-fold cross-validation setup
- All audio is recorded at 44.1kHz sampling rate

## Project Structure
```
d604-advanced-analytics/
├── data/
│   ├── audio/                    # Original ESC-50 audio files
│   └── preprocessed/           # Preprocessed mel spectrograms (NPZ format)
├── models/
│   ├── resnet.py              # ResNet18 model architecture
│   ├── resnet_small.py        # ResNet18 model with half the channels
│   └── saved/                 # Trained model checkpoints
├── audio_preprocessing.py     # Audio preprocessing utilities
├── train.py                   # Script for model training
├── hyperparameter_search.py   # Script for bayesian hyperparameter search and LRFinder
├── requirements.txt           # Project dependencies
└── README.md                  # This file
```

## Model Architecture
The project uses a modified ResNet18 architecture:
- Input: Mel spectrograms (1×64×501)
- 8 convolutional blocks in 4 layers
- Channel dimensions: 64 → 512
- Total parameters: ~11.2M
- Output: 50 class predictions

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/Jvitta/Audio-Classification-ESC50.git
cd d604-advanced-analytics
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preprocessing
The `audio_preprocessing.py` script handles the preparation of audio data:
```bash
python audio_preprocessing.py
```
This script:
- Loads audio files from the raw data directory
- Converts audio to mel spectrograms
- Applies normalization and preprocessing
- Saves processed data in NPZ format
- Generates preprocessing metadata and label mappings

### Model Training
The `train.py` script provides three training modes:

1. Cross-validation training:
```bash
python train.py --mode cv --config configs/bayesian_best_config.json
```

2. Validation training (using fold 4 as validation):
```bash
python train.py --mode validation --config configs/bayesian_best_config.json
```

3. Final model training (using fold 5 as test set):
```bash
python train.py --mode final --config configs/bayesian_best_config.json
```

The training script includes:
- MLflow integration for experiment tracking
- Learning rate scheduling
- Model checkpointing
- Performance visualization
- Confusion matrix generation
- Detailed classification reports

## Results
- Test Accuracy: 73% on held-out test set (fold 5)
- Validation Accuracy: 74% during training
- Performance Comparison:
  - Human listeners: 81.3% accuracy
  - Early ML approaches: ~65% accuracy
  - Our ResNet18 model: 73% accuracy
- Strong generalization evidenced by consistent validation and test performance

## Future Improvements
- Data augmentation techniques
- Ensemble methods
- Architecture modifications
- Hyperparameter optimization

## Dependencies
- PyTorch
- librosa
- numpy
- torchsummary
- torchviz
- matplotlib
- scikit-learn
- pandas
- tqdm

## License
This Project is licensed under the MIT license

## Acknowledgments
- ESC-50 dataset creators
- ResNet architecture authors
- [Other acknowledgments]

## Contact
Jack Vittimberga [jvittimberga@gmail.com](mailto:jvittimberga@gmail.com)