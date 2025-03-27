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
│   ├── raw/                    # Original ESC-50 audio files
│   └── preprocessed/           # Preprocessed mel spectrograms (NPZ format)
├── models/
│   ├── resnet.py              # ResNet18 model architecture
│   └── saved_models/          # Trained model checkpoints
├── utils/
│   └── audio_preprocessing.py  # Audio preprocessing utilities
├── model_summary.py           # Script for model architecture summary
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
git clone [repository-url]
cd d604-advanced-analytics
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install additional visualization packages:
```bash
pip install torchsummary torchviz
```

## Data Preparation
The audio preprocessing pipeline includes:
1. Loading audio files (44.1kHz)
2. Standardizing clip lengths (5 seconds)
3. Converting to mel spectrograms
4. Applying logarithmic scaling
5. Normalizing features
6. Storing in NPZ format

## Model Training
The model uses:
- 5-fold cross-validation
- First 4 folds for training/validation
- 5th fold as holdout test set
- Learning rate optimization
- Dropout regularization

## Usage

### Generate Model Summary
```bash
python model_summary.py
```

### View Model Architecture
The model architecture can be visualized using:
```python
from models.resnet import create_resnet18
model = create_resnet18()
print(model)
```

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

## License
[Specify License]

## Acknowledgments
- ESC-50 dataset creators
- ResNet architecture authors
- [Other acknowledgments]

## Contact
Jack Vittimberga [jvittimberga@gmail.com](mailto:jvittimberga@gmail.com)
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
│   ├── raw/                    # Original ESC-50 audio files
│   └── preprocessed/           # Preprocessed mel spectrograms (NPZ format)
├── models/
│   ├── resnet.py              # ResNet18 model architecture
│   └── saved_models/          # Trained model checkpoints
├── utils/
│   └── audio_preprocessing.py  # Audio preprocessing utilities
├── model_summary.py           # Script for model architecture summary
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
git clone [repository-url]
cd d604-advanced-analytics
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install additional visualization packages:
```bash
pip install torchsummary torchviz
```

## Data Preparation
The audio preprocessing pipeline includes:
1. Loading audio files (44.1kHz)
2. Standardizing clip lengths (5 seconds)
3. Converting to mel spectrograms
4. Applying logarithmic scaling
5. Normalizing features
6. Storing in NPZ format

## Model Training
The model uses:
- 5-fold cross-validation
- First 4 folds for training/validation
- 5th fold as holdout test set
- Learning rate optimization
- Dropout regularization

## Usage

### Generate Model Summary
```bash
python model_summary.py
```

### View Model Architecture
The model architecture can be visualized using:
```python
from models.resnet import create_resnet18
model = create_resnet18()
print(model)
```

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

## License
[Specify License]

## Acknowledgments
- ESC-50 dataset creators
- ResNet architecture authors
- [Other acknowledgments]

## Contact
Jack Vittimberga [jvittimberga@gmail.com](mailto:jvittimberga@gmail.com)
