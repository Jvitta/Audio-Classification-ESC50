import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18SmallForAudio(nn.Module):
    def __init__(self, num_classes=50, in_channels=1, dropout_rate=0.0):
        super(ResNet18SmallForAudio, self).__init__()
        """
        ResNet18 model adapted for audio spectrograms with reduced channel capacity.
        
        Args:
            num_classes (int): Number of output classes (default: 50 for ESC-50)
            in_channels (int): Number of input channels (default: 1 for mono audio)
            dropout_rate (float): Dropout probability (default: 0.0 for no dropout)
        
        Input shape:
            (batch_size, 1, 64, 501) - (batch, channels, mel_bins, time_frames)
        """
        # Initial layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet blocks with reduced channels
        self.layer1 = self._make_layer(32, 32, 2, stride=1)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        self.layer4 = self._make_layer(128, 256, 2, stride=2)
        
        # Dropout for regularization
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
        # Global average pooling and final classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)  # Reduced final dimension
        
        # Initialize weights
        self._initialize_weights()
        
        # Flag for logging feature statistics
        self.log_features = False

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        if hasattr(self, 'log_feature_stats'):
            self.log_feature_stats("Initial conv", x)
        x = self.maxpool(x)
        
        # ResNet blocks
        x = self.layer1(x)
        if hasattr(self, 'log_feature_stats'):
            self.log_feature_stats("Layer 1", x)
        x = self.layer2(x)
        if hasattr(self, 'log_feature_stats'):
            self.log_feature_stats("Layer 2", x)
        x = self.layer3(x)
        if hasattr(self, 'log_feature_stats'):
            self.log_feature_stats("Layer 3", x)
        x = self.layer4(x)
        if hasattr(self, 'log_feature_stats'):
            self.log_feature_stats("Layer 4", x)
        
        # Global average pooling and classification
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if hasattr(self, 'log_feature_stats'):
            self.log_feature_stats("Pre-FC", x.unsqueeze(-1).unsqueeze(-1))
        
        # Apply dropout before the final fully connected layer
        if self.dropout is not None:
            x = self.dropout(x)
            
        x = self.fc(x)
        
        return x

def create_resnet18(num_classes=50, pretrained=False, dropout_rate=0.0):
    """
    Creates a ResNet18 model for audio classification with reduced channel capacity.
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Flag for using pretrained weights (not implemented yet)
        dropout_rate (float): Dropout probability for regularization (default: 0.0)
    
    Returns:
        model: The reduced-capacity ResNet18 model
    """
    model = ResNet18SmallForAudio(num_classes=num_classes, dropout_rate=dropout_rate)
    
    if pretrained:
        # Future implementation could load pretrained weights
        # Potentially adapt ImageNet weights or audio-specific pretrained weights
        pass
    
    return model
