# Import neural network module from PyTorch
import torch.nn as nn

# Define an improved 1D CNN for variable star classification
class LightCurveCNN(nn.Module):
    # Initialize the CNN layer definitions
    def __init__(self, num_classes=4):
        # Call base constructor
        super(LightCurveCNN, self).__init__()
        
        # Conv block 1:
        # Increase filters to 32, add Batch Normalization
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.relu1 = nn.ReLU()
        
        # Conv block 2:
        # Increase filters to 64, add Batch Normalization
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.relu2 = nn.ReLU()
        
        # Conv block 3:
        # Increase filters to 128, add Batch Normalization
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.relu3 = nn.ReLU()
        
        # Adaptive pooling to collapse sequence representation to size 8
        self.adaptive_pool = nn.AdaptiveAvgPool1d(8)
        
        # Add dropout layer to prevent overfitting
        self.dropout = nn.Dropout(p=0.3)
        
        # Final linear classifier layer
        # Input features: 128 channels * 8 sequence length = 1024 flat features
        self.fc = nn.Linear(in_features=128 * 8, out_features=num_classes)

    # Define the forward pass logic
    def forward(self, x):
        # Pass input through conv block 1
        x = self.relu1(self.pool1(self.bn1(self.conv1(x))))
        # Pass through conv block 2
        x = self.relu2(self.pool2(self.bn2(self.conv2(x))))
        # Pass through conv block 3
        x = self.relu3(self.pool3(self.bn3(self.conv3(x))))
        # Apply adaptive average pool
        x = self.adaptive_pool(x)
        
        # Flatten sequence representation
        x = x.view(x.size(0), -1)
        
        # Apply dropout to flat features
        x = self.dropout(x)
        
        # Pass features to linear classifier to compute class logits
        x = self.fc(x)
        
        # Return logits
        return x

