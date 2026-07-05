# Import neural network module from PyTorch
import torch.nn as nn

# Define a standard 1D CNN for variable star classification
class LightCurveCNN(nn.Module):
    # Initialize the CNN layer definitions
    def __init__(self, num_classes=4):
        # Call base constructor
        super(LightCurveCNN, self).__init__()
        
        # Conv block 1:
        # Input channel: 1 (magnitude sequence)
        # Output channels: 16 (features)
        # Kernel size: 3
        # Padding: 1 (preserves sequence length)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        # MaxPool1d: halves the sequence length
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        # Activation ReLU
        self.relu1 = nn.ReLU()
        
        # Conv block 2:
        # Input channels: 16
        # Output channels: 32 (extracts higher-level sequence features)
        # Kernel size: 3
        # Padding: 1
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # MaxPool1d: halves sequence length again
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        # Activation ReLU
        self.relu2 = nn.ReLU()
        
        # Conv block 3:
        # Input channels: 32
        # Output channels: 64
        # Kernel size: 3
        # Padding: 1
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # MaxPool1d: halves sequence length a third time
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        # Activation ReLU
        self.relu3 = nn.ReLU()
        
        # Adaptive pooling to collapse variable sequence lengths down to fixed size 8
        # Output shape is [Batch, 64, 8]
        self.adaptive_pool = nn.AdaptiveAvgPool1d(8)
        
        # Final linear classifier layer
        # Input features: 64 channels * 8 sequence length = 512 flat features
        # Output features: num_classes (4 classes corresponding to variable star types)
        self.fc = nn.Linear(in_features=64 * 8, out_features=num_classes)

    # Define the forward pass logic
    def forward(self, x):
        # Pass input through conv block 1
        x = self.relu1(self.pool1(self.conv1(x)))
        # Pass through conv block 2
        x = self.relu2(self.pool2(self.conv2(x)))
        # Pass through conv block 3
        x = self.relu3(self.pool3(self.conv3(x)))
        # Apply adaptive average pool
        x = self.adaptive_pool(x)
        
        # Flatten sequence representation into flat vector per batch
        # Shape changes from [Batch, 64, 8] to [Batch, 512]
        x = x.view(x.size(0), -1)
        
        # Pass flat features to linear classifier to compute class logits
        x = self.fc(x)
        
        # Return logits
        return x
