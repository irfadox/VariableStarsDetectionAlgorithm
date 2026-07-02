# Import the core neural network module from PyTorch
import torch.nn as nn

# Define a standard Convolutional Neural Network class extending nn.Module
class VariableStarCNN(nn.Module):
    # Initialize the network layers
    def __init__(self):
        # Invoke parent class constructor to initialize internal PyTorch module state
        super(VariableStarCNN, self).__init__()
        
        # Define the first convolutional layer
        # Input channel: 1 (grayscale FITS image tensor)
        # Output channels: 16 (16 features extracted)
        # Kernel size: 3 (3x3 spatial window filter)
        # Padding: 1 (preserves spatial resolution by padding borders)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        
        # Define the first MaxPool layer to reduce spatial dimensions
        # Kernel size: 2 (2x2 spatial window)
        # Stride: 2 (halves height and width dimensions)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Define the first activation function (Rectified Linear Unit)
        self.relu1 = nn.ReLU()
        
        # Define the second convolutional layer
        # Input channels: 16 (matching the output of the first conv layer)
        # Output channels: 32 (32 features extracted)
        # Kernel size: 3 (3x3 spatial window filter)
        # Padding: 1 (preserves spatial resolution by padding borders)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # Define the second MaxPool layer
        # Kernel size: 2 (2x2 spatial window)
        # Stride: 2 (halves spatial dimensions further)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Define the second activation function (Rectified Linear Unit)
        self.relu2 = nn.ReLU()
        
        # Define the third convolutional layer
        # Input channels: 32 (matching the output of the second conv layer)
        # Output channels: 64 (64 features extracted)
        # Kernel size: 3 (3x3 spatial window filter)
        # Padding: 1 (preserves spatial resolution by padding borders)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Define the third MaxPool layer
        # Kernel size: 2 (2x2 spatial window)
        # Stride: 2 (halves spatial dimensions a third time)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Define the third activation function (Rectified Linear Unit)
        self.relu3 = nn.ReLU()
        
        # Define an Adaptive Average Pooling layer
        # Regardless of input image size, this pools feature maps down to exactly 4x4 spatial size
        # This prevents our code from crashing when using different input image resolutions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Define the final linear output layer
        # Input features: 64 channels * 4 height * 4 width = 1024 flat features
        # Output features: 1 (raw logit representing classification target star presence)
        self.fc = nn.Linear(in_features=64 * 4 * 4, out_features=1)

    # Define the forward pass logic of the neural network model
    def forward(self, x):
        # Pass input through conv1 -> pool1 -> relu1 block
        x = self.relu1(self.pool1(self.conv1(x)))
        
        # Pass intermediate representation through conv2 -> pool2 -> relu2 block
        x = self.relu2(self.pool2(self.conv2(x)))
        
        # Pass intermediate representation through conv3 -> pool3 -> relu3 block
        x = self.relu3(self.pool3(self.conv3(x)))
        
        # Apply adaptive average pooling to ensure shape is exactly [Batch, 64, 4, 4]
        x = self.adaptive_pool(x)
        
        # Flatten the spatial dimension into a single flat vector per batch item
        # Input shape: [Batch, Channels, Height, Width] -> Output shape: [Batch, Channels * Height * Width]
        x = x.view(x.size(0), -1)
        
        # Pass flat features through the linear classifier to get raw logits
        x = self.fc(x)
        
        # Return the final classification logit tensor
        return x
