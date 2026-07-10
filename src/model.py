# =================================================================================
# MODEL.PY - 1D CNN Neural Network Structure
# For a conceptual explanation of how this fits into the project, see:
# walkthrough_guide.md (at the repository or workspace root)
# =================================================================================
# Import the neural network blocks from PyTorch (we use these like Lego bricks!)
import torch.nn as nn

# This is our Star Classifier! It takes in brightness curves and guesses the star type.
class LightCurveCNN(nn.Module):
    # This is the construction yard where we prepare all our Lego blocks.
    def __init__(self, num_classes=5):
        # Always tell PyTorch to prepare the parent Lego set first!
        super(LightCurveCNN, self).__init__()
        
        # --- LEGO BLOCK 1: Conv block 1 ---
        # Conv1d is like a small sliding window that looks for tiny patterns in the star's brightness.
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        # BatchNorm is like washing the data so all numbers are clean and neat between -1 and 1.
        self.bn1 = nn.BatchNorm1d(32)
        # MaxPool is like taking a big picture and shrinking it by keeping only the biggest, brightest spots!
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        # ReLU is the "no negatives allowed" filter. If a number is sad/negative, it becomes 0 (goes to sleep).
        self.relu1 = nn.ReLU()
        
        # --- LEGO BLOCK 2: Conv block 2 ---
        # Now we look for medium-sized patterns using the outputs from the first block.
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.relu2 = nn.ReLU()
        
        # --- LEGO BLOCK 3: Conv block 3 ---
        # Now we look for even bigger and more complex patterns!
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.relu3 = nn.ReLU()
        
        # Adaptive pooling squishes the data down to a fixed length of 8 so it's easy to read.
        self.adaptive_pool = nn.AdaptiveAvgPool1d(8)
        
        # Dropout is like playing hide-and-seek. We randomly turn off some connections (30%)
        # during training so the network doesn't get lazy and rely on just one trick!
        self.dropout = nn.Dropout(p=0.3)
        
        # Final linear classifier layer (the brain).
        # It takes all 1024 points (128 channels * 8 length) and decides which of the 4 star classes it is.
        self.fc = nn.Linear(in_features=128 * 8, out_features=num_classes)

    # This is the pipeline where the data actually flows through the Lego blocks!
    def forward(self, x):
        # Flow through block 1: Slide window -> Wash -> Shrink -> Sleep negative numbers
        x = self.relu1(self.pool1(self.bn1(self.conv1(x))))
        # Flow through block 2
        x = self.relu2(self.pool2(self.bn2(self.conv2(x))))
        # Flow through block 3
        x = self.relu3(self.pool3(self.bn3(self.conv3(x))))
        
        # Squish to size 8
        x = self.adaptive_pool(x)
        
        # Flatten: stretch the 2D grid into one long 1D line of numbers
        x = x.view(x.size(0), -1)
        
        # Hide-and-seek filter (only during training)
        x = self.dropout(x)
        
        # Final guess: output 4 numbers (one score for each type of star!)
        x = self.fc(x)
        
        return x

