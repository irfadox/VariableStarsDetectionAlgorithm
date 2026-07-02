# Import standard system modules
import os
# Import PyTorch core library
import torch
# Import DataLoader for batching and loading data
from torch.utils.data import DataLoader
# Import the custom FITS Dataset class
from src.data_setup import PHATFitsDataset
# Import the convolutional neural network architecture
from src.model import VariableStarCNN
# Import train and test epochs from the engine
from src.engine import train_epoch, test_epoch

# Define the main runner function
def main():
    # Setup paths for data directories
    # Define root data directory path
    data_dir = "data"
    # Define training data directory path
    train_dir = os.path.join(data_dir, "train")
    # Define test data directory path
    test_dir = os.path.join(data_dir, "test")
    
    # Check if target data directories exist before proceeding
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        # Print warning indicating directories will need to be populated
        print(f"Warning: Please ensure '{train_dir}' and '{test_dir}' exist and contain .fits files before running.")
        # Create directories to help the user place files
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        print("Created data placeholder directories.")

    # Initialize target device: use GPU if CUDA is available, otherwise default to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Print the selected device to stdout
    print(f"Using training device: {device}")
    
    # Initialize the custom training dataset from the training folder
    train_dataset = PHATFitsDataset(data_dir=train_dir)
    # Initialize the custom testing dataset from the test folder
    test_dataset = PHATFitsDataset(data_dir=test_dir)
    
    # Set hyperparameter batch size
    batch_size = 16
    
    # Initialize PyTorch DataLoader for training data
    # Shuffle set to True to ensure generalizability and eliminate sequence bias
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # Initialize PyTorch DataLoader for testing data
    # Shuffle set to False for evaluation consistency
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Instantiate the CNN model architecture
    model = VariableStarCNN()
    # Move the model parameter tensors to the selected device memory (CPU/GPU)
    model.to(device)
    
    # Instantiate the Adam optimizer with model parameters and learning rate 0.001
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    
    # Define BCEWithLogitsLoss as optimization criterion (numerically stable binary cross entropy)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Set number of training epochs to 5
    epochs = 5
    
    # Print start of training notice
    print("Beginning model training...")
    
    # Run the training loop for the specified number of epochs
    for epoch in range(1, epochs + 1):
        # Print epoch header
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        
        # Check if train dataset contains files before executing training step
        if len(train_dataset) > 0:
            # Execute training epoch and receive loss/accuracy metrics
            train_loss, train_acc = train_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device
            )
            # Print training metrics to console
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
        else:
            # Inform user training skipped due to empty directory
            print("Skipped Training: No FITS files found in data/train.")
            
        # Check if test dataset contains files before executing test step
        if len(test_dataset) > 0:
            # Execute test evaluation epoch and receive loss/accuracy metrics
            test_loss, test_acc = test_epoch(
                model=model,
                dataloader=test_loader,
                criterion=criterion,
                device=device
            )
            # Print testing metrics to console
            print(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc * 100:.2f}%")
        else:
            # Inform user testing skipped due to empty directory
            print("Skipped Testing: No FITS files found in data/test.")
            
    # Setup path to save trained model weights
    # Define target folder path
    models_dir = "models"
    # Ensure models directory exists
    os.makedirs(models_dir, exist_ok=True)
    # Define complete weight path string
    model_save_path = os.path.join(models_dir, "variable_star_model.pth")
    
    # Save the trained model's state dictionary to disk
    torch.save(obj=model.state_dict(), f=model_save_path)
    # Print confirmation message
    print(f"\nTraining completed! Model weights saved successfully to: {model_save_path}")

# Run main script execution guards
if __name__ == "__main__":
    # Call the main training setup function
    main()
