# Import standard path utility library
import os
# Import PyTorch library
import torch
# Import DataLoader for batching
from torch.utils.data import DataLoader
# Import custom light curve dataset
from src.data_setup import LightCurveDataset
# Import custom 1D CNN model architecture
from src.model import LightCurveCNN
# Import train and validation drivers
from src.engine import train_epoch, test_epoch

# Define main orchestrator
def main():
    # Print device selection info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using training device: {device}")
    
    # Initialize mock training and testing datasets
    # Generate 5000 training curves and 1000 testing curves with 100 timesteps each
    train_dataset = LightCurveDataset(num_samples=5000, seq_len=100, is_mock=True)
    test_dataset = LightCurveDataset(num_samples=1000, seq_len=100, is_mock=True)
    
    # Setup DataLoaders
    batch_size = 32
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    # Instantiate the classification model (4 classes output)
    model = LightCurveCNN(num_classes=4)
    model.to(device)
    
    # Instantiate CrossEntropyLoss suitable for multi-class classification
    criterion = torch.nn.CrossEntropyLoss()
    
    # Instantiate Adam optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    
    # Set training epochs to 30
    epochs = 30

    print("Beginning light curve classification training...")
    
    # Run training loop
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        
        # Run training step
        train_loss, train_acc = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
        
        # Run test validation step
        # Print metrics (classification report & confusion matrix) only on the final epoch
        is_final_epoch = (epoch == epochs)
        test_loss, test_acc = test_epoch(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            print_metrics=is_final_epoch
        )
        print(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc * 100:.2f}%")

        
    # Create models export directory
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Save the final model weights
    save_path = os.path.join(models_dir, "star_classifier.pth")
    torch.save(obj=model.state_dict(), f=save_path)
    print(f"\nTraining completed! Model weights saved to: {save_path}")

# Run main script execution guards
if __name__ == "__main__":
    main()
