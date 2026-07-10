# =================================================================================
# TRAIN.PY - Model Training Orchestrator
# For a conceptual explanation of how this fits into the project, see:
# walkthrough_guide.md (at the repository or workspace root)
# =================================================================================
# Standard tool to help make directories (folders)
import os
# PyTorch is the main engine we use to build neural networks
import torch
# DataLoader is like a truck driver: it packs our stars in small batches and delivers them to the model
from torch.utils.data import DataLoader
# Load the helper that cleans and structures our star datasets
from src.data_setup import LightCurveDataset
# Import our robot brain model structure
from src.model import LightCurveCNN
# Import the training and testing steps (the teacher's lesson plans)
from src.engine import train_epoch, test_epoch

# This is where the magic starts!
def main():
    # Choose if we should run on a fancy GPU (CUDA) or a standard computer CPU (the slow but steady turtle)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using training device: {device}")
    
    # Generate 5,000 synthetic stars for the model to practice on, and 1,000 to test it with!
    train_dataset = LightCurveDataset(num_samples=5000, seq_len=100, is_mock=True)
    test_dataset = LightCurveDataset(num_samples=1000, seq_len=100, is_mock=True)
    
    # Pack them in batches of 32 stars per delivery truck
    batch_size = 32
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create the model brain (5 classes output: Cepheid, RR Lyrae, EB, LPV, Non-Variable / Noise)
    model = LightCurveCNN(num_classes=5)
    model.to(device) # Send the brain to our processing device
    
    # Criterion computes the "sadness score" (loss). If the model guesses wrong, this score goes up!
    criterion = torch.nn.CrossEntropyLoss()
    
    # Optimizer (Adam) is the tiny mechanic that adjusts the brain's settings after every guess to make it smarter.
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    
    # We will run the entire school term 30 times!
    epochs = 30
    # Keep track of the best test loss. We start with infinity because anything is better than infinity!
    best_test_loss = float('inf')
    
    # Create models export directory
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, "star_classifier.pth")

    print("Beginning light curve classification training...")
    
    # Start the school term loop!
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        
        # 1. Let the model practice and learn!
        train_loss, train_acc = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
        
        # 2. Test the model to see how smart it got. Only show the final detailed report at the very end!
        is_final_epoch = (epoch == epochs)
        test_loss, test_acc = test_epoch(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            print_metrics=is_final_epoch
        )
        print(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc * 100:.2f}%")

        # 3. Save the best model brain! If this score is lower than our best score so far, lock in the weights!
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(obj=model.state_dict(), f=save_path)
            print(f"🎉 New best model! Test Loss decreased to {best_test_loss:.4f}. Locked in weights to: {save_path}")

    print(f"\nTraining completed! The best model reached a Test Loss of {best_test_loss:.4f}.")

# Run the program!
if __name__ == "__main__":
    main()
