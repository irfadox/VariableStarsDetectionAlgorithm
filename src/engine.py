# Import PyTorch core library
import torch

# Training function for a single epoch
def train_epoch(model, dataloader, optimizer, criterion, device):
    # Set model to train mode
    model.train()
    
    # Track statistics
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # Iterate over batches
    for batch_idx, (sequences, labels) in enumerate(dataloader):
        # Move inputs and labels to target device
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        # Reset optimizer gradients
        optimizer.zero_grad()
        
        # Forward pass: compute logits
        outputs = model(sequences)
        
        # Compute multi-class cross entropy loss
        loss = criterion(outputs, labels)
        
        # Backward pass: compute parameter gradients
        loss.backward()
        
        # Update model parameters
        optimizer.step()
        
        # Accumulate metrics
        running_loss += loss.item() * sequences.size(0)
        # Find index of maximum logit to get predictions
        _, predictions = torch.max(outputs, dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += sequences.size(0)
        
    # Calculate epoch metrics
    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples
    
    # Return average loss and accuracy
    return epoch_loss, epoch_accuracy

# Validation/testing function for a single epoch
def test_epoch(model, dataloader, criterion, device):
    # Set model to eval mode
    model.eval()
    
    # Track statistics
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # Disable gradient calculations
    with torch.no_grad():
        # Iterate over validation batches
        for sequences, labels in dataloader:
            # Move inputs and labels to device
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            # Forward pass: compute logits
            outputs = model(sequences)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Accumulate metrics
            running_loss += loss.item() * sequences.size(0)
            # Fetch target predicted classes
            _, predictions = torch.max(outputs, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += sequences.size(0)
            
    # Calculate average epoch loss and accuracy
    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples
    
    # Return metrics
    return epoch_loss, epoch_accuracy
