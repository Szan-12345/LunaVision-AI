import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import your custom classes from the files you just made
from dataset import LunarDataset
from model import DenoisingAutoencoder

def train_model():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 2. Load Data
    distorted_path = "Distorted images/368x368"
    clean_path = "Distortion-free images/368x368"
    
    dataset = LunarDataset(distorted_dir=distorted_path, clean_dir=clean_path)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True) 

    # 3. Initialize Model, Loss, and Optimizer
    model = DenoisingAutoencoder().to(device)
    criterion = nn.MSELoss() # Mean Squared Error is standard for image reconstruction
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. The Training Loop
    epochs = 5
    
    for epoch in range(epochs):
        running_loss = 0.0
        
        for batch_idx, (distorted, clean) in enumerate(dataloader):
            # Move data to GPU if available
            distorted = distorted.to(device)
            clean = clean.to(device)
            
            # Forward pass
            outputs = model(distorted)
            loss = criterion(outputs, clean)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")
                
        # Epoch summary
        epoch_loss = running_loss / len(dataloader)
        print(f"--- Epoch {epoch+1} Completed | Average Loss: {epoch_loss:.4f} ---")

    # 5. Save the trained weights
    torch.save(model.state_dict(), "moonvision_autoencoder.pth")
    print("Training complete. Model saved!")

if __name__ == "__main__":
    train_model()
    