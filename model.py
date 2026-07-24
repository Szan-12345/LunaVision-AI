import torch
import torch.nn as nn

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        
        # Encoder: Compress the image and extract features
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Halves dimensions to 184x184
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Halves dimensions to 92x92
        )
        
        # Decoder: Reconstruct the image from the compressed features
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.ReLU(), # Scales back up to 184x184
            
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=2, stride=2),
            nn.Sigmoid() # Scales back to 368x368 and squashes pixel values between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Quick test to ensure the math and shapes align perfectly
if __name__ == "__main__":
    model = DenoisingAutoencoder()
    print("Model Architecture Loaded:")
    print(model)
    
    # Create a dummy tensor representing a single 368x368 RGB image
    dummy_input = torch.randn(1, 3, 368, 368)
    output = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    if dummy_input.shape == output.shape:
        print("Success: The network successfully reconstructions the original dimensions!")
        