import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms


class LunarDataset(Dataset):

    def __init__(self, distorted_dir, clean_dir):

        self.distorted_dir = Path(distorted_dir)
        self.clean_dir = Path(clean_dir)

        self.image_names = sorted(
            [img.name for img in self.distorted_dir.glob("*")]
        )

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        image_name = self.image_names[idx]

        distorted = Image.open(
            self.distorted_dir / image_name
        ).convert("RGB")

        clean = Image.open(
            self.clean_dir / image_name
        ).convert("RGB")

        distorted = self.transform(distorted)
        clean = self.transform(clean)

        return distorted, clean



# 1. Point to the specific matching resolution folders
distorted_path = "Distorted images/368x368"
clean_path = "Distortion-free images/368x368"

# 2. Instantiate your custom dataset
moon_dataset = LunarDataset(distorted_dir=distorted_path, clean_dir=clean_path)

# 3. Wrap it in a DataLoader for batching
moon_dataloader = DataLoader(moon_dataset, batch_size=4, shuffle=True)

# 4. Create a quick function to test our pipeline visually
def test_visualization(dataloader):
    # Grab one batch of data
    distorted_batch, clean_batch = next(iter(dataloader))
    
    # Plot 4 pairs side-by-side
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(4):
        # PyTorch uses (Channels, Height, Width), but Matplotlib expects (Height, Width, Channels)
        dist_img = distorted_batch[i].permute(1, 2, 0).numpy()
        clean_img = clean_batch[i].permute(1, 2, 0).numpy()
        
        axes[0, i].imshow(dist_img)
        axes[0, i].set_title(f"Distorted {i+1}")
        axes[0, i].axis("off")
        
        axes[1, i].imshow(clean_img)
        axes[1, i].set_title(f"Clean {i+1}")
        axes[1, i].axis("off")
        
    plt.suptitle("Sanity Check: Distorted vs Clean Target")
    plt.tight_layout()
    plt.show()

# Run the test
if __name__ == "__main__":
    print(f"Total image pairs loaded: {len(moon_dataset)}")
    test_visualization(moon_dataloader)    
    