import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

# -------------------------------------
# Few-Shot Paired Dataset for PEN (uses rain-* and norain-*)
# -------------------------------------
class RainFewShotDataset(Dataset):
    def __init__(self, root_dir, n_shot=5, patch_size=128):
        """
        Args:
            root_dir (str): Path to Rain100 dataset (e.g., D:/IIITH_Assignment/RainTrainL/RainTrainL)
            n_shot (int): Few-shot size {1, 3, 5}
            patch_size (int): Random crop size
        """
        self.root_dir = root_dir
        self.patch_size = patch_size

        # Collect rainy and clean image pairs
        self.rain_files = sorted([f for f in os.listdir(root_dir) if f.lower().startswith("rain-") and f.endswith(".png")])
        self.clean_files = sorted([f for f in os.listdir(root_dir) if f.lower().startswith("norain-") and f.endswith(".png")])

        total_pairs = min(len(self.rain_files), len(self.clean_files))
        if total_pairs == 0:
            raise RuntimeError("❌ No rain/norain image pairs found in the directory!")

        # Randomly select few-shot subset
        selected_indices = random.sample(range(total_pairs), min(n_shot, total_pairs))
        self.pairs = [(self.rain_files[i], self.clean_files[i]) for i in selected_indices]

        # Augmentation
        self.transform = T.Compose([
            T.RandomRotation(180),
            T.RandomCrop(patch_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rain_name, clean_name = self.pairs[idx]
        rain_path = os.path.join(self.root_dir, rain_name)
        clean_path = os.path.join(self.root_dir, clean_name)

        rain_img = Image.open(rain_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB")

        return self.transform(rain_img), self.transform(clean_img)


# -------------------------------------
# Unpaired Rainy Dataset for SSN (uses only rain-*)
# -------------------------------------
class RainUnpairedDataset(Dataset):
    def __init__(self, root_dir, patch_size=128):
        """
        Args:
            root_dir (str): Path to Rain100 dataset (rainy images only)
        """
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.images = sorted([f for f in os.listdir(root_dir) if f.lower().startswith("rain-") and f.endswith(".png")])

        if len(self.images) == 0:
            raise RuntimeError("❌ No rainy images found in the directory!")

        self.transform = T.Compose([
            T.RandomRotation(180),
            T.RandomCrop(patch_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        return self.transform(img)
