# 🩹 SciPy compatibility patch for BRISQUE
import scipy
import numpy as np
if not hasattr(scipy, "ndarray"):
    scipy.ndarray = np.ndarray

import os
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from imquality import brisque
from sewar.full_ref import psnr, ssim
from brisque import BRISQUE
import warnings

# Suppress warnings from PIL/torchvision
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# Imports from your repo
# -----------------------------
from model_ssn import SSNResidualUNet as SSNet

# -----------------------------
# Configuration
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# -----------------------------
# Utility functions
# -----------------------------
def load_image(path):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)

def tensor_to_image(tensor):
    tensor = tensor.squeeze().detach().cpu().clamp(0, 1)
    arr = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)

# -----------------------------
# Test Function
# -----------------------------
def test_brisque(data_dir, ckpt_path, save_dir="./outputs/5_shot_brisque_results"):
    os.makedirs(save_dir, exist_ok=True)
    rain_dir = data_dir  # Directly use folder containing rainy images
    img_files = sorted(os.listdir(rain_dir))

    model = SSNet().to(device)
    print(f"🔄 Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # Flexible loading (handles all checkpoint formats)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
    else:
        model.load_state_dict(ckpt)
    print("✅ Loaded SSN checkpoint successfully.\n")

    model.eval()
    brisque_metric = BRISQUE()
    all_scores = []

    for fname in tqdm(img_files, desc="Evaluating BRISQUE"):
        rain_path = os.path.join(rain_dir, fname)
        rainy_img = load_image(rain_path)

        # Self-supervised — use rainy image as both prior & input
        with torch.no_grad():
            _, derained = model(rainy_img, rainy_img)

        derained_img = tensor_to_image(derained)
        out_path = os.path.join(save_dir, f"derained_{fname}")
        derained_img.save(out_path)

        # Compute BRISQUE (no-reference quality metric, lower = better)
        score = brisque_metric.get_score(out_path)
        all_scores.append(score)

    avg_brisque = np.mean(all_scores)
    print(f"\n✅ BRISQUE evaluation complete on {len(all_scores)} images")
    print(f"📊 Average BRISQUE Score ↓ : {avg_brisque:.3f}")
    
    results_path = os.path.join(save_dir, "brisque_results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
            f.write(f"Dataset: {data_dir}\n")
            f.write(f"Model: {ckpt_path}\n")
            f.write(f"Average BRISQUE Score (lower is better): {avg_brisque:.3f}\n")

    print(f"💾 Results saved at: {results_path}")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BRISQUE Evaluation for Real Rainy Images (FLUID SSN)")
    parser.add_argument("--data", type=str, required=True, help="Path to folder with real rainy images")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to SSN checkpoint")
    parser.add_argument("--save_dir", type=str, default="./outputs/brisque_results", help="Directory to save outputs")
    args = parser.parse_args()

    test_brisque(args.data, args.ckpt, args.save_dir)
