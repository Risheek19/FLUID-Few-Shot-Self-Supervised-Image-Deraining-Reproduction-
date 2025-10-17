import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from sewar.full_ref import psnr, ssim
import warnings

# Suppress any warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Import your SSN model
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
# Utility Functions
# -----------------------------
def load_image(path):
    """Loads and preprocesses an image."""
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)

def tensor_to_image(tensor):
    """Converts tensor → image."""
    tensor = tensor.squeeze().detach().cpu().clamp(0, 1)
    np_img = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(np_img)

# -----------------------------
# Main Evaluation Function
# -----------------------------
def test_ddn_sirr(data_dir, ckpt_path, save_dir="./outputs/ddn_sirr_results"):
    """
    Evaluate FLUID SSN model on DDN-SIRR synthetic test set.
    Expected directory structure:
        data_dir/
          ├── rain/
          └── norain/
    """
    os.makedirs(save_dir, exist_ok=True)

    rain_dir = os.path.join(data_dir, "rain")
    clean_dir = os.path.join(data_dir, "norain")

    rain_files = sorted(os.listdir(rain_dir))
    clean_files = sorted(os.listdir(clean_dir))

    if len(rain_files) == 0 or len(clean_files) == 0:
        raise FileNotFoundError("❌ Could not find images in /rain or /norain folders")

    # Load SSN Model
    print(f"\n🔄 Loading checkpoint: {ckpt_path}")
    model = SSNet().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)

    # Flexible key loading
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
    else:
        model.load_state_dict(ckpt)
    print("✅ SSN checkpoint loaded successfully.\n")

    model.eval()

    total_psnr, total_ssim, count = 0.0, 0.0, 0

    print("🚀 Starting DDN-SIRR Synthetic Evaluation...\n")

    for rname in tqdm(rain_files, desc="Evaluating"):
        # Match 901_1.jpg → 901.jpg
        clean_name = rname.split('_')[0] + ".jpg"

        rain_path = os.path.join(rain_dir, rname)
        clean_path = os.path.join(clean_dir, clean_name)

        if not os.path.exists(clean_path):
            continue

        rainy = load_image(rain_path)
        clean = load_image(clean_path)

        with torch.no_grad():
            _, derained = model(rainy, rainy)

        # Convert to NumPy uint8 (for PSNR/SSIM)
        derained_np = derained.squeeze().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        clean_np = clean.squeeze().cpu().permute(1, 2, 0).numpy()

        derained_uint8 = (derained_np * 255).astype(np.uint8)
        clean_uint8 = (clean_np * 255).astype(np.uint8)

        # Compute metrics
        psnr_val = psnr(clean_uint8, derained_uint8)
        ssim_val = ssim(clean_uint8, derained_uint8)[0]

        total_psnr += psnr_val
        total_ssim += ssim_val
        count += 1

        # Save first few outputs
        if count <= 3:
            out_img = tensor_to_image(derained)
            out_img.save(os.path.join(save_dir, f"sample_{count}_{rname}"))

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count

    print(f"\n✅ Evaluation Complete on {count} image pairs")
    print(f"📊 Average PSNR: {avg_psnr:.3f}")
    print(f"📊 Average SSIM: {avg_ssim:.3f}")

    # Save results
    with open(os.path.join(save_dir, "ddn_sirr_results.txt"), "w", encoding="utf-8") as f:
        f.write(f"Dataset: {data_dir}\n")
        f.write(f"Model: {ckpt_path}\n")
        f.write(f"Average PSNR: {avg_psnr:.3f}\n")
        f.write(f"Average SSIM: {avg_ssim:.3f}\n")


# -----------------------------
# CLI Entry
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FLUID SSN Testing on DDN-SIRR Synthetic Data")
    parser.add_argument("--data", type=str, required=True, help="Path to DDN-SIRR dataset (contains /rain and /norain folders)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to SSN checkpoint")
    parser.add_argument("--save_dir", type=str, default="./outputs/ddn_sirr_results", help="Directory to save derained outputs")
    args = parser.parse_args()

    test_ddn_sirr(args.data, args.ckpt, args.save_dir)
