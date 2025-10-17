import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

# Project imports
from model_ssn import SSNResidualUNet as SSNet


# -----------------------------------------------------------
# Helper: Device info
# -----------------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"⚡ Using GPU: {device_name}")
        return 'cuda'
    else:
        print("🖥️ Using CPU (no GPU detected)")
        return 'cpu'


# -----------------------------------------------------------
# Helper: Match rainy and clean image filenames
# -----------------------------------------------------------
def get_image_pairs(rain_dir, clean_dir):
    rainy_images = sorted(os.listdir(rain_dir))
    pairs = []

    for rain_name in rainy_images:
        rain_path = os.path.join(rain_dir, rain_name)

        # Handle Rain100H & Rain100L naming patterns
        if os.path.exists(os.path.join(clean_dir, rain_name)):
            clean_name = rain_name
        elif rain_name.startswith("rain"):
            clean_name = rain_name.replace("rain", "norain")
        elif rain_name.startswith("norain"):
            # e.g. "norain-1.png" → "norain-1.png" already matches clean image
            clean_name = rain_name
        else:
            # fallback
            clean_name = rain_name.replace("rain", "norain")

        clean_path = os.path.join(clean_dir, clean_name)

        if os.path.exists(clean_path):
            pairs.append((rain_path, clean_path))
        else:
            print(f"⚠️ Missing clean image for {rain_name}")

    return pairs


# -----------------------------------------------------------
# Test SSN model on given dataset
# -----------------------------------------------------------
def test_ssn(test_root, ckpt_path="./checkpoints_ssn/last_ssn.pth", save_dir="./outputs/test"):
    device = get_device()
    print(f"\n🧪 Testing SSN on dataset: {test_root}")

    # Load model
    model = SSNet().to(device)

    print(f"🔄 Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # ✅ Robust checkpoint loader — supports multiple formats
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            print("✅ Loaded checkpoint (new format)")
        elif 'model_state' in ckpt:
            model.load_state_dict(ckpt['model_state'])
            print("✅ Loaded checkpoint (old format)")
        else:
            model.load_state_dict(ckpt)
            print("⚠️ Loaded checkpoint (weights only, no metadata)")
    else:
        model.load_state_dict(ckpt)
        print("⚠️ Loaded plain weights checkpoint")

    model.eval()

    # Infer subfolder names
    rain_dir = os.path.join(test_root, "rain")
    clean_dir = os.path.join(test_root, "norain")

    os.makedirs(save_dir, exist_ok=True)
    image_pairs = get_image_pairs(rain_dir, clean_dir)

    total_psnr, total_ssim, count = 0.0, 0.0, 0

    with torch.no_grad():
        for rain_path, clean_path in tqdm(image_pairs, desc="Testing"):
            rainy_img = Image.open(rain_path).convert("RGB")
            clean_img = Image.open(clean_path).convert("RGB")

            rainy = TF.to_tensor(rainy_img).unsqueeze(0).to(device)
            clean = TF.to_tensor(clean_img).unsqueeze(0).to(device)

            # Forward pass
            _, derained = model(rainy, rainy)
            derained = torch.clamp(derained, 0, 1).detach().cpu()

            derained_np = derained.squeeze().permute(1, 2, 0).numpy()
            clean_np = clean.squeeze().permute(1, 2, 0).cpu().numpy()

            # Metrics
            psnr_val = psnr_metric(clean_np, derained_np, data_range=1.0)
            ssim_val = ssim_metric(clean_np, derained_np, data_range=1.0, channel_axis=-1)

            total_psnr += psnr_val
            total_ssim += ssim_val
            count += 1

            # Save derained image
            out_name = os.path.basename(rain_path).replace("rain", "derained")
            out_path = os.path.join(save_dir, out_name)
            out_img = (derained_np * 255).astype(np.uint8)
            Image.fromarray(out_img).save(out_path)

    avg_psnr = total_psnr / max(1, count)
    avg_ssim = total_ssim / max(1, count)

    print(f"\n✅ Testing complete on {test_root}")
    print(f"📊 Average PSNR: {avg_psnr:.3f} | Average SSIM: {avg_ssim:.3f}")

    return avg_psnr, avg_ssim


# -----------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test SSN Model on Rain Datasets")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to test dataset root (contains /rain and /norain folders)")
    parser.add_argument("--ckpt", type=str, default="./checkpoints_ssn/last_ssn.pth",
                        help="Path to trained SSN checkpoint")
    parser.add_argument("--save_dir", type=str, default="./outputs/Rain100H_test",
                        help="Directory to save derained outputs")
    args = parser.parse_args()

    test_ssn(args.data, args.ckpt, args.save_dir)
