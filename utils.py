import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision import transforms
from datetime import datetime

# -----------------------------------------------------------
# 1️⃣ Prior Generation (Stage 2)
# -----------------------------------------------------------
def generate_prior(image_tensor, pen_model, T_h=0.95, device='cuda'):
    """
    Args:
        image_tensor (torch.Tensor): Rainy input tensor (B, 3, H, W)
        pen_model (torch.nn.Module): Trained PEN model
        T_h (float): Inpainting threshold (default 0.95)
        device (str): 'cuda' or 'cpu'
    Returns:
        prior (torch.Tensor): Inpainted image tensor (B, 3, H, W)
    """
    pen_model.eval()
    with torch.no_grad():
        P_NL = pen_model(image_tensor.to(device))  # (B, 1, H, W)
        M = (P_NL < T_h).float()  # Clean = 1, Rain = 0

    priors = []
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()

    for i in range(image_tensor.size(0)):
        img_np = image_tensor[i].cpu().permute(1, 2, 0).numpy()
        mask_np = (1 - M[i].cpu().squeeze().numpy()).astype(np.uint8) * 255  # Rain = 255

        img_bgr = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        inpainted = cv2.inpaint(img_bgr, mask_np, 3, cv2.INPAINT_TELEA)
        inpainted_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)

        priors.append(to_tensor(inpainted_rgb))

    prior_tensor = torch.stack(priors).to(device)
    return prior_tensor


# -----------------------------------------------------------
# 2️⃣ Evaluation Metrics: PSNR, SSIM, BRISQUE
# -----------------------------------------------------------

def calculate_psnr(img1, img2):
    """Compute Peak Signal-to-Noise Ratio."""
    img1_np = img1.detach().cpu().numpy().transpose(1, 2, 0)
    img2_np = img2.detach().cpu().numpy().transpose(1, 2, 0)
    return peak_signal_noise_ratio(img1_np, img2_np, data_range=1.0)

def calculate_ssim(img1, img2):
    """Compute Structural Similarity Index Measure."""
    img1_np = img1.detach().cpu().numpy().transpose(1, 2, 0)
    img2_np = img2.detach().cpu().numpy().transpose(1, 2, 0)
    return structural_similarity(img1_np, img2_np, channel_axis=2, data_range=1.0)

def calculate_brisque(img):
    """
    Compute BRISQUE score (no-reference IQA metric).
    Lower score = better quality.
    Uses OpenCV’s NIQE surrogate if BRISQUE is unavailable.
    """
    try:
        import piq
        score = piq.brisque(img.unsqueeze(0)).item()
    except Exception:
        img_np = img.detach().cpu().numpy().transpose(1, 2, 0)
        gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        score = cv2.quality.QualityBRISQUE_compute(gray, "brisque_model_live.yml", "brisque_range_live.yml")[0]
    return score


# -----------------------------------------------------------
# 3️⃣ Checkpoint Saving & Loading
# -----------------------------------------------------------
import os
import torch

def save_checkpoint(model, optimizer, epoch, metric, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        'epoch': epoch,
        'model_state': model.state_dict(),      # consistent naming
        'optimizer_state': optimizer.state_dict(),
        'metric': metric,
    }
    torch.save(state, path)
    print(f"💾 Checkpoint saved at {path} (epoch {epoch})")



def load_checkpoint(model, optimizer, path, device):
    if not os.path.exists(path):
        print(f"⚠️ No checkpoint found at {path}, starting fresh.")
        return model, optimizer, 0, float('inf')

    checkpoint = torch.load(path, map_location=device)
    keys = checkpoint.keys()

    # ✅ Handle multiple common naming conventions
    if 'model_state' in keys:
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint.get('epoch', 0)
        metric = checkpoint.get('metric', float('inf'))
    elif 'model' in keys:  # old version
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0)
        metric = checkpoint.get('best_val_loss', float('inf'))
    elif 'model_state_dict' in keys:  # your current file format
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        metric = checkpoint.get('metric', float('inf'))
    else:
        raise KeyError("❌ Invalid checkpoint format: missing model weights")

    print(f"🔄 Resumed from epoch {start_epoch + 1} (metric: {metric:.4f})")
    return model, optimizer, start_epoch + 1, metric
