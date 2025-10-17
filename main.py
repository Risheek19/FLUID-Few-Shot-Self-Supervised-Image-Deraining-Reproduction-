import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

# Project imports
from config import PEN_CONFIG as PEN_CFG, SSN_CONFIG as SSN_CFG, DATA_ROOT
from model_pen import UNetPEN as PENet
from model_ssn import SSNResidualUNet as SSNet
from dataset import RainFewShotDataset, RainUnpairedDataset
from losses import BCELoss, SSNLoss
from utils import calculate_psnr, calculate_ssim, save_checkpoint


# -----------------------------------------------------------
# Helper
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
# Stage 1 – PEN Training
# -----------------------------------------------------------
def train_pen():
    device = get_device()
    print(f"\n🚀 Stage 1: Training PEN on {device}")

    model = PENet().to(device)
    optimizer = Adam(model.parameters(), lr=PEN_CFG['lr'])
    criterion = BCELoss()

    ckpt_path = os.path.join(PEN_CFG['checkpoint_dir'], "last_pen.pth")
    os.makedirs(PEN_CFG['checkpoint_dir'], exist_ok=True)

    start_epoch = 0

    # ✅ Resume checkpoint logic (compatible with all formats)
    if PEN_CFG['resume'] and os.path.exists(ckpt_path):
        print(f"🔄 Found existing checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)

        try:
            if 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
                if 'optimizer_state_dict' in ckpt:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                start_epoch = ckpt.get('epoch', 0) + 1
                print(f"✅ Resumed from epoch {start_epoch}")

            elif 'model_state' in ckpt:
                model.load_state_dict(ckpt['model_state'])
                if 'optimizer_state' in ckpt:
                    optimizer.load_state_dict(ckpt['optimizer_state'])
                start_epoch = ckpt.get('epoch', 0) + 1
                print(f"✅ Resumed (old format) from epoch {start_epoch}")

            else:
                model.load_state_dict(ckpt)
                print("⚠️ Loaded plain model weights only (no optimizer or epoch info).")

        except Exception as e:
            print(f"❌ Error loading checkpoint ({e}) — starting fresh from epoch 0.")
            start_epoch = 0

    dataset = RainFewShotDataset(PEN_CFG['data_root'], n_shot=PEN_CFG['few_shot_n'])
    loader = DataLoader(dataset, batch_size=PEN_CFG['batch_size'], shuffle=True)

    log_file = os.path.join(PEN_CFG['checkpoint_dir'], "pen_training_log.txt")  # 🔹 Added logging file

    model.train()
    for epoch in range(start_epoch, PEN_CFG['epochs']):
        epoch_loss = 0.0
        for rainy, clean in tqdm(loader, desc=f"[PEN Epoch {epoch}]"):
            rainy, clean = rainy.to(device), clean.to(device)
            target = torch.clamp((rainy - clean).abs(), 0, 1)

            optimizer.zero_grad()
            pred = model(rainy)
            loss = criterion(pred, target.mean(1, keepdim=True))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch} → Loss: {avg_loss:.6f}")

        # 🔹 Save checkpoint (latest)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, ckpt_path)

        # 🔹 Backup every 100 epochs
        if epoch % 100 == 0 or epoch == PEN_CFG['epochs'] - 1:
            backup_path = os.path.join(PEN_CFG['checkpoint_dir'], f"pen_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, backup_path)
            print(f"🧱 Backup saved → {backup_path}")

        # 🔹 Log to text file
        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch}: loss={avg_loss:.6f}\n")

    print("✅ PEN training complete ✔️")
    return model



# -----------------------------------------------------------
# Stage 2 – Prior Generation (PNG only)
# -----------------------------------------------------------
def generate_priors(pen_model):
    device = get_device()
    dataset = RainUnpairedDataset(SSN_CFG['data_root'])
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    os.makedirs("./1_shot_outputs/pen_1_shot", exist_ok=True)
    os.makedirs("./1_shot_outputs/masked_1_shot", exist_ok=True)
    os.makedirs("./1_shot_outputs/priors_1_shot", exist_ok=True)

    print(f"\n⚙️ Generating priors using trained PEN ... ({len(loader)} images)")
    pen_model.eval()

    with torch.no_grad():
        for idx, rainy in enumerate(tqdm(loader)):
            rainy = rainy.to(device)
            print(f"🖼️ Processing image {idx+1}/{len(loader)}")

            p_map = torch.sigmoid(pen_model(rainy))
            p_map_np = p_map.squeeze().cpu().numpy()
            p_map_vis = (p_map_np * 255).astype(np.uint8)
            Image.fromarray(p_map_vis).save(f"./1_shot_outputs/pen_1_shot/pen_{idx+1}.png")

            mask = (p_map_np < SSN_CFG['threshold']).astype(np.float32)
            rainy_np = rainy.squeeze().permute(1, 2, 0).cpu().numpy()
            masked_img = np.clip(rainy_np * mask[..., None], 0, 1)
            Image.fromarray((masked_img * 255).astype(np.uint8)).save(f"./1_shot_outputs/masked_1_shot/masked_{idx+1}.png")

            rainy_uint8 = (rainy_np * 255).astype(np.uint8)
            mask_cv = ((1 - mask) * 255).astype(np.uint8)
            prior_bgr = cv2.inpaint(cv2.cvtColor(rainy_uint8, cv2.COLOR_RGB2BGR),
                                    mask_cv, 3, cv2.INPAINT_TELEA)
            prior_rgb = cv2.cvtColor(prior_bgr, cv2.COLOR_BGR2RGB)
            Image.fromarray(prior_rgb).save(f"./1_shot_outputs/priors_1_shot/prior_{idx+1}.png")

    print("✅ Prior generation complete ✔️")




def train_ssn():
    device = get_device()
    print(f"\n🚀 Stage 3: Training SSN on {device}")

    model = SSNet().to(device)
    optimizer = Adam(model.parameters(), lr=SSN_CFG['lr'])
    criterion = SSNLoss(lambda_tv=SSN_CFG['lambda_tv'], lambda_vgg=SSN_CFG['lambda_vgg'])

    os.makedirs(SSN_CFG['checkpoint_dir'], exist_ok=True)
    os.makedirs(os.path.join(SSN_CFG['checkpoint_dir'], "sample_outputs"), exist_ok=True)

    dataset = RainUnpairedDataset(SSN_CFG['data_root'])
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # -------------------------------------------------
    # 🔄 Resume training from checkpoint if available
    # -------------------------------------------------
    start_epoch = 0
    ckpt_dir = SSN_CFG['checkpoint_dir']

    # Identify checkpoints (supports 1/3/5-shot naming)
    checkpoint_files = sorted(
        [f for f in os.listdir(ckpt_dir) if f.endswith(".pth") and "ssn_epoch_" in f],
        key=lambda x: int(x.split("_")[-1].split(".")[0])
    )

    resume_path = os.path.join(ckpt_dir, checkpoint_files[-1]) if checkpoint_files else None

    if SSN_CFG.get('resume', False) and resume_path and os.path.exists(resume_path):
        print(f"🔄 Resuming SSN training from checkpoint: {resume_path}")
        try:
            ckpt = torch.load(resume_path, map_location=device)
            if 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                start_epoch = ckpt.get('epoch', 0) + 1
            elif isinstance(ckpt, dict):
                model.load_state_dict(ckpt)
                start_epoch = int(resume_path.split("_")[-1].split(".")[0]) + 1
            print(f"✅ Resumed from epoch {start_epoch}")
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")
            print("Starting fresh training...")
    else:
        print("🚀 Starting fresh SSN training")

    # -------------------------------------------------
    # 🧠 Training Loop
    # -------------------------------------------------
    prior_folder = SSN_CFG.get('prior_save_dir', './outputs/priors')

    model.train()
    for epoch in range(start_epoch, SSN_CFG['epochs']):
        total_loss, total_psnr, total_ssim = 0.0, 0.0, 0.0

        for idx, rainy in enumerate(tqdm(loader, desc=f"[SSN Epoch {epoch}]")):
            rainy = rainy.to(device).clamp(0, 1)

            # Load correct prior (1, 3, or 5 shot)
            prior_path = os.path.join(prior_folder, f"prior_{idx+1}.png")
            if not os.path.exists(prior_path):
                continue

            prior_img = Image.open(prior_path).convert("RGB")
            prior = TF.to_tensor(prior_img).unsqueeze(0).to(device).clamp(0, 1)

            optimizer.zero_grad()
            _, pred = model(prior, rainy)

            loss, _ = criterion(pred, prior, rainy)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_psnr += calculate_psnr(prior[0].detach(), pred[0].detach())
            total_ssim += calculate_ssim(prior[0].detach(), pred[0].detach())

            # ✅ Save sample outputs every 10 epochs
            if epoch % 10 == 0 and idx < 3:
                out = pred[0].detach().cpu()
                out = (out - out.min()) / (out.max() - out.min() + 1e-8)
                TF.to_pil_image(out.clamp(0, 1)).save(
                    os.path.join(SSN_CFG['checkpoint_dir'], "sample_outputs", f"derained_e{epoch}_i{idx+1}.png")
                )

        avg_loss = total_loss / max(1, len(loader))
        avg_psnr = total_psnr / max(1, len(loader))
        avg_ssim = total_ssim / max(1, len(loader))

        print(f"Epoch {epoch} → Loss:{avg_loss:.6f} | PSNR:{avg_psnr:.3f} | SSIM:{avg_ssim:.3f}")

        # Save checkpoint (shot-specific)
        shot_tag = SSN_CFG.get('shot_tag', '1_shot')
        ckpt_path = os.path.join(SSN_CFG['checkpoint_dir'], f"last_{shot_tag}_ssn.pth")
        save_checkpoint(model, optimizer, epoch, avg_psnr, ckpt_path)

        # Save backup every 50 epochs
        if epoch % 50 == 0 or epoch == SSN_CFG['epochs'] - 1:
            backup_path = os.path.join(SSN_CFG['checkpoint_dir'], f"{shot_tag}_ssn_epoch_{epoch}.pth")
            save_checkpoint(model, optimizer, epoch, avg_psnr, backup_path)
            print(f"🧱 Backup saved → {backup_path}")

    print("✅ SSN training complete ✔️")





# -----------------------------------------------------------
# Main Entry
# -----------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FLUID Rain Removal Training")
    parser.add_argument("--stage", type=str, default="pen",
                        choices=["pen", "prior", "ssn", "all"],
                        help="Stage to run: pen | prior | ssn | all")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to PEN checkpoint for prior generation (optional)")
    args = parser.parse_args()

    os.makedirs("checkpoints", exist_ok=True)

    device = get_device()
    print(f"🖥️ Using device: {device}")

    if args.stage == "pen":
        train_pen()

    elif args.stage == "prior":
        pen_model = PENet().to(device)

        # ✅ Checkpoint priority: user-specified > default
        if args.ckpt is not None and os.path.exists(args.ckpt):
            print(f"🔄 Loading PEN checkpoint from: {args.ckpt}")
            ckpt = torch.load(args.ckpt, map_location=device)
            if "model_state_dict" in ckpt:
                pen_model.load_state_dict(ckpt["model_state_dict"])
            else:
                pen_model.load_state_dict(ckpt)
            print("✅ Successfully loaded provided PEN checkpoint.")
        else:
            ckpt_path = os.path.join(PEN_CFG['checkpoint_dir'], "last_pen.pth")
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=device)
                pen_model.load_state_dict(ckpt["model_state_dict"])
                print("✅ Loaded default PEN checkpoint for prior generation.")
            else:
                raise FileNotFoundError("❌ No PEN checkpoint found (neither provided nor default).")

        generate_priors(pen_model)

    elif args.stage == "ssn":
        train_ssn()

    elif args.stage == "all":
        pen_model = train_pen()
        generate_priors(pen_model)
        train_ssn()
