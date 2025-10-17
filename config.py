# config.py

import torch
import os

# -----------------------------
# General Configuration
# -----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

DATA_ROOT = r"D:\IIITH_Assignment\RainTrainL\RainTrainL"
RAIN_DIR = os.path.join(DATA_ROOT, 'rain')
NORAIN_DIR = os.path.join(DATA_ROOT, 'norain')

# -----------------------------
# PEN (Probability Estimation Network)
# -----------------------------
PEN_CONFIG = {
    'architecture': 'UNet',
    'input_channels': 3,
    'output_channels': 1,
    'batch_size': 1,
    'epochs': 20000,                   # ⬆️ from 1000 → more stable probability maps
    'lr': 1e-4,
    'lr_decay_epoch': 2500,
    'input_size': (128, 128),
    'few_shot_n': 1,
    'augmentation': True,
    'save_interval': 500,
    'checkpoint_dir': './checkpoints_pen_1_shot',
    'data_root': DATA_ROOT,
    'resume': True,
}

# -----------------------------
# SSN (Self-Supervised Network)
# -----------------------------
SSN_CONFIG = {
    'architecture': 'ResUNet',
    'input_channels': 3,
    'output_channels': 3,
    'batch_size': 4,
    'epochs': 500,
    'lr': 5e-5,             # ↓ Lower LR to stabilize gradients
    'lr_min': 1e-6,
    'input_size': (128, 128),
    'augmentation': True,

    # Mask thresholds
    'threshold': 0.6,
    'inpaint_threshold': 0.6,

    # ✅ Balanced losses
    'lambda_tv': 1e-3,      # ↓ Reduced to avoid over-smoothing
    'lambda_vgg': 0.1,    # ↓ Less perceptual pull → avoids dark bias

    'checkpoint_dir': './outputs_ssn_5_shot',
    'resume': False,
    'data_root': DATA_ROOT,
    'shot_tag': '5_shot',
    'prior_save_dir': r'D:\IIITH_Assignment\5_shot_outputs\priors_5_shot',
}

# -----------------------------
# Training Resumption and Checkpointing
# -----------------------------
RESUME_PEN = True
RESUME_SSN = False

# -----------------------------
# Miscellaneous
# -----------------------------
SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
