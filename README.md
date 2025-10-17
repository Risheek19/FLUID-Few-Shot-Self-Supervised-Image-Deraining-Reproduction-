# 🌧️ FLUID: Few-Shot Self-Supervised Image Deraining (Reproduction)

This repository contains a full reproduction of the paper:  
**“FLUID: Few-Shot Self-Supervised Image Deraining” (CVPR 2022)**  

---

## 🧩 1. Environment Setup

### Step 1: Create a new Conda environment
```bash
conda create -n rainremoval python=3.10 -y
conda activate rainremoval
```

### Step 2: Install dependencies
An `environment.yml` file is provided to install all required packages:
```bash
conda env update --file environment.yml
```

---

## 📁 2. Dataset Setup

Ensure the following datasets are downloaded and arranged in this structure:

```
DATA_ROOT/
│
├── RainTrainL/
│   ├── rain-*.png
│   └── norain-*.png
│
├── Rain100L/
│   └── test/
│       ├── rain/
│       └── norain/
│
├── Rain100H/
│   └── test/
│       ├── rain/
│       └── norain/
│
├── DDN-SIRR/
│   ├── rainy_image_dataset/
│   │   └── testing/
│   │       ├── rain/
│   │       └── norain/
│   └── real_input/
│       ├── 1.jpg
│       ├── 2.jpg
│       └── ...
```

---

## ⚙️ 3. Training Stages

Download Checkpoints from https://drive.google.com/drive/folders/1KWDcmZ7ql0pr6Q00EjbuXBcud3JjLrUG?usp=sharing

### (1) Train PEN (Probability Estimation Network)
For **1-shot:**
```bash
python main.py --stage pen
```

For **3-shot** or **5-shot:**
Edit `config.py`:
```python
'few_shot_n': 3,  # or 5
'checkpoint_dir': './checkpoints_pen_3_shot'  # or './checkpoints_pen_5_shot'
```
Then run:
```bash
python main.py --stage pen
```

---

### (2) Generate Priors
Once PEN training completes (e.g., checkpoint at 19999 epochs):
```bash
python main.py --stage prior --ckpt "path/to/trained_pen_checkpoint.pth"
```
Generated priors will be saved under:
```
./<shot_number>_shot_outputs/priors_<shot_number>_shot/
```

---

### (3) Train SSN (Self-Supervised Network)
Edit `config.py`:
```python
'prior_save_dir': './<shot_number>_shot_outputs/priors_<shot_number>_shot',
'checkpoint_dir': './outputs_ssn_<shot_number>_shot'
```

Then run:
```bash
python main.py --stage ssn
```

---

## 🧪 4. Testing and Evaluation

### (A) Rain100L Test Set
```bash
python test.py --data "path/to/Rain100L/test" --ckpt "path/to/ssn_checkpoint.pth" --save_dir "./results/Rain100L"
```

### (B) Rain100H Test Set
```bash
python test.py --data "path/to/Rain100H/test" --ckpt "path/to/ssn_checkpoint.pth" --save_dir "./results/Rain100H"
```

### (C) DDN-SIRR Synthetic Dataset
```bash
python ddn_test.py --data "path/to/DDN-SIRR/testing" --ckpt "path/to/ssn_checkpoint.pth" --save_dir "./results/DDN-SIRR-synthetic"
```

### (D) DDN-SIRR Real Dataset (BRISQUE Evaluation)
```bash
python brisque_test.py --data "path/to/DDN-SIRR/real_input" --ckpt "path/to/ssn_checkpoint.pth" --save_dir "./results/DDN-SIRR-real"
```

---



---

## 🧠 5. Notes

- All experiments use **128×128** image patches.  
- Automatically detects **GPU (CUDA)** and falls back to CPU if unavailable.  
- Each stage (`PEN`, `PRIOR`, `SSN`) can run **independently**.  
- All checkpoints and results are automatically saved.  
- Tested on **Windows 11 + Python 3.10 + PyTorch 2.2**.

---

## 📂 6. Key Files

| File | Description |
|------|--------------|
| `main.py` | Training pipeline for PEN / PRIOR / SSN |
| `test.py` | Evaluation on Rain100L / Rain100H |
| `ddn_test.py` | Evaluation on DDN-SIRR synthetic dataset |
| `brisque_test.py` | Evaluation on real rainy dataset |
| `config.py` | Configuration file for all experiments |

---

## 📚 7. Citation

If you use this code, please cite:

```bibtex
@inproceedings{fluid2022,
  title={FLUID: Few-Shot Self-Supervised Image Deraining},
  booktitle={CVPR},
  year={2022}
}
```


✅ *End of Document*
