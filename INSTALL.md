# Installation Guide

## 📋 Prerequisites

- Python >= 3.8
- CUDA >= 11.7 (for GPU support)
- GPU with >= 12GB VRAM (recommended for training)

## 🚀 Quick Install

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/WaveRNet.git
cd WaveRNet

# 2. Create virtual environment (recommended)
conda create -n waverNet python=3.10
conda activate waverNet

# 3. Install PyTorch (adjust CUDA version as needed)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# 4. Install other dependencies
pip install -r requirements.txt

# 5. Install SAM
pip install git+https://github.com/facebookresearch/segment-anything.git

# 6. Download SAM pretrained weights
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
mv sam_vit_b_01ec64.pth pretrained/
```

## 📦 Detailed Installation Steps

### Step 1: Environment Setup

#### Option A: Using Conda (Recommended)

```bash
conda create -n waverNet python=3.10
conda activate waverNet
```

#### Option B: Using venv

```bash
python -m venv waverNet_env
source waverNet_env/bin/activate  # Linux/Mac
# or
waverNet_env\Scripts\activate  # Windows
```

### Step 2: Install PyTorch

Choose the appropriate command based on your CUDA version:

**CUDA 11.8:**
```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1:**
```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu121
```

**CPU Only:**
```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu
```

Verify installation:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- opencv-python
- scikit-image
- albumentations
- monai
- numpy, pandas
- matplotlib, seaborn
- tqdm

### Step 4: Setup SAM Modules

See [SAM_README.md](SAM_README.md) for detailed instructions.

**Quick setup:**
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### Step 5: Download Pretrained Weights

#### SAM ViT-B Weights (Required)

```bash
cd pretrained
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
cd ..
```

#### WaveRNet Pretrained Models (Optional)

Download from:
- [Google Drive](https://drive.google.com/xxx)
- [Baidu Netdisk](https://pan.baidu.com/xxx)

Place in `pretrained/` directory.

## 📊 Dataset Preparation

### Download Datasets

1. **DRIVE**: https://drive.grand-challenge.org/
2. **STARE**: http://cecas.clemson.edu/~ahoover/stare/
3. **CHASE_DB1**: https://blogs.kingston.ac.uk/retinal/chasedb1/
4. **RECOVERY-FA19**: https://github.com/rmaphoh/RVD

### Organize Data Structure

```
data/
├── image_1024/
│   ├── CHASE_01.png
│   ├── DRIVE_01.png
│   ├── STARE_01.png
│   └── RECOVERY_01.png
└── mask_1024/
    ├── CHASE_01.png
    ├── DRIVE_01.png
    ├── STARE_01.png
    └── RECOVERY_01.png
```

### Preprocess Images

Resize all images to 1024×1024:

```python
from PIL import Image
import os

def resize_images(input_dir, output_dir, size=(1024, 1024)):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(os.path.join(input_dir, filename))
            img_resized = img.resize(size, Image.BILINEAR)
            img_resized.save(os.path.join(output_dir, filename))

# Resize images
resize_images('data/images_raw', 'data/image_1024')
resize_images('data/masks_raw', 'data/mask_1024')
```

## ✅ Verify Installation

Run the verification script:

```bash
python -c "
import torch
import torchvision
import cv2
import albumentations
import monai
from models.waverNet import SAMB
print('✅ All imports successful!')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
"
```

## 🎯 Quick Test

Test the model with a single image:

```bash
python eval.py \
  --data_config configs/config_test_drive.json \
  --checkpoint pretrained/waverNet_drive.pth \
  --output_dir results/test
```

## 🐛 Troubleshooting

### Issue: CUDA out of memory

**Solution**: Reduce batch size in training script:
```bash
python train.py --batch 1  # instead of --batch 2
```

### Issue: `ModuleNotFoundError: No module named 'SAM'`

**Solution**: Follow [SAM_README.md](SAM_README.md) to setup SAM modules.

### Issue: `FileNotFoundError: data/image_1024/`

**Solution**: Make sure you've prepared the dataset following the structure above.

### Issue: Slow data loading

**Solution**: Increase `num_workers` in dataloader (in `train.py`):
```python
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=args.batch,
    shuffle=True,
    num_workers=4,  # Increase this
    pin_memory=True
)
```

### Issue: Import errors with albumentations

**Solution**: Reinstall albumentations:
```bash
pip uninstall albumentations -y
pip install albumentations==1.3.1
```

## 💻 System Requirements

### Minimum Requirements

- CPU: 4 cores
- RAM: 16GB
- GPU: 8GB VRAM
- Storage: 50GB

### Recommended Requirements

- CPU: 8+ cores
- RAM: 32GB
- GPU: 16GB+ VRAM (RTX 3090, A100, etc.)
- Storage: 100GB SSD

## 📚 Next Steps

After installation:

1. **Training**: See [README.md](README.md#training) for training instructions
2. **Evaluation**: See [README.md](README.md#evaluation) for evaluation guide
3. **Pretrained Models**: Download from [pretrained/README.md](pretrained/README.md)

## 🆘 Getting Help

If you encounter issues:

1. Check [Troubleshooting](#troubleshooting) section above
2. Search existing [GitHub Issues](https://github.com/YOUR_USERNAME/WaveRNet/issues)
3. Open a new issue with:
   - Error message
   - Python version
   - CUDA version
   - Steps to reproduce

## 📝 Notes

- **Windows Users**: Some commands may need adjustment (e.g., use `\` instead of `/` for paths)
- **Mac Users**: CUDA is not available on Mac. Use CPU mode or cloud GPU services
- **Docker**: We plan to provide a Docker image in the future for easier setup

---

**Installation Complete! 🎉**

You're now ready to train and evaluate WaveRNet. See [README.md](README.md) for usage instructions.
