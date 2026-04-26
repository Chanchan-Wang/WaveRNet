# SAM (Segment Anything Model) Setup Guide

## 📌 Overview

WaveRNet is built on top of Meta's Segment Anything Model (SAM). Due to the large size of SAM codebase, we don't include it directly in this repository. This guide shows you how to obtain the required SAM modules.

## 🚀 Quick Start (Recommended)

### Option 1: Install SAM via pip

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Then create a symbolic link or copy the SAM modules:

```bash
# After pip install, SAM will be in your Python site-packages
# Create SAM directory structure
mkdir -p SAM/modeling

# Copy required modules (adjust path based on your Python environment)
cp -r $(python -c "import segment_anything; print(segment_anything.__path__[0])")/modeling/* SAM/modeling/
```

### Option 2: Clone SAM Repository

```bash
# Clone official SAM repository
git clone https://github.com/facebookresearch/segment-anything.git temp_sam

# Copy required modules
mkdir -p SAM/modeling
cp -r temp_sam/segment_anything/modeling/* SAM/modeling/

# Clean up
rm -rf temp_sam
```

### Option 3: Manual Download

Download the following files from [SAM GitHub](https://github.com/facebookresearch/segment-anything/tree/main/segment_anything/modeling) and place them in `SAM/modeling/`:

**Required files:**
- `__init__.py`
- `common.py`
- `image_encoder.py`
- `mask_decoder.py`
- `prompt_encoder.py`
- `transformer.py`

**Note**: Our `adapter_encoder.py` is a modified version of `image_encoder.py` with adapter layers, already included in `models/` directory.

## 📦 Required SAM Modules

WaveRNet uses the following SAM components:

```
SAM/
└── modeling/
    ├── __init__.py
    ├── common.py              # LayerNorm2d, MLPBlock
    ├── image_encoder.py       # ImageEncoderViT (ViT-B backbone)
    ├── mask_decoder.py        # MaskDecoder
    ├── prompt_encoder.py      # PromptEncoder
    └── transformer.py         # TwoWayTransformer
```

## 🔑 Download Pretrained Weights

### SAM ViT-B Checkpoint

```bash
# Download SAM ViT-B pretrained weights (~375MB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# Move to pretrained directory
mv sam_vit_b_01ec64.pth pretrained/
```

Or download manually from:
- [sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

### WaveRNet Pretrained Models

Download our trained models from:
- [Google Drive](https://drive.google.com/xxx) (Coming soon)
- [Baidu Netdisk](https://pan.baidu.com/xxx) (Coming soon)

See `pretrained/README.md` for model performance details.

## ✅ Verify Installation

Run this Python script to verify SAM modules are correctly installed:

```python
import sys
sys.path.insert(0, '.')

try:
    from SAM.modeling.mask_decoder import MaskDecoder
    from SAM.modeling.prompt_encoder import PromptEncoder
    from SAM.modeling.transformer import TwoWayTransformer
    from SAM.modeling.image_encoder import ImageEncoderViT
    from SAM.modeling.common import LayerNorm2d
    print("✅ All SAM modules imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please follow the setup guide above.")
```

## 🔧 Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'SAM'`

**Solution**: Make sure you've created the `SAM/modeling/` directory and copied all required files.

### Issue: `ImportError: cannot import name 'ImageEncoderViT'`

**Solution**: Check that `SAM/modeling/image_encoder.py` exists and contains the `ImageEncoderViT` class.

### Issue: Adapter encoder not found

**Solution**: Our modified adapter encoder is in `models/adapter_encoder.py`, not in SAM directory. The import in `models/waverNet.py` should be:

```python
from SAM.modeling.adapter_encoder import ImageEncoderViT as Adapter
```

If this fails, check that you've copied our custom `adapter_encoder.py` to `SAM/modeling/`.

## 📚 Additional Resources

- [SAM Official Repository](https://github.com/facebookresearch/segment-anything)
- [SAM Paper](https://arxiv.org/abs/2304.02643)
- [SAM Demo](https://segment-anything.com/)

## 💡 Notes

1. **License**: SAM is released under Apache 2.0 license. Please comply with their license terms.

2. **Version Compatibility**: WaveRNet is tested with SAM commit `6fdee8f` (April 2023). Newer versions should also work but haven't been extensively tested.

3. **Custom Modifications**: We've modified the image encoder to add adapter layers (`adapter_encoder.py`). This is our contribution and is included in the repository.

4. **Minimal Dependencies**: We only use SAM's modeling components, not the full SAM package, to keep dependencies minimal.

## 🆘 Still Having Issues?

If you encounter problems:

1. Check that your Python environment has all dependencies: `pip install -r requirements.txt`
2. Verify SAM modules are in the correct directory structure
3. Open an issue on our GitHub with the error message

---

**Quick Setup Summary:**

```bash
# 1. Install SAM
pip install git+https://github.com/facebookresearch/segment-anything.git

# 2. Download SAM weights
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
mv sam_vit_b_01ec64.pth pretrained/

# 3. Verify
python -c "from SAM.modeling import *; print('✅ SAM ready!')"
```
