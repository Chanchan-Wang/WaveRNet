# WaveRNet: Wavelet-Guided Frequency Learning for Domain-Generalized Retinal Vessel Segmentation

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/xxxx.xxxxx)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Official PyTorch implementation of **WaveRNet** from the paper:

> **WaveRNet: Wavelet-Guided Frequency Learning for Multi-Source Domain-Generalized Retinal Vessel Segmentation**  
> Chanchan Wang, et al.  
> *Expert Systems with Applications*, 2025

## 🌟 Highlights

- **Spectral-guided Domain Modulator (SDM)**: Integrates wavelet decomposition with learnable domain tokens
- **Frequency-Adaptive Domain Fusion (FADF)**: Intelligent test-time domain selection via frequency similarity
- **Hierarchical Mask-Prompt Refiner (HMPR)**: Progressive coarse-to-fine refinement with long-range dependency modeling

## 📊 Results

Performance on Leave-One-Domain-Out (LODO) protocol:

| Method | DRIVE | STARE | CHASE_DB1 | RECOVERY-FA19 | Average |
|--------|-------|-------|-----------|---------------|---------|
| WaveRNet | **78.55** | **81.06** | **76.58** | **41.75** | **69.49** |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/WaveRNet.git
cd WaveRNet

# Create conda environment
conda create -n waverNet python=3.8
conda activate waverNet

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

1. Download the datasets:
   - [DRIVE](https://drive.grand-challenge.org/)
   - [STARE](http://cecas.clemson.edu/~ahoover/stare/)
   - [CHASE_DB1](https://blogs.kingston.ac.uk/retinal/chasedb1/)
   - [RECOVERY-FA19](https://github.com/rmaphoh/RVD_Challenge)

2. Organize the data structure:
```
data/
├── DRIVE/
│   ├── training/
│   └── test/
├── STARE/
├── CHASE_DB1/
└── RECOVERY-FA19/
```

### Training

```bash
# Train on LODO protocol (leave DRIVE out)
python train.py --config configs/config_test_drive.json --gpu 0

# Train on all domains
python train.py --config configs/config_mixed.json --gpu 0
```

### Evaluation

```bash
# Evaluate on DRIVE dataset
python eval.py --config configs/config_test_drive.json --checkpoint pretrained/model_best.pth

# Evaluate on all datasets
bash scripts/eval_all.sh
```

## 📁 Project Structure

```
WaveRNet/
├── models/
│   ├── waverNet.py          # Main WaveRNet model
│   ├── sdm.py               # Spectral-guided Domain Modulator
│   ├── fadf.py              # Frequency-Adaptive Domain Fusion
│   └── hmpr.py              # Hierarchical Mask-Prompt Refiner
├── datasets/
│   └── retinal_dataset.py   # Dataset loader
├── utils/
│   ├── loss.py              # Loss functions
│   └── transforms.py        # Data augmentation
├── configs/                  # Configuration files
├── train.py                  # Training script
├── eval.py                   # Evaluation script
└── README.md
```

## 🔧 Requirements

- Python >= 3.8
- PyTorch >= 1.10.0
- CUDA >= 11.3
- See `requirements.txt` for full dependencies

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@article{wang2025waverNet,
  title={WaveRNet: Wavelet-Guided Frequency Learning for Multi-Source Domain-Generalized Retinal Vessel Segmentation},
  author={Wang, Chanchan and others},
  journal={Expert Systems with Applications},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- SAM pretrained weights from [Segment Anything](https://github.com/facebookresearch/segment-anything)
- Dataset providers: DRIVE, STARE, CHASE_DB1, RECOVERY-FA19

## 📧 Contact

For questions and feedback, please contact: wusheng070@gmail.com
