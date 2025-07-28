# Quick Start Guide

This guide will help you get up and running with VTF-18V in just a few minutes.

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 10GB+ free disk space

## Installation

### 1. Clone and Setup
```bash
git clone <repository-url>
cd VTF-18V-Clean
pip install -e .
```

### 2. Download Pre-trained Models
```bash
# Download pre-trained weights (if available)
wget <model-url> -O pretrained/hrnet18_pretrained.pth
```

## Quick Test

### 1. Prepare Sample Data
Place your X-ray images in `dataset/data/` and annotations in `dataset/labels/`.

### 2. Run Inference
```bash
python src/test.py \
    --model_path pretrained/hrnet18_pretrained.pth \
    --data_dir dataset/data \
    --test_list dataset/test_samples.txt \
    --visualize
```

### 3. View Results
Results will be saved in the `results/` directory with:
- Vertebrae detection overlays
- Cobb angle measurements
- Statistical analysis

## Training Your Own Model

### 1. Prepare Training Data
```bash
# Organize your dataset
dataset/
├── data/           # X-ray images (.png, .jpg)
├── labels/         # Annotations (.mat files)
└── splits/         # Train/val/test splits
    └── fold0/
        ├── train.txt
        ├── val.txt
        └── test.txt
```

### 2. Start Training
```bash
bash scripts/train.sh
```

Or customize training:
```python
from src.train import main
from configs.default import Config

config = Config()
config.backbone = 'hrnet18'
config.epochs = 50
config.batch_size = 2  # Reduce if GPU memory is limited

main(config)
```

### 3. Monitor Training
```bash
tensorboard --logdir logs/
```

## Common Issues

### GPU Memory Issues
- Reduce batch size: `config.batch_size = 2`
- Use smaller input size: `config.input_h = 1024`

### Missing Dependencies
```bash
pip install torch torchvision opencv-python scipy matplotlib
```

### Data Format Issues
Ensure your annotations follow the expected MAT file format with vertebrae coordinates.

## Next Steps

1. **Read the full documentation**: See README.md for detailed usage
2. **Explore configurations**: Check `configs/default.py` for all options
3. **Try different models**: Experiment with HRNet-32 for better accuracy
4. **Customize for your data**: Adapt the dataset loader for your specific format

## Getting Help

- Check the [Issues](repository-url/issues) for common problems
- Read the full documentation in `docs/`
- Contact us at [email]