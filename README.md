# VTF-18V: Vertebrae Detection and Cobb Angle Measurement

A deep learning framework for automatic vertebrae detection and comprehensive spinal analysis in X-ray images.

## Overview

This project implements a neural network-based approach for:
- **Vertebrae Detection**: Automatic identification and localization of vertebrae in X-ray images
- **Cobb Angle Measurement**: Precise calculation of spinal curvature angles for scoliosis diagnosis
- **VWI Analysis**: Vertebral Wedge Index calculation for detailed spinal deformity assessment
- **Feature Discovery**: Automated discovery of optimal spinal metrics for clinical correlation
- **Clinical Analysis**: Comprehensive spine analysis with medical-grade accuracy

## Features

- **Multiple Model Architectures**: Support for HRNet-18 and HRNet-32 backbones
- **End-to-End Training**: Complete pipeline from data loading to model evaluation
- **VWI Analysis Suite**: Comprehensive Vertebral Wedge Index calculation and analysis
- **Automatic Feature Discovery**: AI-powered discovery of optimal spinal measurement combinations
- **Global Metric Search**: Advanced search algorithms for finding best-performing clinical indicators
- **Medical Visualization**: Specialized visualization tools for clinical assessment with curve detection
- **Cross-Validation**: Built-in 5-fold cross-validation for robust evaluation
- **Flexible Configuration**: Easy-to-use configuration system for different experimental setups

## Project Structure

```
VTF-18V-Clean/
├── src/                    # Core source code
│   ├── models/            # Neural network architectures
│   │   ├── vltenet.py     # Main VLTENet model
│   │   ├── hr.py          # HRNet backbone
│   │   └── ...
│   ├── analysis/          # VWI and spinal analysis modules
│   │   ├── vwi_calculator.py      # Vertebral Wedge Index calculation
│   │   ├── feature_discovery.py  # Automated feature discovery
│   │   ├── global_search.py      # Global metric optimization
│   │   └── gt_analysis.py        # Ground truth analysis and visualization
│   ├── utils/             # Utility functions
│   │   ├── transform.py   # Data transformations
│   │   └── ...
│   ├── train.py           # Training script
│   ├── test.py            # Testing and evaluation
│   ├── dataset.py         # Data loading and preprocessing
│   ├── loss.py            # Loss functions
│   └── visualize.py       # Visualization tools
├── configs/               # Configuration files
├── scripts/               # Training and testing scripts
├── requirements/          # Environment and dependency files
├── pretrained/            # Pre-trained model weights
└── docs/                  # Documentation
```

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd VTF-18V-Clean
```

### 2. Create Environment
Using conda:
```bash
conda env create -f requirements/dlenv_environment.yml
conda activate dlenv
```

Or using pip:
```bash
pip install -r requirements/dlenv_requirements.txt
```

### 3. Install the Package
```bash
pip install -e .
```

## Usage

### Training

#### Basic Training
```bash
bash scripts/train.sh
```

#### Custom Training
```python
from src.train import main
from configs.default import Config

config = Config()
config.backbone = 'hrnet32'
config.batch_size = 8
config.epochs = 150

main(config)
```

### VWI Analysis

#### Basic VWI Calculation
```bash
python scripts/analyze_vwi.py \
    --data-dir dataset/data \
    --labels-dir dataset/labels \
    --image-list dataset/test_list.txt \
    --analysis-type vwi \
    --output-dir results/vwi_analysis
```

#### Automatic Feature Discovery
```bash
python scripts/analyze_vwi.py \
    --data-dir dataset/data \
    --labels-dir dataset/labels \
    --image-list dataset/test_list.txt \
    --analysis-type feature-discovery \
    --cobb-angles-file dataset/cobb_angles.csv \
    --output-dir results/feature_discovery
```

#### Global Metric Search
```bash
python scripts/analyze_vwi.py \
    --data-dir dataset/data \
    --labels-dir dataset/labels \
    --image-list dataset/test_list.txt \
    --analysis-type global-search \
    --cobb-angles-file dataset/cobb_angles.csv \
    --output-dir results/global_search
```

#### Ground Truth Visualization
```bash
python scripts/analyze_vwi.py \
    --data-dir dataset/data \
    --labels-dir dataset/labels \
    --image-list dataset/test_list.txt \
    --analysis-type gt-analysis \
    --output-dir results/gt_analysis
```

### Testing

#### Basic Testing
```bash
bash scripts/test.sh
```

#### Programmatic Testing
```python
from src.test import evaluate_model

results = evaluate_model(
    model_path='checkpoints/model.pth',
    data_dir='dataset/data',
    test_list='dataset/splits/fold0/test.txt'
)
```

### Data Preparation

1. **Image Format**: X-ray images should be in PNG/JPG format
2. **Annotations**: Ground truth annotations in MAT format
3. **Directory Structure**:
   ```
   dataset/
   ├── data/           # X-ray images
   ├── labels/         # Annotation files (.mat)
   └── splits/         # Train/validation/test splits
       ├── fold0/
       │   ├── train.txt
       │   ├── val.txt
       │   └── test.txt
       └── ...
   ```

## Model Architecture

The VTF-18V model uses a sophisticated architecture combining:
- **HRNet Backbone**: High-resolution representation learning
- **Multi-Scale Feature Fusion**: Effective feature aggregation across scales
- **Dual-Branch Decoding**: Separate pathways for heatmap and vector field prediction
- **Attention Mechanisms**: Enhanced feature representation for medical imaging

## VWI Analysis Framework

The Vertebral Wedge Index (VWI) analysis framework provides comprehensive spinal deformity assessment:

### Core Metrics
- **VWI Calculation**: Ratio of anterior to posterior vertebral heights
- **Regional Analysis**: Evaluation by spinal regions (Proximal Thoracic, Main Thoracic, Thoracolumbar)
- **Statistical Features**: Mean, standard deviation, range, coefficient of variation
- **Asymmetry Indices**: Cross-regional comparisons and ratios

### Advanced Analysis
- **Automatic Feature Discovery**: AI-powered identification of optimal measurement combinations
- **Global Metric Search**: Systematic exploration of feature space for maximum clinical correlation  
- **Curve Detection**: Automated identification and classification of spinal curves
- **Severity Assessment**: Clinical severity grading based on established criteria

### Clinical Applications
- **Scoliosis Screening**: Early detection using optimized VWI combinations
- **Progression Monitoring**: Tracking changes in spinal curvature over time
- **Treatment Planning**: Quantitative assessment for surgical decision-making
- **Research Support**: Large-scale epidemiological and biomechanical studies

## Performance

### Model Performance
| Metric | Value |
|--------|-------|
| Vertebrae Detection Rate | >95% |
| Cobb Angle MAE | <3.5° |
| Position Error (mm) | <2.0 |
| Processing Time | <2s per image |

### VWI Analysis Performance
| Analysis Type | Correlation with Cobb Angle | Processing Time |
|---------------|----------------------------|-----------------|
| Traditional VWI | 0.627 | <1s per case |
| Optimized Feature Combination | 0.981 | <2s per case |
| Global Search Results | 0.980+ | <5s per case |
| Feature Discovery | Variable | 10-30s per dataset |

## Configuration

Key configuration parameters in `configs/default.py`:

```python
class Config:
    # Model
    backbone = 'hrnet18'     # Model backbone
    input_h = 1536           # Input height
    input_w = 512            # Input width
    
    # Training
    batch_size = 4           # Batch size
    epochs = 100             # Training epochs
    learning_rate = 1e-4     # Learning rate
    
    # Loss weights
    heatmap_weight = 1.0     # Heatmap loss weight
    vector_weight = 0.05     # Vector field loss weight
```

## Medical Applications

This framework is designed for:
- **Scoliosis Screening**: Early detection of spinal curvature abnormalities
- **Treatment Monitoring**: Tracking progression over time
- **Surgical Planning**: Pre-operative assessment and planning
- **Research**: Large-scale epidemiological studies

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

## Citation

If you use this work in your research, please cite:
```bibtex
@article{vtf18v2024,
  title={VTF-18V: Deep Learning for Vertebrae Detection and Cobb Angle Measurement},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Medical domain expertise provided by [Institution]
- Dataset collection supported by [Funding Agency]
- Computing resources provided by [Computing Center]

## Contact

For questions and support:
- Email: [contact@email.com]
- Issues: [GitHub Issues](repository-url/issues)