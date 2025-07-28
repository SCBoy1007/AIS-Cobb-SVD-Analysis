# Accurate Cobb Angle Estimation via SVD-Based Curve Detection and Vertebral Wedging Quantification

A novel deep learning framework for automated assessment of Adolescent Idiopathic Scoliosis (AIS) using advanced computer vision techniques.

## Overview

This project implements the research presented in "Accurate Cobb Angle Estimation via SVD-Based Curve Detection and Vertebral Wedging Quantification" published in IEEE Journal of Biomedical and Health Informatics. The framework provides:

- **Dual-Task Vertebral Morphology Preservation**: Simultaneous prediction of superior and inferior endplate angles with corresponding midpoint coordinates for each vertebra
- **SVD-Based Principal Curve Detection**: Novel curve identification algorithm using Singular Value Decomposition to analyze angle predictions without predefined patterns
- **Vertebral Wedging Index (VWI)**: New metric quantifying vertebral deformation that complements traditional Cobb angle measurements
- **Biomechanically Informed Loss**: Incorporation of anatomical knowledge directly into model training
- **Clinical-Grade Accuracy**: 83.33% diagnostic accuracy with robust generalization across different clinical centers

## Key Features

- **HRNet + Swin Transformer Architecture**: High-resolution feature extraction with global context enhancement
- **Dual-Task Output Head**: Simultaneous prediction of endplate heatmaps and vector fields for improved accuracy
- **SVD-Based Curve Detection**: Automatic identification of spinal curves without predefined patterns using Singular Value Decomposition
- **Vertebral Wedging Index (VWI)**: Novel metric measuring vertebral deformation for enhanced clinical assessment
- **Biomechanically Informed Loss**: Anatomical constraints integrated into the loss function for physiologically plausible predictions
- **Multi-Center Validation**: Tested across multiple clinical centers with 630 full-spine anteroposterior radiographs
- **Superior Performance**: 83.33% diagnostic accuracy, 2.55° mean absolute error for Cobb angles
- **Clinical Interpretability**: Complete workflow from landmark detection to diagnostic classification

## Project Structure

```
JBHI-Cleaned/
├── src/                    # Core source code
│   ├── models/            # Neural network architectures
│   │   ├── hr.py          # HRNet backbone implementation
│   │   ├── transformer.py # Swin Transformer modules
│   │   ├── decoding.py    # Dual-task output head
│   │   └── ...           # Other model architectures
│   ├── analysis/          # VWI and spinal analysis modules
│   │   ├── vwi_calculator.py      # Vertebral Wedging Index calculation
│   │   ├── feature_discovery.py  # Automated feature discovery
│   │   ├── global_search.py      # Global metric optimization
│   │   └── gt_analysis.py        # Ground truth analysis and visualization
│   ├── utils/             # Utility functions
│   │   ├── transform.py   # Data transformations and augmentations
│   │   ├── vis_hm.py      # Heatmap visualization utilities
│   │   └── ...
│   ├── train.py           # Training script with biomechanically informed loss
│   ├── test.py            # Testing and evaluation
│   ├── dataset.py         # Data loading and preprocessing
│   ├── loss.py            # Multi-component loss functions
│   └── visualize.py       # Clinical visualization tools
├── configs/               # Configuration files
├── scripts/               # Training and testing scripts
│   ├── train.sh          # Training script
│   ├── test.sh           # Testing script
│   └── analyze_vwi.py    # VWI analysis script
├── requirements/          # Environment and dependency files
├── pretrained/            # Pre-trained model weights
├── results/               # Experimental results
└── docs/                  # Documentation
    ├── QUICKSTART.md     # Quick start guide
    └── VWI_ANALYSIS.md   # VWI analysis documentation
```

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/[username]/JBHI-Cleaned.git
cd JBHI-Cleaned
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

Our framework employs a sophisticated deep learning architecture that addresses key limitations of existing methods:

### Core Components

1. **HRNet Backbone**: High-resolution representation learning for multi-scale feature extraction at 1/4, 1/8, 1/16, and 1/32 resolutions
2. **Swin Transformer Enhancement**: Hierarchical window-based attention to capture long-range dependencies while maintaining computational efficiency
3. **Dual-Task Output Head**: Simultaneous prediction of:
   - 18 upper and 18 lower endplate heatmaps for landmark localization
   - 36-channel vector maps for direct angle regression
4. **Biomechanically Informed Loss**: Three-component loss function:
   - Heatmap loss (λ₁=1.0) for spatial localization
   - Vector loss (λ₂=0.05) for angle prediction
   - Constraint loss (λ₃=0.05) for anatomical consistency

### SVD-Based Curve Detection Algorithm

Our novel curve detection approach constructs an angle matrix Γ where:
```
Γ = θᵘᵖᵖᵉʳ1ᵀ - 1θᵀˡᵒʷᵉʳ
```

The matrix is decomposed using SVD to identify principal curvature patterns, enabling flexible detection of diverse scoliosis patterns without predefined constraints.

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

### Clinical Assessment Metrics
| Metric | Our Method | VLTENet | Seg4Reg |
|--------|------------|---------|---------|
| Max Cobb Angle MAE (°) | **2.55±0.20** | 2.89±0.23 | 3.24±0.24 |
| Diagnostic Accuracy (%) | **83.45±3.33** | 78.65±3.73 | 76.12±3.82 |
| Curve Detection Rate (%) | 97.84±0.90 | 98.72±0.73 | - |
| False Detection Rate (%) | **6.69±1.57** | 15.89±1.98 | - |

### Disease Severity Classification
| Severity | Precision (%) | Recall (%) | F1-Score (%) |
|----------|---------------|------------|--------------|
| **Severe** | 85.71 | 75.00 | 80.00 |
| **Moderate** | 81.08 | 89.55 | 85.11 |
| **Normal/Mild** | 86.67 | 76.47 | 81.25 |

### Model Performance Metrics
| Metric | HRNet-18 (Ours) | HRNet-32 | ResNet-50 |
|--------|-----------------|----------|-----------|
| Mean Position Error (px) | **4.74±0.80** | 4.76±1.05 | 5.74±2.10 |
| Mean Angle Error (°) | **2.10±0.08** | 2.13±0.10 | 2.16±0.10 |
| Parameters | 9.56M | 29.31M | 25.75M |

### VWI Analysis Capabilities
- **VWI Calculation**: Quantifies vertebral deformation within spinal curves
- **Prognostic Value**: Complements Cobb angles for enhanced clinical assessment
- **Multi-Pattern Detection**: Handles complex curve patterns including congenital scoliosis

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
@article{shi2024accurate,
  title={Accurate Cobb Angle Estimation via SVD-Based Curve Detection and Vertebral Wedging Quantification},
  author={Shi, Chang and Meng, Nan and Zhuang, Yipeng and Zhao, Moxin and Huang, Hua and Chen, Xiuyuan and Nie, Cong and Zhong, Wenting and Jiang, Guiqiang and Wei, Yuxin and Yu, Jacob Hong Man and Chen, Si and Ou, Xiaowen and Cheung, Jason Pui Yin and Zhang, Teng},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2024},
  publisher={IEEE}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Medical domain expertise provided by Department of Orthopaedics and Traumatology, The University of Hong Kong
- Dataset collection supported by Queen Mary Hospital and The Duchess of Kent Children's Hospital
- Funding supported by National Natural Science Foundation of China Young Scientists Fund (Grant ID: 82402398, 82303957)
- Computing resources provided by The University of Hong Kong

## Dataset

The JBHI-Cleaned dataset contains 630 full-spine anteroposterior radiographs from patients aged 10-18 years with Adolescent Idiopathic Scoliosis, collected from two tertiary medical centers in Hong Kong. Each radiograph includes:

- **Dual-rater annotations**: All cases annotated by two experienced orthopedic surgeons
- **18 vertebrae landmarks**: From C7 to L5 with 4 landmarks per vertebra
- **Quality assurance**: Cases with disagreement >5° underwent consensus review
- **Multi-center validation**: Ensures robust generalization across clinical settings

## Contact

For questions and support:
- Primary contact: Prof. Teng Zhang (tgzhang@hku.hk)
- Issues: [GitHub Issues](https://github.com/[username]/JBHI-Cleaned/issues)
- Institutional Review Board approval: UW15-596