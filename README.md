# TransParaX

A Transformer-based framework for efficient device I-V parameter extraction, featuring semi-supervised learning, hierarchical attention architecture, and Bayesian optimization refinement.

## Overview

TransParaX is a novel framework that achieves breakthrough performance in semiconductor device parameter extraction. Key features include:

- Mean Relative Error (MRE) of 3.43% on GaN HEMTs
- 30x speed increase over traditional methods
- 1.7x accuracy improvement over CNN-LSTM approaches
- Efficient transfer learning with only 1k labeled samples
- 94.3% confidence interval coverage rate

## Project Structure

```
TransParaX/
├── baseline/           # Baseline model implementations
├── dmodel/            # Device modeling components
├── main.py            # Main training and evaluation script
├── requirements.txt   # Project dependencies
└── TransParaX/        # Core framework implementation
    ├── models.py      # Network structure
    ├── loss.py
    ├── BO.py
    ├── train.py
    ├── data.py
```

## Installation

1. Create a new Python environment:
```bash
conda create -n transparax python=3.12
conda activate transparax
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset
*Due to the dataset being for commercial use, we only provide the data collection method in the dmodle folder.
## Key Components

### 1. Hierarchical Transformer Architecture
- Local Feature Enhancement Layer with SE blocks
- Temporal Relationship Layer with multi-head attention
- Global Curve Interaction Layer with adapter-based transfer learning

### 2. Semi-supervised Learning
- Adaptive pseudo-labeling mechanism
- Uncertainty-guided sample selection
- Dynamic thresholding based on validation accuracy

### 3. Bayesian Optimization Refinement
- Uncertainty-guided initial sampling
- Hybrid kernel incorporating transformer predictions
- Adaptive acquisition function

## Performance

| Method | MRE (%) | RMSE | Time(s) | CICR(%) |
|--------|---------|------|---------|----------|
| GA-based | 3.75 | 0.113 | >3600 | N/A |
| CNN-LSTM | 5.78 | 0.196 | 1.5 | N/A |
| TransParaX | 3.43 | 0.086 | 124 | 94.3 |

On A800 40G

## Transfer Learning

Supports efficient cross-process adaptation:
- Fine-tune with only 1k labeled samples
- 84% reduction in training time
- Comparable performance to full training with 100k samples