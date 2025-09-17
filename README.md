# PCG_FTSeg: Heart Sound Segmentation using PhysioNet Datasets

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.10%2B-orange)](https://pytorch.org/)

This project implements a deep learning model for heart sound segmentation using PhysioNet datasets. It segments phonocardiogram (PCG) signals into different cardiac cycle phases including S1, systole, S2, and diastole.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [🗂️ Datasets](#️-datasets)
- [📦 Installation](#-installation)
- [🚀 Usage](#-usage)
- [🏗️ Model Architecture](#️-model-architecture)
- [📊 Results](#-results)
- [📁 Project Structure](#-project-structure)
- [🙏 Acknowledgements](#-acknowledgements)

## Project Overview

This repository contains the implementation of a segmentation model for heart sound signals (phonocardiograms). The model is designed to identify and segment different components of the cardiac cycle from PCG recordings, which is crucial for diagnosing various heart conditions.

Key features:
- Multi-scale convolutional blocks for feature extraction
- Self-Attention and Cross-Attention mechanisms
- Support for both 2016 and 2022 PhysioNet challenge datasets
- Fourier Transform modules for frequency domain analysis

## 🗂️ Datasets

- **Internal Dataset (Train/Validation):**
    - Uses the publicly available [PhysioNet Computing in Cardiology Challenge 2016](https://physionet.org/content/challenge-2016/1.0.0/) dataset
    - Contains heart sound recordings with annotations for cardiac cycle phases
    - Used for training and internal validation of the model

- **External Datasets (Test):**
    - Public [PhysioNet George B. Moody PhysioNet Challenge 2022](https://physionet.org/content/challenge-2022/1.0.0/) dataset
    - Used for external testing and evaluation of model generalization

### Dataset Details

The heart sound recordings are segmented into four main components:
1. **S1**: First heart sound (occurs at the beginning of isovolumetric ventricular contraction)
2. **Systole**: Period between S1 and S2
3. **S2**: Second heart sound (occurs at the beginning of diastole)
4. **Diastole**: Period between S2 and the next S1

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/PCG_FTSeg.git
cd PCG_FTSeg
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 1.10+
- MONAI 0.9+
- PyTorch Lightning 1.6+
- NumPy, SciPy
- Scikit-learn
- NeuroKit2
- Matplotlib

## 🚀 Usage

### Data Preprocessing

Before training or inference, you need to preprocess the datasets:

For the 2016 dataset:
```bash
python preprocessing/signal_2016_and_AMC.py
```

For the 2022 dataset:
```bash
python preprocessing/signal_2022.py
```

### Training

To train the model with default settings:
```bash
python main.py
```

To train with specific configurations:
```bash
python main.py --gpu "0" --ver 45 --featureLength 4096 --target_sr 1000 --lowpass "240" --year 2016 --multi --use_sa --use_ca --mlp_expansion 4.0 --mlp_dropout 0.1 --batch 32 --seed 42
```

### Inference

To run inference on the 2022 external dataset:
```bash
python main.py --infer --infer_2022 --ver 45
```

### Configuration Options

Key configuration arguments:
- `--gpu`: GPU device ID (e.g., "0,1" for multi-GPU)
- `--ver`: Experiment version for log folder naming
- `--featureLength`: Length of input signal segment
- `--target_sr`: Target sampling rate of PCG signals
- `--lowpass`: Low-pass filter cutoff frequency
- `--year`: Dataset year (2016/2022)
- `--infer`: Enable inference mode
- `--multi`: Enable multi-scale convolution blocks
- `--use_sa`: Enable 8-head Self-Attention
- `--use_ca`: Enable 8-head Cross-Attention
- `--mlp_expansion`: MLP channel expansion ratio
- `--mlp_dropout`: MLP dropout rate
- `--batch`: Training batch size

## 🏗️ Model Architecture

The model architecture includes:
1. **Multi-Scale Convolution Blocks**: Extract features at different scales using dilated convolutions
2. **Self-Attention Mechanism**: Capture long-range dependencies in the signal
3. **Cross-Attention Mechanism**: Enhance feature representation through attention
4. **MLP Blocks**: Process features with expanded dimensions
5. **Four-class Segmentation Head**: Output segmentation for S1, systole, S2, and diastole

![Model Architecture](image/model_figure.png)

## 📊 Results

The model achieves competitive performance on both internal and external test sets. For detailed results and comparisons with other methods, please refer to the evaluation scripts and metrics implemented in this repository.

Performance metrics include:
- Dice coefficient
- Jaccard index
- Precision and recall for each cardiac phase

## 📁 Project Structure

```
PCG_FTSeg/
├── config.py                    # Configuration arguments
├── dataset.py                   # Data loading and preprocessing
├── main.py                      # Main training/inference script
├── model.py                     # Model architecture definition
├── modules.py                   # Custom neural network modules
├── utils.py                     # Utility functions
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── image/
│   └── model_figure.png         # Model architecture diagram
└── preprocessing/
    ├── signal_2016_and_AMC.py    # Preprocessing script for 2016 dataset
    └── signal_2022.py           # Preprocessing script for 2022 dataset
```

## 🙏 Acknowledgements

- This project uses data from [PhysioNet](https://physionet.org/).
- We acknowledge the organizers and contributors of the PhysioNet Computing in Cardiology Challenges 2016 and 2022.
- The implementation is based on PyTorch, MONAI, and PyTorch Lightning frameworks.

When using this code or the datasets, please cite the appropriate sources:

For the 2016 dataset:
```
Liu C, Springer D, Li Q, et al. An open access database for the evaluation of heart sound algorithms. Physiol Meas. 2016 Dec;37(12):2181-2213.
```

For the 2022 dataset:
```
Reyna M, Kiarashi Y, Elola A, et al. Heart Murmur Detection from Phonocardiogram Recordings: The George B. Moody PhysioNet Challenge 2022. PhysioNet. 2023.
```

For PhysioNet in general:
```
Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation. 2000;101(23):e215–e220.
```
