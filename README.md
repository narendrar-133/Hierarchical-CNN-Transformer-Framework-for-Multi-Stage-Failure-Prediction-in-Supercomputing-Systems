# Hierarchical CNN-Transformer Framework for Multi-Stage Failure Prediction in Supercomputing Systems

## Overview

This project implements an advanced deep learning framework designed to predict hardware and system failures in supercomputing environments across multiple severity stages. By combining Convolutional Neural Networks (CNNs) and Transformer architectures, the model captures both local temporal patterns and long-range dependencies in system logs and telemetry data.

Traditional failure prediction often treats the problem as binary (failure vs. non-failure). This framework extends the paradigm to a multi-stage prediction model:
- **0: Normal Operation** (typically ~96% of data)
- **1: Early Warning** (subtle anomalies, highly imbalanced ~0.01%)
- **2: Pre-Failure** (imminent issue ~0.05%)
- **3: Failure** (critical state ~3.6%)

## Key Features

- **Hybrid CNN-Transformer Architecture**: Utilizes 1D-CNNs for local feature extraction from time-series sequences and multi-head attention (Transformers) to understand long-term system degradation.
- **Robust Class Imbalance Handling**: Implements a custom `WeightedMSELoss` that penalizes misclassifications for rare but critical stages (Early Warning, Pre-Failure) to ensure the model doesn't simply collapse into predicting "Normal" for everything.
- **Strict Temporal Validation**: Employs chronological train/validation splitting (e.g., first 80% for training, last 20% for testing) to prevent data leakage from the future, ensuring the model performs realistically in a production setting.
- **Stable Training Dynamics**: Features learning rate scheduling (ReduceLROnPlateau) and parameter tuning optimized for stable convergence on highly skewed distributions.

## Project Structure

- `code/`: Contains all scripts for data processing, model definition, training, and evaluation.
  - `novel_models.py`: Defines the `HybridCNNTransformer` model architecture.
  - `train_model_corrected.py`: The main training script with fixes for class imbalance, temporal splitting, and stable learning rate scheduling.
  - `create_sequence.py`: Transforms raw tabular or log data into temporal sequences using sliding windows.
  - `evaluate_model.py`: Script to generate comprehensive metrics (MAE, R², Confusion Matrices) and analyze per-class performance.
- `dataset/`: Directory for storing raw data samples (e.g., `sample.csv`).
- `models/`: Directory where trained PyTorch model checkpoints and artifacts are saved.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch (with CUDA support strongly recommended)
- NumPy, scikit-learn, etc.

To install dependencies, you can use:
```bash
pip install -r requirements.txt
```
*(If `requirements.txt` is missing, ensure `torch`, `numpy`, and `scikit-learn` are installed).*

### Data Preparation

System logs must be converted into numerical features and temporal sequences.
```bash
python code/create_sequence.py
```
This script will produce `X_sequences_features.npy` and `y_labels_features.npy` along with `sequence_metadata.pkl`.

### Training

To train the model on your sequences:
```bash
python code/train_model_corrected.py
```
This script will automatically detect CUDA devices, load the sequences, perform a temporal split, and train the `HybridCNNTransformer`. The best model checkpoint will be saved as `best_model_CORRECTED.pth` based on balanced accuracy.

### Evaluation

To evaluate a trained model and generate classification reports and confusion matrices:
```bash
python code/evaluate_model.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
