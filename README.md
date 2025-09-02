# ECG Arrhythmia Detection

This project is a system for detecting arrhythmia in ECG signals. It uses a deep learning model to classify heartbeats from time-frequency representations of ECG data. The system includes a full training pipeline.

> **Disclaimer**
> This project is for learning purposes and not for professional usage as the accuracy and macro F1 score is subpar.

## How It Works

1.  **Preprocessing**: Raw ECG signals (from the MIT-BIH database) are filtered, resampled, and segmented into 10-second windows.
2.  **Feature Extraction**: Each window is converted into a 2D time-frequency image using the Stockwell Transform (S-Transform).
3.  **Classification**: A ConvNeXt v2 model classifies the data into different arrhythmia types.
4.  **Training**: The model is trained using techniques like Focal Loss and SMOTE to attempt to improve the natural class imbalance found in ECG data.
5.  **ONNX Export**: After training, the best model is automatically exported to both standard and quantized ONNX formats for optimized inference and deployment.

## Technology Stack

*   **ML**: Python, PyTorch, Optuna, NumPy, SciPy, scikit-learn
*   **Data**: `wfdb` for the MIT-BIH Arrhythmia Database

## Project Structure

```
ecg/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── ecg_cli.py                  # Command-line interface
├── system_verification.py      # Installation and system tests
├── data/                       # ECG patient data (NPZ format)
│   ├── patient_101.npz
│   └── ...
├── src/                        # Core ML implementation
│   ├── config.py               # Configuration and parameters
│   ├── model.py                # ConvNeXt v2 model architecture
│   ├── training_pipeline.py    # Training and evaluation pipeline
│   ├── signal_processing.py    # S-Transform and feature extraction
│   └── ...
├── frontend/                   # React web application
│   ├── src/
│   ├── package.json
│   └── README.md
├── models/                     # Trained model storage
│   ├── best_model_fold_X.pth  # PyTorch model checkpoints
│   ├── best_ecg_model_fold_X.onnx         # Standard ONNX models
│   └── best_ecg_model_fold_X_quantized.onnx  # Quantized ONNX models
└── results/                    # Training results and HPO studies
```

## Installation and Setup

### Prerequisites

*   Python 3.8+
*   Node.js 14+ (for frontend)
*   Git

### Backend Setup

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd ecg
    ```

2.  **Install Python dependencies**:
    ```bash
    python -m pip install -r requirements.txt
    ```

3.  **Verify installation**:
    ```bash
    python ecg_cli.py test
    ```

## Usage

The project is managed through a command-line interface.

```bash
# Install dependencies and download data
python ecg_cli.py install

# Run system verification tests
python ecg_cli.py test

# Train the model using cross-validation
python ecg_cli.py train

# Run hyperparameter optimization
python ecg_cli.py hpo

# Evaluate the best trained model
python ecg_cli.py eval
```

### ONNX Model Export

After training completes, the system automatically exports the best-performing model to ONNX format:

- **Standard ONNX** (`best_ecg_model_fold_X.onnx`): Full precision model for maximum accuracy

- **Quantized ONNX** (`best_ecg_model_fold_X_quantized.onnx`): INT8 quantized model for optimized inference

## Acknowledgements and Citations

The architecture and methods used in this project are inspired by recent academic research in ECG classification.


> Talukder, M.A., Khalid, M., Kazi, M. et al. A hybrid cardiovascular arrhythmia disease detection using ConvNeXt-X models on electrocardiogram signals. *Sci Rep* **14**, 30366 (2024). [https://doi.org/10.1038/s41598-024-81992-w](https://doi.org/10.1038/s41598-024-81992-w)

> El-Ghaish, H., Eldele, E. ECGTransForm: Empowering adaptive ECG arrhythmia classification framework with bidirectional transformer. *Biomedical Signal Processing and Control*, Volume 89, 105714 (2024). [https://doi.org/10.1016/j.bspc.2023.105714](https://doi.org/10.1016/j.bspc.2023.105714)

**3. Dataset**

This project uses the **MIT-BIH Arrhythmia Database**, made available by PhysioNet.

*   **DOI:** [https://doi.org/10.13026/C2F305](https://doi.org/10.13026/C2F305)

*   **For the Database:**
    > Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. *IEEE Eng in Med and Biol* 20(3):45-50 (May-June 2001). (PMID: 11446209)

*   **For PhysioNet:**
    > Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation [Online]*. 101 (23), pp. e215–e220.