"""
Integrated training pipeline for the ECG classification model.

This module handles the end-to-end training process, including data loading,
class imbalance handling with SMOTE, model instantiation, training loop,
evaluation, and patient-wise cross-validation.
"""

import os
import random
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE

try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX libraries not available. Install with: pip install onnx onnxruntime")

from config import *
from dataset import ECGDataset
from model import create_ecg_model
from loss import FocalLoss
from preprocess import create_segmented_windows


def load_best_hpo_params():
    """
    Loads the best hyperparameters from the most recent HPO study.
    
    Returns:
        dict or None: Best hyperparameters if found, None otherwise.
    """
    try:
        import optuna
        storage_name = f"sqlite:///{os.path.join(RESULTS_DIRECTORY, 'ecg_hpo_study.db')}"
        study = optuna.load_study(study_name='ecg_hpo_study', storage=storage_name)
        
        if len(study.trials) > 0:
            best_params = study.best_params.copy()
            best_value = study.best_value
            return best_params, best_value
    except Exception as e:
        print(f"Could not load HPO results: {e}")
    
    return None, None


def set_seed(seed: int):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def load_data(data_dir: str) -> dict:
    """Loads each training patient's windows, caching them in data_dir. Returns {patient_id: (X, y)}."""
    print("Loading patient data for cross-validation...")
    patient_data = {}
    for patient_id in tqdm(TRAINING_PATIENTS, desc="Loading Patients"):
        cache_path = os.path.join(data_dir, f"patient_{patient_id}.npz")
        if os.path.exists(cache_path):
            with np.load(cache_path) as data:
                X, y = data['x'], data['y']
        else:
            X, y = create_segmented_windows([patient_id])
            np.savez_compressed(cache_path, x=X, y=y)
        patient_data[patient_id] = (X, y)
    return patient_data


def apply_smote(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """
    Applies a conservative SMOTE to balance rare classes in the training data.

    Args:
        X_train: The training features.
        y_train: The training labels.

    Returns:
        A tuple of the potentially oversampled (X_train, y_train).
    """
    original_dist = {k: v for k, v in zip(*np.unique(y_train, return_counts=True))}
    n_majority = original_dist.get(0, 0)
    
    strategy = {
        cls: max(count, min(int(n_majority * 0.05), count * 2))
        for cls, count in original_dist.items() if cls != 0
    }
    
    if not any(new > old for new, old in zip(strategy.values(), [original_dist[k] for k in strategy.keys()])):
        print("SMOTE not applied: No class requires upsampling under the current strategy.")
        return X_train, y_train

    min_samples = min(v for k, v in original_dist.items() if k in strategy)
    k_neighbors = min(2, min_samples - 1)
    
    if k_neighbors < 1:
        print(f"SMOTE not applied: Not enough samples in minority class for k_neighbors={k_neighbors+1}.")
        return X_train, y_train
        
    print(f"Applying SMOTE... Original size: {len(y_train)}")
    smote = SMOTE(sampling_strategy=strategy, random_state=RANDOM_SEED, k_neighbors=k_neighbors)
    X_flat = X_train.reshape(X_train.shape[0], -1)
    X_res, y_res = smote.fit_resample(X_flat, y_train)
    print(f"SMOTE applied. New size: {len(y_res)}")
    return X_res.reshape(-1, 2, S_TRANSFORM_HEIGHT, WINDOW_SIZE_SAMPLES), y_res


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """Runs a single training epoch."""
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc="Training Epoch", leave=False)
    for data, targets in pbar:
        data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
            outputs = model(data)
            loss = criterion(outputs, targets)
        
        # Check for NaN loss early
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN or Inf loss detected: {loss.item()}")
            return float('nan')
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    """Evaluates the model on a given dataset."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, targets in tqdm(loader, desc="Evaluating", leave=False):
            data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(data)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    return f1_score(all_targets, all_preds, average='macro', zero_division=0)


def export_to_quantized_onnx(model, sample_input, output_path, model_name="ecg_model"):
    """
    Exports a PyTorch model to quantized ONNX format.
    
    Args:
        model: The trained PyTorch model
        sample_input: A sample input tensor for tracing the model
        output_path: Directory to save the ONNX files
        model_name: Base name for the ONNX files
    
    Returns:
        tuple: (onnx_path, quantized_onnx_path) if successful, (None, None) if failed
    """
    if not ONNX_AVAILABLE:
        print("ONNX libraries not available. Skipping ONNX export.")
        return None, None
    
    try:
        os.makedirs(output_path, exist_ok=True)
        
        # Set model to evaluation mode
        model.eval()
        
        # Export to ONNX format
        onnx_path = os.path.join(output_path, f"{model_name}.onnx")
        print(f"Exporting model to ONNX format: {onnx_path}")
        
        with torch.no_grad():
            torch.onnx.export(
                model,
                sample_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
        
        # Verify the ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model exported and verified successfully")
        
        # Quantize the ONNX model
        quantized_onnx_path = os.path.join(output_path, f"{model_name}_quantized.onnx")
        print(f"Quantizing ONNX model: {quantized_onnx_path}")
        
        quantize_dynamic(
            onnx_path,
            quantized_onnx_path,
            weight_type=QuantType.QUInt8
        )
        print("ONNX model quantized successfully")
        
        # Verify the quantized model can be loaded
        ort_session = ort.InferenceSession(quantized_onnx_path)
        print("Quantized ONNX model verified with ONNX Runtime")
        
        # Compare file sizes
        original_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
        quantized_size = os.path.getsize(quantized_onnx_path) / (1024 * 1024)  # MB
        compression_ratio = (1 - quantized_size / original_size) * 100
        
        print(f"📊 Model size comparison:")
        print(f"   Original ONNX: {original_size:.2f} MB")
        print(f"   Quantized ONNX: {quantized_size:.2f} MB")
        print(f"   Size reduction: {compression_ratio:.1f}%")
        
        return onnx_path, quantized_onnx_path
        
    except Exception as e:
        print(f"Error during ONNX export/quantization: {str(e)}")
        return None, None


def validate_onnx_model(model, onnx_path, sample_input, tolerance=1e-5):
    """
    Validates the ONNX model by comparing outputs with the original PyTorch model.
    
    Args:
        model: Original PyTorch model
        onnx_path: Path to the ONNX model
        sample_input: Sample input for testing
        tolerance: Tolerance for output comparison
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if not ONNX_AVAILABLE or not os.path.exists(onnx_path):
        return False
    
    try:
        # Get PyTorch model output
        model.eval()
        with torch.no_grad():
            pytorch_output = model(sample_input).cpu().numpy()
        
        # Get ONNX model output
        ort_session = ort.InferenceSession(onnx_path)
        ort_inputs = {ort_session.get_inputs()[0].name: sample_input.cpu().numpy()}
        onnx_output = ort_session.run(None, ort_inputs)[0]
        
        # Compare outputs
        max_diff = np.max(np.abs(pytorch_output - onnx_output))
        mean_diff = np.mean(np.abs(pytorch_output - onnx_output))
        
        print(f"ONNX validation - Max difference: {max_diff:.6f}, Mean difference: {mean_diff:.6f}")
        
        if max_diff < tolerance:
            print("NNX model validation passed")
            return True
        else:
            print(f"ONNX model validation failed (tolerance: {tolerance})")
            return False
            
    except Exception as e:
        print(f"Error during ONNX validation: {str(e)}")
        return False


def train_model(hyperparams=None, return_f1=False, trial_num=None):
    """
    Main training function that orchestrates the entire training process.

    Args:
        hyperparams (dict, optional): A dictionary of hyperparameters to override defaults.
        return_f1 (bool): If True, returns the final F1 score (for HPO).
        trial_num (int, optional): The HPO trial number, for logging purposes.
    """
    set_seed(RANDOM_SEED)
    
    # Load HPO results if available and no hyperparams provided
    if not hyperparams and not return_f1:  # Only for full training runs, not HPO trials
        print("\n" + "="*50)
        print("CHECKING FOR HPO RESULTS")
        print("="*50)
        
        best_hpo_params, best_hpo_score = load_best_hpo_params()
        
        if best_hpo_params:
            print(f"Found HPO results! Best F1 score: {best_hpo_score:.4f}")
            print("Best hyperparameters found:")
            for key, value in best_hpo_params.items():
                print(f"  {key}: {value}")
            
            response = input("\nUse these optimized hyperparameters? [Y/n]: ").strip().lower()
            if response in ['', 'y', 'yes']:
                hyperparams = best_hpo_params
                print("Using optimized hyperparameters from HPO")
            else:
                print("Using default hyperparameters")
        else:
            print("No HPO results found. Using default hyperparameters.")
            print("Tip: Run 'python ecg_cli.py hpo' to optimize hyperparameters first.")
        
        print("="*50)
    
    hp = hyperparams or {}
    lr = hp.get('learning_rate', LEARNING_RATE)
    bs = hp.get('batch_size', BATCH_SIZE)
    gamma = hp.get('focal_gamma', FOCAL_LOSS_GAMMA)
    epochs = hp.get('epochs', HPO_MAX_EPOCHS if return_f1 else MAX_EPOCHS)
    
    # Display the hyperparameters being used
    if not return_f1:  # Only for full training runs
        print("\n" + "="*50)
        print("TRAINING CONFIGURATION")
        print("="*50)
        print(f"Learning Rate: {lr}")
        print(f"Batch Size: {bs}")
        print(f"Focal Loss Gamma: {gamma}")
        print(f"Max Epochs: {epochs}")
        print(f"Device: {DEVICE}")
        print("="*50)
    
    log_prefix = f"[Trial {trial_num}] " if trial_num is not None else ""
    device = torch.device(DEVICE)
    patient_data = load_data(DATA_DIRECTORY)
    patients = list(patient_data.keys())
    kf = KFold(n_splits=CROSS_VALIDATION_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(patients)):
        print(f"\n{log_prefix}--- Starting Fold {fold + 1}/{CROSS_VALIDATION_FOLDS} ---")
        train_p = [patients[i] for i in train_idx]
        val_p = [patients[i] for i in val_idx]
        X_train = np.concatenate([patient_data[p][0] for p in train_p])
        y_train = np.concatenate([patient_data[p][1] for p in train_p])
        X_val = np.concatenate([patient_data[p][0] for p in val_p])
        y_val = np.concatenate([patient_data[p][1] for p in val_p])
        X_train_bal, y_train_bal = apply_smote(X_train, y_train)
        all_classes = np.arange(NUM_CLASSES)
        unique_classes_in_fold = np.unique(y_train_bal)
        fold_weights = compute_class_weight('balanced', classes=unique_classes_in_fold, y=y_train_bal)
        weights = np.ones(NUM_CLASSES, dtype=np.float32)
        for i, class_idx in enumerate(unique_classes_in_fold):
            weights[class_idx] = fold_weights[i]
        weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)
        print(f"{log_prefix}Classes in fold {fold + 1}: {unique_classes_in_fold}")
        print(f"{log_prefix}Class weights: {weights}")
        assert y_train_bal.min() >= 0 and y_train_bal.max() < NUM_CLASSES, \
            f"Invalid labels in training data: min={y_train_bal.min()}, max={y_train_bal.max()}, expected range=[0, {NUM_CLASSES-1}]"
        assert y_val.min() >= 0 and y_val.max() < NUM_CLASSES, \
            f"Invalid labels in validation data: min={y_val.min()}, max={y_val.max()}, expected range=[0, {NUM_CLASSES-1}]"
        train_ds = ECGDataset(X_train_bal, y_train_bal)
        val_ds = ECGDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
        model = create_ecg_model(NUM_CLASSES).to(device)
        criterion = FocalLoss(alpha=weights_tensor, gamma=gamma)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=8)
        scaler = GradScaler()
        best_f1 = 0.0
        patience = 0
        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
            
            # Check for NaN training loss and abort this fold early
            if np.isnan(train_loss):
                print(f"{log_prefix}NaN loss detected in fold {fold + 1}, epoch {epoch + 1}. Aborting this fold.")
                best_f1 = 0.0  # Set to 0 for failed folds
                break
                
            val_f1 = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_f1)
            print(f"{log_prefix}Fold {fold + 1}, Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}")
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience = 0
                if not hp: # Only save the best model during a full training run
                    os.makedirs(MODELS_DIRECTORY, exist_ok=True)
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'f1': val_f1
                    }, os.path.join(MODELS_DIRECTORY, f'best_model_fold_{fold+1}.pth'))
            else:
                patience += 1
            if patience > EARLY_STOPPING_PATIENCE:
                print(f"{log_prefix}Early stopping at epoch {epoch + 1}")
                break
        fold_scores.append(best_f1)
    mean_f1 = np.mean(fold_scores)
    std_f1 = np.std(fold_scores)
    print("\n" + "="*50)
    print("Cross-Validation Complete")
    print(f"F1 Scores per fold: {[f'{s:.4f}' for s in fold_scores]}")
    print(f"Mean F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")
    print("="*50)
    # Export the best model to quantized ONNX format
    if not hp:  # Only during full training runs (not HPO)
        print("\n" + "="*50)
        print("Exporting to Quantized ONNX")
        print("="*50)
        best_fold_idx = np.argmax(fold_scores)
        best_fold_model_path = os.path.join(MODELS_DIRECTORY, f'best_model_fold_{best_fold_idx+1}.pth')
        if os.path.exists(best_fold_model_path):
            best_model = create_ecg_model(NUM_CLASSES).to(device)
            checkpoint = torch.load(best_fold_model_path, map_location=device)
            best_model.load_state_dict(checkpoint['model_state_dict'])
            best_model.eval()
            sample_input = torch.randn(1, 2, S_TRANSFORM_HEIGHT, WINDOW_SIZE_SAMPLES).to(device)
            onnx_path, quantized_onnx_path = export_to_quantized_onnx(
                best_model, 
                sample_input, 
                MODELS_DIRECTORY, 
                f"best_ecg_model_fold_{best_fold_idx+1}"
            )
            if onnx_path:
                validate_onnx_model(best_model, onnx_path, sample_input)
            if quantized_onnx_path:
                validate_onnx_model(best_model, quantized_onnx_path, sample_input, tolerance=1e-3)
            print(f"Best model from fold {best_fold_idx+1} exported to ONNX format")
        else:
            print(f"Warning: Best model file not found at {best_fold_model_path}")
    return mean_f1 if return_f1 else None

if __name__ == "__main__":
    train_model()