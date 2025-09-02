"""
Model evaluation script for the ECG arrhythmia classification system.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from config import *
from preprocess import create_segmented_windows
from dataset import ECGDataset
from model import create_ecg_model


def plot_confusion_matrix(cm, class_names, save_path=None):
    """Renders and optionally saves a confusion matrix plot."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def evaluate_model(model, dataloader, device):
    """
    Evaluates the model on the provided dataloader.

    Returns:
        A tuple of (all_targets, all_predictions).
    """
    model.eval()
    all_predictions, all_targets = [], []
    with torch.no_grad():
        for data, targets in tqdm(dataloader, desc="Evaluating"):
            data = data.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    return np.array(all_targets), np.array(all_predictions)


def main():
    """Main evaluation function."""
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"Error: Model file not found at {BEST_MODEL_PATH}. Please run training first.")
        return
    
    device = torch.device(DEVICE)
    
    # Load and prepare test data.
    cache_path = os.path.join(DATA_DIRECTORY, "test_data.npz")
    if os.path.exists(cache_path):
        with np.load(cache_path) as data:
            X_test, y_test = data['x'], data['y']
    else:
        X_test, y_test = create_segmented_windows(TESTING_PATIENTS)
        np.savez_compressed(cache_path, x=X_test, y=y_test)

    test_dataset = ECGDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Load the trained model.
    model = create_ecg_model(NUM_CLASSES).to(device)
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    targets, predictions = evaluate_model(model, test_loader, device)
    
    # Generate and display results.
    report = classification_report(
        targets, predictions, target_names=CLASS_NAMES, digits=4, zero_division=0
    )
    print("\nEVALUATION RESULTS\n" + "="*60)
    print(report)
    
    cm = confusion_matrix(targets, predictions)
    os.makedirs(RESULTS_DIRECTORY, exist_ok=True)
    plot_confusion_matrix(cm, CLASS_NAMES, save_path=os.path.join(RESULTS_DIRECTORY, "confusion_matrix.png"))
    
    with open(os.path.join(RESULTS_DIRECTORY, "evaluation_report.txt"), 'w') as f:
        f.write(report)


if __name__ == "__main__":
    main()