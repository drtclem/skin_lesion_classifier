import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns

from dataset import SkinLesionDataset, get_transforms, CLASSES
from model import get_model, get_hybrid_model

# Paths
RAW_DIR  = "../data/raw"
IMG_DIR  = os.path.join(RAW_DIR, "ISIC_2019_Training_Input")
CSV_PATH = os.path.join(RAW_DIR, "ISIC_2019_Training_GroundTruth.csv")


def evaluate(model_type='hybrid'):
    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    print(f"Evaluating model: {model_type}\n")

    # ── Recreate the same val split used during training ──────────────────
    # random_state=42 and test_size=0.2 must match train.py exactly.
    df = pd.read_csv(CSV_PATH)
    df = df.drop(columns=['UNK'], errors='ignore')
    _, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df[CLASSES].values.argmax(axis=1),
        random_state=42
    )

    _, val_transform = get_transforms()
    val_dataset = SkinLesionDataset(val_df, IMG_DIR, transform=val_transform)
    val_loader  = DataLoader(val_dataset, batch_size=16, shuffle=False,
                             num_workers=4, pin_memory=True)

    # ── Load model ────────────────────────────────────────────────────────
    model_path = f"../data/best_model_{model_type}.pth"

    if model_type == 'baseline':
        model = get_model(num_classes=8, device=device)
    else:
        model = get_hybrid_model(num_classes=8, device=device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded weights from {model_path}\n")

    # ── Run predictions ───────────────────────────────────────────────────
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── Metrics ───────────────────────────────────────────────────────────
    accuracy = 100 * (all_preds == all_labels).sum() / len(all_labels)
    print(f"Overall Val Accuracy: {accuracy:.2f}%\n")
    print("Per-Class Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASSES))

    # ── Confusion matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f'Confusion Matrix — {model_type}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    out_path = f'../data/confusion_matrix_{model_type}.png'
    plt.savefig(out_path)
    plt.show()
    print(f"Confusion matrix saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        choices=['baseline', 'hybrid'],
        default='hybrid',
        help="Which model to evaluate: 'baseline' or 'hybrid'"
    )
    args = parser.parse_args()
    evaluate(model_type=args.model)
