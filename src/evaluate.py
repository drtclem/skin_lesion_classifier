import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns

from dataset import SkinLesionDataset, get_transforms, CLASSES
from model import get_model

# Paths
RAW_DIR = "../data/raw"
IMG_DIR = os.path.join(RAW_DIR, "ISIC_2019_Training_Input")
CSV_PATH = os.path.join(RAW_DIR, "ISIC_2019_Training_GroundTruth.csv")
MODEL_PATH = "../data/best_model.pth"

def evaluate():
    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Load data
    df = pd.read_csv(CSV_PATH)
    df = df.drop(columns=['UNK'], errors='ignore')

    # Recreate the same val split as training
    _, val_df = train_test_split(df, test_size=0.2,
                                  stratify=df[CLASSES].values.argmax(axis=1),
                                  random_state=42)

    # Dataset and loader
    _, val_transform = get_transforms()
    val_dataset = SkinLesionDataset(val_df, IMG_DIR, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Load model
    model = get_model(num_classes=8, device=device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully")

    # Run predictions
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Overall accuracy
    accuracy = 100 * (all_preds == all_labels).sum() / len(all_labels)
    print(f"\nOverall Val Accuracy: {accuracy:.2f}%")

    # Per class report
    print("\nPer Class Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASSES))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('../data/confusion_matrix.png')
    plt.show()
    print("\nConfusion matrix saved to data/confusion_matrix.png")

if __name__ == "__main__":
    evaluate()