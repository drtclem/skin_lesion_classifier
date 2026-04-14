import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from dataset import SkinLesionDataset, get_transforms, CLASSES
from model import get_model

# Paths
RAW_DIR = "../data/raw"
IMG_DIR = os.path.join(RAW_DIR, "ISIC_2019_Training_Input")
CSV_PATH = os.path.join(RAW_DIR, "ISIC_2019_Training_GroundTruth.csv")

def get_class_weights(df):
    counts = df[CLASSES].sum().values
    total = counts.sum()
    weights = total / (len(CLASSES) * counts)
    return torch.FloatTensor(weights)

def get_sampler(df):
    labels = df[CLASSES].values.argmax(axis=1)
    class_counts = np.bincount(labels)
    sample_weights = [1.0 / class_counts[label] for label in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights))

def train():
    # Device
    device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
    print(f"Using device: {device}")

    # Load and split data
    df = pd.read_csv(CSV_PATH)
    df = df.drop(columns=['UNK'], errors='ignore')
    train_df, val_df = train_test_split(df, test_size=0.2, 
                                         stratify=df[CLASSES].values.argmax(axis=1),
                                         random_state=42)

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    # Transforms
    train_transform, val_transform = get_transforms()

    # Datasets
    train_dataset = SkinLesionDataset(train_df, IMG_DIR, transform=train_transform)
    val_dataset = SkinLesionDataset(val_df, IMG_DIR, transform=val_transform)

    # Sampler and DataLoaders
    sampler = get_sampler(train_df)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Model
    model = get_model(num_classes=8, device=device)

    # Loss with class weights
    class_weights = get_class_weights(train_df).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer - only train the final layer
    # Train both layer4 and final FC layer
    optimizer = torch.optim.Adam([
        {'params': model.layer4.parameters(), 'lr': 0.0001},
        {'params': model.fc.parameters(), 'lr': 0.001}
])

    # Scheduler - reduces LR when progress stalls
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                            patience=2, 
                                                            factor=0.5)

    # Training loop
    epochs = 5
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Accuracy: {accuracy:.2f}%")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "../data/best_model.pth")
            print(f"  --> Best model saved")

        scheduler.step(avg_val_loss)

if __name__ == "__main__":
    train()