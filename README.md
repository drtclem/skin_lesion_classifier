# PyTorch Skin Lesion Classifier with Transfer Learning

A deep learning classifier for dermoscopic skin lesion images built with PyTorch and transfer learning on the ISIC 2019 dataset. Deployed as a live web application on Hugging Face Spaces.

> ⚠️ **Disclaimer:** This is a research and learning project, not a medical diagnostic tool. Do not use for clinical decision making.

---

## 🔬 Live Demo

**Try it here:** [huggingface.co/spaces/drtclem/Skin_Exam](https://huggingface.co/spaces/drtclem/Skin_Exam)

Upload a dermoscopy image and receive a classification prediction across 8 diagnostic categories with confidence scores and clinical context.

---

## Overview

This project trains a convolutional neural network to classify dermoscopy images into 8 diagnostic categories using transfer learning with a pretrained ResNet50 backbone. The goal was to explore the challenges of medical image classification, particularly class imbalance and the tradeoff between precision and recall in high-stakes prediction tasks.

---

## Dataset

**ISIC 2019 Challenge Dataset**
- 25,331 dermoscopy images across 8 diagnostic categories
- Source: [ISIC Archive](https://challenge.isic-archive.com/data/#2019)
- License: CC-BY-NC

### Diagnostic Categories

| Code | Diagnosis | Type |
|------|-----------|------|
| MEL | Melanoma | Malignant |
| NV | Melanocytic Nevus | Benign |
| BCC | Basal Cell Carcinoma | Malignant |
| AK | Actinic Keratosis | Pre-cancerous |
| BKL | Benign Keratosis | Benign |
| DF | Dermatofibroma | Benign |
| VASC | Vascular Lesion | Benign |
| SCC | Squamous Cell Carcinoma | Malignant |

### Class Distribution

![Class Distribution](images/class_distribution.png)

The dataset has severe class imbalance with NV (12,875 images) outnumbering DF (239 images) by a 54:1 ratio. This was a central challenge of the project.

---

## Sample Images by Class

![Sample Images](images/sample_images.png)

Visual inspection reveals why some classes are harder than others. MEL and NV share similar dark irregular morphology while VASC has a distinctive red/purple appearance that makes it visually separable.

---

## Approach

### Model Architecture
- **Backbone:** ResNet50 pretrained on ImageNet
- **Transfer learning strategy:** Froze all layers except `layer4` and the final FC layer
- **Custom head:** `Linear(2048→256) → ReLU → Dropout(0.3) → Linear(256→8)`
- **Rationale:** Medical imagery differs significantly from ImageNet. Unfreezing `layer4` allowed the model to adapt mid-level features to dermoscopic patterns

### Handling Class Imbalance
Three strategies were combined:
1. **WeightedRandomSampler** -- oversampled minority classes during training
2. **Weighted CrossEntropyLoss** -- penalized mistakes on rare classes more heavily
3. **Data augmentation** -- random flips, rotations, and color jitter on training images

### Training
- Optimizer: Adam with layer-specific learning rates (`layer4: 1e-4`, `fc: 1e-3`)
- Scheduler: ReduceLROnPlateau (patience=2, factor=0.5)
- Epochs: 25
- Batch size: 32
- Hardware: Apple Silicon MPS

---

## Results

### Overall Performance

| Training Run | Epochs | Val Accuracy |
|---|---|---|
| Initial run | 5 | 47.98% |
| Extended run | 25 | 66.02% (best saved model) |
| Peak | 25 | 70.48% (epoch 23) |

Baseline random guessing: 12.5%

### Per Class Performance (25 epochs)

| Class | Precision | Recall | F1 | vs 5 epochs |
|-------|-----------|--------|----|-------------|
| MEL | 0.46 | 0.75 | 0.57 | +0.10 |
| NV | 0.95 | 0.55 | 0.69 | +0.16 |
| BCC | 0.75 | 0.83 | 0.79 | +0.22 |
| AK | 0.52 | 0.74 | 0.61 | +0.21 |
| BKL | 0.48 | 0.77 | 0.59 | +0.17 |
| DF | 0.53 | 0.75 | 0.62 | +0.34 |
| VASC | 0.66 | 1.00 | 0.80 | +0.30 |
| SCC | 0.63 | 0.71 | 0.67 | +0.31 |

Every class improved with extended training. The macro avg F1 improved from 0.44 to 0.67.

### Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

### Key Findings

**Strengths:**
- VASC achieved 100% recall due to its visually distinctive appearance
- NV precision of 95% means very few false melanoma alarms on moles
- Every class improved meaningfully from 5 to 25 epochs
- Model learned meaningful class separation well above random baseline

**Limitations:**
- MEL recall of 75% means 25% of melanomas were missed, not acceptable for clinical use
- High confidence predictions were not always correct, indicating room for better calibration
- NV recall of 55% reflects the difficulty of the dominant class under imbalanced sampling
- Model is not robust to out-of-distribution inputs such as non-dermoscopy images

**Clinical tradeoff:**
In cancer screening, recall matters more than precision. A missed melanoma is far more dangerous than a false alarm. The current MEL recall of 75% would need to reach 90%+ before this model could be considered for any assistive clinical role.

---

## Live Application

The model is deployed as a Gradio app on Hugging Face Spaces at [huggingface.co/spaces/drtclem/Skin_Exam](https://huggingface.co/spaces/drtclem/Skin_Exam).

### Application Features
- Upload any dermoscopy image for classification
- Color coded severity indicators (🔴 malignant, 🟡 pre-cancerous, 🟢 benign)
- Full probability distribution across all 8 classes
- Low confidence warning triggered below 80% confidence threshold
- Persistent medical disclaimer on every prediction
- Out-of-distribution detection flags non-dermoscopy inputs

---

## Project Structure


\```
skin_lesion_classifier/
├── data/
│   ├── raw/
│   ├── best_model.pth
│   └── class_distribution.png
├── images/
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── demo.py
└── requirements.txt
\```


---

## Usage

### Setup
```bash
git clone https://github.com/drtclem/skin_lesion_classifier.git
cd skin_lesion_classifier
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Train
```bash
cd src
python train.py
```

### Evaluate
```bash
cd src
python evaluate.py
```

### Demo
```bash
python demo.py path/to/image.jpg
```

---

## Future Improvements

- **Larger backbone:** ResNet101 or EfficientNet for higher capacity
- **Confidence calibration:** Temperature scaling to make confidence scores more reliable
- **Out-of-distribution detection:** Dedicated OOD classifier to reject non-dermoscopy inputs
- **Ensemble modeling:** Combine multiple model checkpoints to improve stability
- **Longer training:** Cosine annealing scheduler with warm restarts

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![ResNet50](https://img.shields.io/badge/Model-ResNet50-green)
![HuggingFace](https://img.shields.io/badge/Deploy-HuggingFace-yellow)

---

*Taylor Clements, PhD*