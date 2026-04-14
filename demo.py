import torch
import sys
from PIL import Image
from torchvision import transforms
from src.model import get_model
from src.dataset import CLASSES

# Paths
MODEL_PATH = "data/best_model.pth"

# Class descriptions for readable output
CLASS_INFO = {
    'MEL':  'Melanoma - malignant, requires urgent attention',
    'NV':   'Melanocytic Nevus - common mole, usually benign',
    'BCC':  'Basal Cell Carcinoma - skin cancer, treatable',
    'AK':   'Actinic Keratosis - pre-cancerous lesion',
    'BKL':  'Benign Keratosis - benign skin growth',
    'DF':   'Dermatofibroma - benign skin tumor',
    'VASC': 'Vascular Lesion - blood vessel related',
    'SCC':  'Squamous Cell Carcinoma - skin cancer',
}

def predict(image_path):
    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Load model
    model = get_model(num_classes=8, device=device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Run prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_idx = probabilities.argmax().item()
        predicted_class = CLASSES[predicted_idx]
        confidence = probabilities[predicted_idx].item()

    # Print results
    print(f"\nImage: {image_path}")
    print(f"Prediction: {predicted_class}")
    print(f"Description: {CLASS_INFO[predicted_class]}")
    print(f"Confidence: {confidence*100:.1f}%")
    print(f"\nAll class probabilities:")
    
    # Sort by probability
    probs_sorted = sorted(zip(CLASSES, probabilities.tolist()), 
                         key=lambda x: x[1], reverse=True)
    for cls, prob in probs_sorted:
        bar = '█' * int(prob * 30)
        print(f"  {cls:4s}: {prob*100:5.1f}% {bar}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python demo.py <path_to_image>")
        print("Example: python demo.py data/raw/ISIC_2019_Training_Input/ISIC_0000002.jpg")
        sys.exit(1)
    
    predict(sys.argv[1])