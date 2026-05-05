"""
AURA - Model 2 Predict: EfficientNet-B4 + Transformer
======================================================
With Test Time Augmentation (TTA) for improved accuracy.

TTA applies 5 augmented versions of the same image and averages
the predictions, reducing uncertainty without any retraining.
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

MODEL_PATH    = r'C:\Users\USER\Desktop\AURA_PROJECT\Results\image_model\image_model_weights.pth'
CLASSES       = ['Low', 'Mild', 'Medium', 'High']
SEVERITY_PROB = {'Low': 0.1, 'Mild': 0.4, 'Medium': 0.7, 'High': 1.0}
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE    = 224

# Same architecture as model2_upgraded.py
class HybridAuraModel_V2(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        efficientnet = models.efficientnet_b4(weights=None)
        self.cnn = nn.Sequential(*list(efficientnet.children())[:-1])
        self.projection = nn.Sequential(
            nn.Linear(1792, 512), nn.LayerNorm(512), nn.GELU())
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=8, dim_feedforward=1024,
            dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.classifier  = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(512, 128),
            nn.GELU(), nn.Dropout(0.2), nn.Linear(128, num_classes))

    def forward(self, x):
        f = self.cnn(x).view(x.size(0), -1)
        f = self.projection(f).unsqueeze(1)
        o = self.transformer(f).squeeze(1)
        return self.classifier(o)

_model = None

def _load():
    global _model
    if _model is None:
        _model = HybridAuraModel_V2(num_classes=4).to(DEVICE)
        _model.load_state_dict(
            torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        _model.eval()
        print('Eye Image Model (EfficientNet-B4 + TTA) loaded')

# Standard transform for base prediction
_base_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# TTA transforms: 5 augmented views of the same image
# Each view gives a slightly different prediction; averaging improves accuracy
_tta_transforms = [
    # Original
    transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    # Horizontal flip
    transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    # Slight rotation +10 degrees
    transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 20, IMAGE_SIZE + 20)),
        transforms.RandomRotation((10, 10)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    # Slight rotation -10 degrees
    transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 20, IMAGE_SIZE + 20)),
        transforms.RandomRotation((-10, -10)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    # Center crop
    transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 30, IMAGE_SIZE + 30)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
]

def predict_eye_image(image_path: str, use_tta: bool = True) -> dict:
    """
    Predict ASD severity from an eye image.

    Args:
        image_path : path to eye image file
        use_tta    : use Test Time Augmentation (default True)

    Returns:
        severity      : Low / Mild / Medium / High
        severity_prob : ASD probability (0.0 to 1.0)
        confidence    : model confidence (%)
        class_probs   : probability per class (%)
        tta_used      : whether TTA was applied
    """
    _load()
    img = Image.open(image_path).convert('RGB')

    if use_tta:
        # Run all TTA transforms and average probabilities
        all_probs = []
        with torch.no_grad():
            for tfm in _tta_transforms:
                tensor = tfm(img).unsqueeze(0).to(DEVICE)
                logits = _model(tensor)
                probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
                all_probs.append(probs)
        probs = np.mean(all_probs, axis=0)
    else:
        tensor = _base_transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = _model(tensor)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx   = probs.argmax()
    severity   = CLASSES[pred_idx]
    confidence = float(probs[pred_idx]) * 100
    asd_prob   = SEVERITY_PROB[severity]

    return {
        'severity'     : severity,
        'severity_prob': asd_prob,
        'confidence'   : round(confidence, 2),
        'class_probs'  : {c: round(float(p) * 100, 2) for c, p in zip(CLASSES, probs)},
        'tta_used'     : use_tta,
    }

if __name__ == '__main__':
    print('=' * 50)
    print('  AURA - Model 2 Predict Test (with TTA)')
    print('=' * 50)
    test_dir = r'C:\Users\USER\Desktop\AURA_PROJECT\Eye-Tracking Dataset\Eye tracking (photos)'
    for cls in ['Low', 'Mild', 'Medium', 'High']:
        folder = os.path.join(test_dir, cls)
        if os.path.exists(folder):
            imgs = [f for f in os.listdir(folder)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if imgs:
                path   = os.path.join(folder, imgs[0])
                result = predict_eye_image(path)
                print(f'  Image     : {imgs[0]}')
                print(f'  Severity  : {result["severity"]}')
                print(f'  ASD Prob  : {result["severity_prob"] * 100:.0f}%')
                print(f'  Confidence: {result["confidence"]}%')
                print(f'  TTA Used  : {result["tta_used"]}')
                break
    else:
        print('No test images found.')
