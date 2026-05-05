import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from PIL import Image
import os

# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_DIR_MODEL2 = r'C:\Users\USER\Desktop\AURA_PROJECT\Results\image_model'
RESULTS_DIR_MODEL3 = r'C:\Users\USER\Desktop\AURA_PROJECT\Results\eyetracking_model'
RESULTS_DIR_FUSION = r'C:\Users\USER\Desktop\AURA_PROJECT\Results\fusion'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {DEVICE}")

# Severity label mapping for Model 2 output
SEVERITY_LABELS = {0: 'Low', 1: 'Mild', 2: 'Medium', 3: 'High'}

# Normalize severity level (0-3) to probability (0-1)
SEVERITY_TO_PROB = {0: 0.1, 1: 0.4, 2: 0.7, 3: 1.0}

# Fusion weights (must sum to 1.0)
WEIGHT_MODEL2 = 0.4   # CNN eye image model
WEIGHT_MODEL3 = 0.6   # Transformer CSV model

# ============================================================
# MODEL 2 ARCHITECTURE - EfficientNet-B4 + Transformer (Eye Images)
# [مُصلَح] متوافق مع model2_upgraded.py
# ============================================================

class HybridAuraModel(nn.Module):
    """
    Hybrid EfficientNet-B4 + Transformer for eye image severity classification.
    Output: 4 classes → Low (0), Mild (1), Medium (2), High (3)
    """
    def __init__(self, num_classes=4):
        super(HybridAuraModel, self).__init__()

        # EfficientNet-B4 backbone
        efficientnet = models.efficientnet_b4(weights=None)
        self.cnn = nn.Sequential(*list(efficientnet.children())[:-1])

        # EfficientNet-B4 outputs 1792 features
        self.d_model = 1792

        # Projection: 1792 → 512
        self.projection = nn.Sequential(
            nn.Linear(1792, 512),
            nn.LayerNorm(512),
            nn.GELU()
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.cnn(x)                              # [B, 1792, 1, 1]
        features = features.view(features.size(0), -1)     # [B, 1792]
        features = self.projection(features).unsqueeze(1)  # [B, 1, 512]
        t_out    = self.transformer(features)               # [B, 1, 512]
        return self.classifier(t_out.squeeze(1))            # [B, 4]

# ============================================================
# MODEL 3 ARCHITECTURE - BiLSTM + Transformer (CSV Eye Tracking)
# ============================================================

class AURA_EyeTransformer(nn.Module):
    """
    BiLSTM + Transformer classifier for eye-tracking time series.
    Output: ASD probability (0-1) via sigmoid
    """
    def __init__(self, input_dim=11, seq_len=50, num_heads=4, num_layers=4, hidden_dim=128):
        super(AURA_EyeTransformer, self).__init__()

        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        self.lstm_norm     = nn.LayerNorm(hidden_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        lstm_out    = self.lstm_norm(lstm_out)
        x = lstm_out + self.pos_embedding
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)

# ============================================================
# LOAD PRETRAINED MODELS
# ============================================================

def load_models():
    """Load both pretrained models from saved weights."""

    # Load Model 2 (EfficientNet-B4 + Transformer)
    model2 = HybridAuraModel(num_classes=4).to(DEVICE)
    model2_path = os.path.join(RESULTS_DIR_MODEL2, 'image_model_weights.pth')
    model2.load_state_dict(torch.load(model2_path, map_location=DEVICE, weights_only=True))
    model2.eval()
    print(f"[INFO] Model 2 (EfficientNet-B4) loaded from: {model2_path}")

    # Load Model 3 (BiLSTM + Transformer)
    model3 = AURA_EyeTransformer().to(DEVICE)
    model3_path = os.path.join(RESULTS_DIR_MODEL3, 'eyetracking_model_weights.pth')
    model3.load_state_dict(torch.load(model3_path, map_location=DEVICE, weights_only=False))
    model3.eval()
    print(f"[INFO] Model 3 (BiLSTM+Transformer) loaded from: {model3_path}")

    return model2, model3

# ============================================================
# IMAGE PREPROCESSING
# ============================================================

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_path):
    """Load and preprocess a single eye image for Model 2."""
    image = Image.open(image_path).convert('RGB')
    return image_transforms(image).unsqueeze(0).to(DEVICE)  # [1, 3, 224, 224]

# ============================================================
# SEQUENCE PREPROCESSING
# ============================================================

def preprocess_sequence(sequence_array):
    """
    Prepare a numpy sequence array for Model 3.
    Input : numpy array of shape (50, 11)
    Output: torch tensor of shape (1, 50, 11)
    """
    tensor = torch.tensor(sequence_array, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    return tensor

# ============================================================
# FUSION PREDICTION
# ============================================================

def predict_fusion(model2, model3, image_path, sequence_array):
    """
    Run both models and combine their outputs using weighted average.

    Args:
        model2       : Loaded EfficientNet-B4 model
        model3       : Loaded BiLSTM+Transformer model
        image_path   : Path to eye image file
        sequence_array: numpy array (50, 11) - eye tracking sequence

    Returns:
        dict with individual and final predictions
    """
    with torch.no_grad():

        # --- Model 2 Prediction (EfficientNet-B4 - Eye Image) ---
        image_tensor = preprocess_image(image_path)
        image_output = model2(image_tensor)
        severity_idx = torch.argmax(image_output, dim=1).item()
        severity_label = SEVERITY_LABELS[severity_idx]
        model2_prob = SEVERITY_TO_PROB[severity_idx]

        # --- Model 3 Prediction (BiLSTM+Transformer - CSV) ---
        seq_tensor   = preprocess_sequence(sequence_array)
        model3_prob  = model3(seq_tensor).item()   # Already a probability (0-1)

        # --- Fusion: Weighted Average ---
        final_risk = (WEIGHT_MODEL2 * model2_prob) + (WEIGHT_MODEL3 * model3_prob)
        final_risk_pct = round(final_risk * 100, 2)

        # --- Risk Level Label ---
        if final_risk_pct < 30:
            risk_level = "Low Risk"
        elif final_risk_pct < 55:
            risk_level = "Moderate Risk"
        elif final_risk_pct < 75:
            risk_level = "High Risk"
        else:
            risk_level = "Very High Risk"

    return {
        'model2_severity'   : severity_label,
        'model2_probability': round(model2_prob * 100, 2),
        'model3_probability': round(model3_prob * 100, 2),
        'final_risk_pct'    : final_risk_pct,
        'risk_level'        : risk_level
    }

# ============================================================
# ADD MODEL 1 (QUESTIONNAIRE) - When Ready
# ============================================================

def predict_fusion_with_questionnaire(model2, model3, image_path,
                                       sequence_array, model1_prob):
    """
    Full fusion including Model 1 (Behavioral Questionnaire).
    Call this when Model 1 (XGBoost) is ready from the behavioral team.

    Args:
        model1_prob: float (0-1) — probability from XGBoost questionnaire model
    """
    WEIGHT_MODEL1 = 0.4
    WEIGHT_MODEL2 = 0.2
    WEIGHT_MODEL3 = 0.4

    with torch.no_grad():
        image_tensor = preprocess_image(image_path)
        image_output = model2(image_tensor)
        severity_idx = torch.argmax(image_output, dim=1).item()
        model2_prob  = SEVERITY_TO_PROB[severity_idx]

        seq_tensor  = preprocess_sequence(sequence_array)
        model3_prob = model3(seq_tensor).item()

    final_risk = (WEIGHT_MODEL1 * model1_prob +
                  WEIGHT_MODEL2 * model2_prob +
                  WEIGHT_MODEL3 * model3_prob)

    final_risk_pct = round(final_risk * 100, 2)

    if final_risk_pct < 30:
        risk_level = "Low Risk"
    elif final_risk_pct < 55:
        risk_level = "Moderate Risk"
    elif final_risk_pct < 75:
        risk_level = "High Risk"
    else:
        risk_level = "Very High Risk"

    return {
        'model1_probability': round(model1_prob * 100, 2),
        'model2_severity'   : SEVERITY_LABELS[severity_idx],
        'model2_probability': round(model2_prob * 100, 2),
        'model3_probability': round(model3_prob * 100, 2),
        'final_risk_pct'    : final_risk_pct,
        'risk_level'        : risk_level
    }

# ============================================================
# TEST THE FUSION (Demo)
# ============================================================

if __name__ == "__main__":

    os.makedirs(RESULTS_DIR_FUSION, exist_ok=True)

    # Load both models
    model2, model3 = load_models()

    print("\n" + "="*60)
    print("  AURA Fusion Model - Ready")
    print("="*60)
    print(f"  Model 2 weight : {WEIGHT_MODEL2 * 100:.0f}%  (Eye Image - EfficientNet-B4)")
    print(f"  Model 3 weight : {WEIGHT_MODEL3 * 100:.0f}%  (Eye Tracking - BiLSTM+Transformer)")
    print("="*60)

    # --- Demo with dummy data ---
    print("\n[INFO] Running demo with dummy data...")

    dummy_sequence = np.random.rand(50, 11).astype(np.float32)

    # Find any image in model2 dataset to test
    test_image_path = None
    image_root = r'C:\Users\USER\Desktop\AURA_PROJECT\Eye-Tracking Dataset\Eye tracking (photos)'
    for cls in ['Low', 'Mild', 'Medium', 'High']:
        folder = os.path.join(image_root, cls)
        if os.path.exists(folder):
            imgs = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
            if imgs:
                test_image_path = os.path.join(folder, imgs[0])
                break

    if test_image_path:
        result = predict_fusion(model2, model3, test_image_path, dummy_sequence)

        print("\n" + "="*60)
        print("  FUSION RESULT")
        print("="*60)
        print(f"  Model 2 (Image)  → Severity  : {result['model2_severity']}")
        print(f"  Model 2 (Image)  → Prob      : {result['model2_probability']}%")
        print(f"  Model 3 (CSV)    → ASD Prob  : {result['model3_probability']}%")
        print(f"  ─────────────────────────────────")
        print(f"  Final Risk Score : {result['final_risk_pct']}%")
        print(f"  Risk Level       : {result['risk_level']}")
        print("="*60)
    else:
        print("[WARNING] No test image found. Please check DATA_PATH.")
