"""
AURA - Model 3: BiLSTM + Transformer (Upgraded)
================================================
Eye Tracking Time-Series Classifier

Changes from original:
    - BiLSTM added before Transformer (bidirectional temporal modeling)
    - hidden_dim: 64 -> 128
    - num_layers: 2 -> 4
    - num_heads:  2 -> 4
    - norm_first=True (Pre-LN for stable training)
    - AdamW optimizer with weight decay
    - CosineAnnealingLR scheduler
    - Gradient clipping
    - Early stopping based on F1 score
    - Best model checkpoint saved automatically
"""

import pandas as pd
import numpy as np
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIGURATION
# ============================================================

CSV_FOLDER_PATH    = r'C:\Users\USER\Desktop\AURA_PROJECT\Eye-Tracking Dataset\Eye-tracking Output'
METADATA_FILE_PATH = r'C:\Users\USER\Desktop\AURA_PROJECT\Eye-Tracking Dataset\Metadata_Participants.csv'
RESULTS_DIR        = r'C:\Users\USER\Desktop\AURA_PROJECT\Results\eyetracking_model'
DATA_DIR           = r'C:\Users\USER\Desktop\AURA_PROJECT\Results\eyetracking_model\data'

WINDOW_SIZE = 50
STEP_SIZE   = 10
EPOCHS      = 120
PATIENCE    = 12

CATEGORY_MAP = {'Fixation': 1, 'Saccade': 2, 'Blink': 3, 'Separator': 0, '-': 0}
AOI_MAP = {
    'corps'          : 1,
    'BallonVisible'  : 2,
    'BallonInvisible': 2,
    'Pointage D'     : 2,
    'Pointage G'     : 2,
    '-'              : 0
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# PREPROCESSING FUNCTIONS
# ============================================================

def to_float(val):
    try:
        return float(str(val).replace('-', '').strip() or '0')
    except:
        return 0.0

def to_float_signed(val):
    try:
        v = str(val).strip()
        if v in ('', '-', 'nan', 'NaN'):
            return 0.0
        return float(v)
    except:
        return 0.0

def clean_row(row, has_pupil_right, has_pupil_left, has_aoi):
    category_right = CATEGORY_MAP.get(str(row['Category Right']), 0)
    pupil_right    = to_float(row['Pupil Diameter Right [mm]'])            if has_pupil_right else 0.0
    gaze_x_right   = to_float_signed(row['Point of Regard Right X [px]'])
    gaze_y_right   = to_float_signed(row['Point of Regard Right Y [px]'])
    gaze_z_right   = to_float_signed(row['Gaze Vector Right Z'])           if 'Gaze Vector Right Z'         in row.index else 0.0
    aoi            = AOI_MAP.get(str(row['AOI Name Right']), 0)            if has_aoi else 0
    tracking       = to_float(row['Tracking Ratio [%]'])
    pupil_left     = to_float(row['Pupil Diameter Left [mm]'])             if has_pupil_left else 0.0
    gaze_x_left    = to_float_signed(row['Point of Regard Left X [px]'])   if 'Point of Regard Left X [px]' in row.index else 0.0
    gaze_y_left    = to_float_signed(row['Point of Regard Left Y [px]'])   if 'Point of Regard Left Y [px]' in row.index else 0.0
    gaze_z_left    = to_float_signed(row['Gaze Vector Left Z'])            if 'Gaze Vector Left Z'          in row.index else 0.0
    return [
        category_right, pupil_right, gaze_x_right, gaze_y_right, gaze_z_right,
        aoi, tracking, pupil_left, gaze_x_left, gaze_y_left, gaze_z_left
    ]

def prepare_sequences():
    participants_metadata = pd.read_csv(METADATA_FILE_PATH)
    participants_metadata['ParticipantID'] = participants_metadata['ParticipantID'].astype(str)
    participants_metadata['Label'] = participants_metadata['Class'].map({'ASD': 1, 'TD': 0})
    all_sequences, all_labels = [], []
    eye_tracking_files = sorted(glob.glob(os.path.join(CSV_FOLDER_PATH, "[0-9]*.csv")))
    print(f"[INFO] Found {len(eye_tracking_files)} CSV files")
    for file_path in eye_tracking_files:
        print(f"  Processing: {os.path.basename(file_path)}")
        eye_tracking_data = pd.read_csv(file_path, low_memory=False)
        eye_tracking_data['Participant'] = eye_tracking_data['Participant'].astype(str).str.strip()
        eye_tracking_data = eye_tracking_data[
            ~eye_tracking_data['Participant'].str.contains('Unidentified', na=False)
        ]
        has_pupil_right = 'Pupil Diameter Right [mm]' in eye_tracking_data.columns
        has_pupil_left  = 'Pupil Diameter Left [mm]'  in eye_tracking_data.columns
        has_aoi         = 'AOI Name Right'             in eye_tracking_data.columns
        for participant_id, participant_rows in eye_tracking_data.groupby('Participant'):
            participant_meta = participants_metadata[
                participants_metadata['ParticipantID'] == participant_id
            ]
            if len(participant_meta) == 0:
                continue
            label = participant_meta['Label'].values[0]
            eye_movement_rows = participant_rows[
                participant_rows['Category Right'].isin(['Fixation', 'Saccade', 'Blink'])
            ].reset_index(drop=True)
            if len(eye_movement_rows) < WINDOW_SIZE:
                continue
            numeric_array = np.nan_to_num(
                np.array([
                    clean_row(row, has_pupil_right, has_pupil_left, has_aoi)
                    for _, row in eye_movement_rows.iterrows()
                ], dtype=float), nan=0.0
            )
            for start_idx in range(0, len(numeric_array) - WINDOW_SIZE + 1, STEP_SIZE):
                all_sequences.append(numeric_array[start_idx : start_idx + WINDOW_SIZE])
                all_labels.append(label)
    sequences_array = np.array(all_sequences)
    labels_array    = np.array(all_labels)
    print(f"\n[INFO] Shape: {sequences_array.shape}")
    print(f"[INFO] ASD: {(labels_array==1).sum():,} | TD: {(labels_array==0).sum():,}")
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.save(os.path.join(DATA_DIR, 'eyetracking_sequences.npy'), sequences_array)
    np.save(os.path.join(DATA_DIR, 'eyetracking_labels.npy'),    labels_array)
    return sequences_array, labels_array

# ============================================================
# DATA LOADING
# Delete old cached sequences and retrain from scratch each run
# to ensure results always reflect the latest data and code
# ============================================================

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

x_path = os.path.join(DATA_DIR, 'eyetracking_sequences.npy')
y_path = os.path.join(DATA_DIR, 'eyetracking_labels.npy')

# Delete old cached sequences to force fresh preparation
for old_file in [x_path, y_path]:
    if os.path.exists(old_file):
        os.remove(old_file)
        print(f"[INFO] Removed old cache: {os.path.basename(old_file)}")

# Delete old model weights
old_weights = os.path.join(RESULTS_DIR, 'eyetracking_model_weights.pth')
if os.path.exists(old_weights):
    os.remove(old_weights)
    print(f"[INFO] Removed old weights: eyetracking_model_weights.pth")

# Delete old results chart
old_chart = os.path.join(RESULTS_DIR, 'eyetracking_model_results.png')
if os.path.exists(old_chart):
    os.remove(old_chart)
    print(f"[INFO] Removed old chart: eyetracking_model_results.png")

print("[INFO] Preparing data from scratch...")
X, y = prepare_sequences()

# ============================================================
# NORMALIZATION AND SPLITTING
# ============================================================

N_SAMPLES, N_TIME, N_FEAT = X.shape

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_flat = X_train.reshape(-1, N_FEAT)
X_test_flat  = X_test.reshape(-1, N_FEAT)

scaler = MinMaxScaler()
scaler.fit(X_train_flat)

X_train = scaler.transform(X_train_flat).reshape(len(X_train), N_TIME, N_FEAT)
X_test  = scaler.transform(X_test_flat).reshape(len(X_test),   N_TIME, N_FEAT)

print(f"[INFO] X_train: {X_train.shape} | X_test: {X_test.shape}")

# ============================================================
# DATALOADERS
# ============================================================

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32).view(-1, 1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t),  batch_size=32)

# ============================================================
# MODEL ARCHITECTURE
# BiLSTM captures temporal dependencies in both directions.
# Transformer encoder captures long-range global dependencies.
# Pre-LayerNorm (norm_first=True) provides stable gradient flow.
# ============================================================

class AURA_EyeTransformer_V2(nn.Module):
    def __init__(self, input_dim=11, seq_len=50,
                 num_heads=4,
                 num_layers=4,
                 hidden_dim=128):
        super().__init__()

        # Bidirectional LSTM: hidden_size per direction = hidden_dim//2
        # Output dimension = hidden_dim (both directions concatenated)
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        self.lstm_norm = nn.LayerNorm(hidden_dim)

        # Learned positional embedding initialized with truncated normal
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # Transformer with Pre-LN for more stable optimization
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Binary classification head with GELU activations
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
# TRAINING SETUP
# ============================================================

model     = AURA_EyeTransformer_V2().to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-5)

print(f"[INFO] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# TRAINING LOOP
# ============================================================

history      = {'loss': [], 'train_acc': [], 'test_acc': [], 'f1': []}
best_f1      = 0.0
patience_cnt = 0
model_path   = os.path.join(RESULTS_DIR, 'eyetracking_model_weights.pth')

print("\n" + "=" * 60)
print("  AURA Model 3 - BiLSTM + Transformer")
print("=" * 60)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, correct_train = 0, 0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss    = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss    += loss.item()
        correct_train += ((outputs > 0.5).float() == batch_y).sum().item()

    model.eval()
    all_preds, all_labels_eval = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x.to(DEVICE))
            preds   = (outputs > 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels_eval.extend(batch_y.numpy())

    avg_loss  = total_loss / len(train_loader)
    train_acc = (correct_train / len(X_train_t)) * 100
    test_acc  = accuracy_score(all_labels_eval, all_preds) * 100
    f1        = f1_score(all_labels_eval, all_preds, zero_division=0)

    scheduler.step()

    history['loss'].append(avg_loss)
    history['train_acc'].append(train_acc)
    history['test_acc'].append(test_acc)
    history['f1'].append(f1)

    print(f"Epoch [{epoch:03d}/{EPOCHS}] Loss: {avg_loss:.4f} "
          f"Train: {train_acc:.2f}% Test: {test_acc:.2f}% F1: {f1:.4f}")

    # Save best model based on F1 score
    if f1 > best_f1:
        best_f1      = f1
        patience_cnt = 0
        os.makedirs(RESULTS_DIR, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"  Best model saved (F1={best_f1:.4f})")
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"\n[INFO] Early stopping at epoch {epoch}")
            break

# ============================================================
# FINAL EVALUATION ON BEST CHECKPOINT
# ============================================================

model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
model.eval()
all_preds, all_labels_eval = [], []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x.to(DEVICE))
        preds   = (outputs > 0.5).float().cpu().numpy()
        all_preds.extend(preds)
        all_labels_eval.extend(batch_y.numpy())

final_acc = accuracy_score(all_labels_eval, all_preds) * 100
final_f1  = f1_score(all_labels_eval, all_preds, zero_division=0)

print(f"\n{'=' * 60}")
print(f"  Best Accuracy : {final_acc:.2f}%")
print(f"  Best F1 Score : {final_f1:.4f}")
print(f"{'=' * 60}")
print(classification_report(all_labels_eval, all_preds, target_names=['TD', 'ASD']))

# ============================================================
# RESULTS VISUALIZATION
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Model 3 Results - Accuracy: {final_acc:.2f}% | F1: {final_f1:.4f}',
             fontsize=14, fontweight='bold')

axes[0, 0].plot(history['train_acc'], label='Train', color='steelblue')
axes[0, 0].plot(history['test_acc'],  label='Test',  color='orange')
axes[0, 0].set_title('Accuracy per Epoch')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy (%)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(history['loss'], color='crimson')
axes[0, 1].set_title('Training Loss per Epoch')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(history['f1'], color='green')
axes[1, 0].set_title('F1 Score per Epoch')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('F1 Score')
axes[1, 0].grid(True, alpha=0.3)

cm = confusion_matrix(all_labels_eval, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['TD', 'ASD'],
            yticklabels=['TD', 'ASD'],
            ax=axes[1, 1])
axes[1, 1].set_title('Confusion Matrix')
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'eyetracking_model_results.png'), dpi=150)
plt.show()

print(f"[INFO] Model saved to: {model_path}")
print(f"[INFO] Training complete. Best F1 = {final_f1:.4f}")
