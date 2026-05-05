"""
AURA — Model 3 Predict (BiLSTM + Transformer)
متوافق 100% مع model3_upgraded.py
"""
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

MODEL_PATH  = r'C:\Users\USER\Desktop\AURA_PROJECT\Results\eyetracking_model\eyetracking_model_weights.pth'
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WINDOW_SIZE = 50
N_FEAT      = 11

CATEGORY_MAP = {'Fixation':1,'Saccade':2,'Blink':3,'Separator':0,'-':0}
AOI_MAP = {'corps':1,'BallonVisible':2,'BallonInvisible':2,
           'Pointage D':2,'Pointage G':2,'-':0}

# ============================================================
# نفس الـ Architecture بالظبط من model3_upgraded.py
# ============================================================
class AURA_EyeTransformer_V2(nn.Module):
    def __init__(self, input_dim=11, seq_len=50,
                 num_heads=4, num_layers=4, hidden_dim=128):
        super().__init__()

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
# تحميل الموديل
# ============================================================
_model = None

def _load():
    global _model
    if _model is None:
        _model = AURA_EyeTransformer_V2().to(DEVICE)
        _model.load_state_dict(
            torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
        _model.eval()
        print('✅ Eye Tracking Model (BiLSTM+Transformer) loaded')

# ============================================================
# Helper functions
# ============================================================
def to_float(v):
    try: return float(str(v).replace('-','').strip() or '0')
    except: return 0.0

def to_float_s(v):
    try:
        s = str(v).strip()
        return 0.0 if s in ('','-','nan','NaN') else float(s)
    except: return 0.0

def _prepare_csv(csv_path: str) -> torch.Tensor:
    df = pd.read_csv(csv_path, low_memory=False)

    hpr  = 'Pupil Diameter Right [mm]' in df.columns
    hpl  = 'Pupil Diameter Left [mm]'  in df.columns
    haoi = 'AOI Name Right'            in df.columns

    if 'Category Right' in df.columns:
        df = df[df['Category Right'].isin(
            ['Fixation','Saccade','Blink'])].reset_index(drop=True)

    rows = []
    for _, row in df.iterrows():
        rows.append([
            CATEGORY_MAP.get(str(row.get('Category Right','')), 0),
            to_float(row.get('Pupil Diameter Right [mm]',0))            if hpr  else 0.0,
            to_float_s(row.get('Point of Regard Right X [px]',0)),
            to_float_s(row.get('Point of Regard Right Y [px]',0)),
            to_float_s(row.get('Gaze Vector Right Z',0)),
            AOI_MAP.get(str(row.get('AOI Name Right','-')),0)           if haoi else 0,
            to_float(row.get('Tracking Ratio [%]',0)),
            to_float(row.get('Pupil Diameter Left [mm]',0))             if hpl  else 0.0,
            to_float_s(row.get('Point of Regard Left X [px]',0)),
            to_float_s(row.get('Point of Regard Left Y [px]',0)),
            to_float_s(row.get('Gaze Vector Left Z',0)),
        ])

    data = np.nan_to_num(np.array(rows, dtype=float), nan=0.0)

    # Normalize
    mean = data.mean(axis=0)
    std  = data.std(axis=0) + 1e-8
    data = (data - mean) / std

    sequences = []
    for i in range(0, len(data) - WINDOW_SIZE + 1, WINDOW_SIZE):
        sequences.append(data[i:i+WINDOW_SIZE])

    if not sequences:
        padded = np.zeros((WINDOW_SIZE, N_FEAT))
        padded[:min(len(data), WINDOW_SIZE)] = data[:WINDOW_SIZE]
        sequences = [padded]

    return torch.FloatTensor(np.array(sequences)).to(DEVICE)

# ============================================================
# دالة التنبؤ
# ============================================================
def predict_eye_tracking(csv_path: str) -> dict:
    _load()
    sequences = _prepare_csv(csv_path)
    with torch.no_grad():
        probs = _model(sequences).cpu().numpy().flatten()
    asd_prob = float(probs.mean()) * 100
    if   asd_prob < 30: risk = 'Low'
    elif asd_prob < 55: risk = 'Moderate'
    elif asd_prob < 75: risk = 'High'
    else:               risk = 'Very High'
    return {
        'asd_prob'     : round(asd_prob, 2),
        'num_sequences': len(sequences),
        'risk_level'   : risk,
    }

# ============================================================
# اختبار
# ============================================================
if __name__ == '__main__':
    print('='*50)
    print('  AURA — Model 3 Predict Test')
    print('='*50)
    csv_dir = r'C:\Users\USER\Desktop\AURA_PROJECT\Eye-Tracking Dataset\Eye-tracking Output'
    if os.path.isdir(csv_dir):
        csvs = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        if csvs:
            result = predict_eye_tracking(os.path.join(csv_dir, csvs[0]))
            print(f'\n  ASD Prob   : {result["asd_prob"]}%')
            print(f'  Sequences  : {result["num_sequences"]}')
            print(f'  Risk Level : {result["risk_level"]}')