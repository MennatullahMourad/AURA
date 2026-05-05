# AURA — Autism Understanding & Risk Analyzer

Early detection of Autism Spectrum Disorder (ASD) using multi-modal AI: behavioral questionnaire analysis, eye image classification, and eye tracking time-series analysis.

---

## Project Structure

```
AURA_PROJECT/
├── behavioral data/
│   └── AURA_Data_B.csv
├── Eye-Tracking Dataset/
│   ├── Eye tracking (photos)/
│   │   ├── Low/
│   │   ├── Mild/
│   │   ├── Medium/
│   │   └── High/
│   ├── Eye-tracking Output/
│   └── Metadata_Participants.csv
├── F_Notebooks/
│   ├── AURA_Model1_train.py
│   ├── model2_upgraded.py
│   ├── model3_upgraded.py
│   ├── model1_predict.py
│   ├── model2_predict.py
│   ├── model3_predict.py
│   └── model4_fusion.py
├── Results/
│   ├── behavioral_model/
│   │   ├── xgb_model.pkl
│   │   └── preprocessor.pkl
│   ├── image_model/
│   │   └── image_model_weights.pth
│   └── eyetracking_model/
│       ├── eyetracking_model_weights.pth
│       └── scaler_final.pkl
├── Videos/
│   ├── social.mp4
│   └── nonsocial_trimmed.mp4
├── web/
│   ├── app_final.py
│   ├── database.py
│   └── templates/
├── requirements.txt
└── SETUP.bat
```

---

## Models

| Model | Architecture | Input | Accuracy |
|-------|-------------|-------|----------|
| Model 1 | XGBoost + LightGBM + GradientBoosting (Stacking) | Behavioral questionnaire (Q-CHAT-10 / AQ-10) | 94.71% |
| Model 2 | EfficientNet-B4 + Transformer Encoder | Eye images | 97%+ |
| Model 3 | BiLSTM + Transformer Encoder | Eye tracking CSV sequences | 89.92% |
| Fusion  | Weighted Average (0.4 / 0.2 / 0.4) | Models 1, 2, 3 | — |

---

## Setup

```bash
# Windows
SETUP.bat

# Or manually
python -m venv aura
aura\Scripts\activate
pip install -r requirements.txt
```

---

## Training

```bash
# Model 1 — Behavioral Questionnaire
python F_Notebooks\AURA_Model1_train.py

# Model 2 — Eye Images
python F_Notebooks\model2_upgraded.py

# Model 3 — Eye Tracking
python F_Notebooks\model3_upgraded.py
```

---

## Run Web Application

```bash
cd web
python app_final.py
```

Open: http://127.0.0.1:5000

---

## Tech Stack

- **Backend:** Python, Flask, SQLite
- **ML:** PyTorch, XGBoost, LightGBM, Scikit-learn, Optuna
- **Vision:** EfficientNet-B4, Transformer Encoder
- **Sequence:** BiLSTM, Transformer Encoder
- **Eye Tracking:** MediaPipe Face Mesh

---

## Team

| Name | ID |
|------|----|
| Mohamed Mohamed Mostafa Agena | 4241349 |
| Mohamed Abdelkhaleq Abdelfattah | 4241907 |
| Menna Essam Rashash | 42411022 |
| Ola Asad Anwar | 4241332 |
| Mennatullah Mourad Awad Ali | 4241344 |
| Shimaa Mansour Elshahat | 4241580 |

**Supervisor:** Dr. Reham AbdElbaset AbdElwahab

---

## Notes

- Dataset and trained model weights are not included in this repository.
- Place your data files in the paths shown in the project structure above.
- The `Results/` folder will be created automatically after training.
