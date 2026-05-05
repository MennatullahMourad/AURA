"""
AURA - Model 1: XGBoost Stacking Ensemble
==========================================
Behavioral Questionnaire Risk Classifier

Architecture:
    - Feature Engineering (42 features from 14 raw inputs)
    - SMOTE for class balancing
    - Optuna hyperparameter tuning
    - Stacking: XGBoost + LightGBM + GradientBoosting -> LogisticRegression

Input  : A1-A10 answers + age + gender + jaundice + family_history
Output : ASD probability (0-1) + risk level
"""

import numpy as np
import pandas as pd
import pickle, os, json, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = r'C:\Users\USER\Desktop\AURA_PROJECT\behavioral data\AURA_Data_B.csv'
SAVE_DIR  = r'C:\Users\USER\Desktop\AURA_PROJECT\Results\behavioral_model'
os.makedirs(SAVE_DIR, exist_ok=True)

TARGET    = 'Final_Class'
AQ_COLS   = [f'A{i}' for i in range(1, 11)]
META_COLS = ['age', 'gender', 'jaundice', 'family_history']
FEATURES  = AQ_COLS + META_COLS

# ============================================================
# DATA LOADING
# ============================================================

print('=' * 55)
print('  AURA - Model 1: XGBoost Ensemble Training')
print('=' * 55)

df = pd.read_csv(DATA_PATH)
print(f'\nData loaded: {df.shape}')
print(f'Columns: {df.columns.tolist()}')

missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    raise ValueError(f'Missing required columns: {missing}')

print(f'\nTarget distribution:\n{df[TARGET].value_counts()}')

df = df[FEATURES + [TARGET]].copy()
df[AQ_COLS] = df[AQ_COLS].fillna(0)
df['age']   = df['age'].fillna(df['age'].median())
df.dropna(inplace=True)
df[TARGET]  = df[TARGET].astype(int)
print(f'After cleaning: {df.shape}')

# ============================================================
# FEATURE ENGINEERING
# Expands 14 raw features into 42 engineered features:
#   - AQ score aggregates (total, ratio, polynomial)
#   - Social vs Repetitive behavior subscores
#   - Age group flags
#   - High-risk thresholds
#   - Interaction terms (jaundice*AQ, family*social)
#   - Pairwise question interactions
# ============================================================

class AURAPreprocessor(BaseEstimator, TransformerMixin):
    """
    Custom sklearn-compatible preprocessor.
    Fits a StandardScaler on engineered features.
    Must be saved alongside the model for inference.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names_out_ = []

    def fit(self, X, y=None):
        Xe = self._engineer(pd.DataFrame(X, columns=FEATURES))
        self.feature_names_out_ = Xe.columns.tolist()
        self.scaler.fit(Xe)
        return self

    def transform(self, X, y=None):
        Xe = self._engineer(pd.DataFrame(X, columns=FEATURES))
        return self.scaler.transform(Xe)

    def _engineer(self, df):
        out = df.copy()
        aq = [c for c in AQ_COLS if c in df.columns]

        # AQ score aggregates
        out['AQ_Total']  = df[aq].sum(axis=1)
        out['AQ_Ratio']  = out['AQ_Total'] / len(aq)
        out['AQ_Sq']     = out['AQ_Total'] ** 2
        out['AQ_Cube']   = out['AQ_Total'] ** 3

        # Social and repetitive behavior subscores
        # Social items: joint attention, eye contact, pointing, social interest
        # Repetitive items: routine adherence, repetitive play, sensory behaviors
        social = [c for c in ['A1','A2','A4','A5','A7'] if c in df.columns]
        repet  = [c for c in ['A3','A6','A8','A9','A10'] if c in df.columns]
        out['Social_Score']       = df[social].sum(axis=1)
        out['Repetitive_Score']   = df[repet].sum(axis=1)
        out['Social_Repet_Ratio'] = out['Social_Score'] / (out['Repetitive_Score'] + 1)
        out['Score_Diff']         = out['Social_Score'] - out['Repetitive_Score']

        # Age group flags for developmental stage segmentation
        if 'age' in df.columns:
            out['Age_Log']    = np.log1p(df['age'])
            out['Age_Sqrt']   = np.sqrt(df['age'].clip(0))
            out['Age_Child']  = (df['age'] < 12).astype(int)
            out['Age_Teen']   = ((df['age'] >= 12) & (df['age'] < 18)).astype(int)
            out['Age_Adult']  = ((df['age'] >= 18) & (df['age'] < 40)).astype(int)
            out['Age_Senior'] = (df['age'] >= 40).astype(int)

        # Clinical risk thresholds based on AQ scoring guidelines
        out['High_Risk_Flag'] = (out['AQ_Total'] >= 6).astype(int)
        out['Very_High_Risk'] = (out['AQ_Total'] >= 8).astype(int)

        # Risk factor interaction terms
        if 'jaundice' in df.columns:
            out['Jaundice_AQ'] = df['jaundice'] * out['AQ_Total']
        if 'family_history' in df.columns:
            out['Family_AQ']     = df['family_history'] * out['AQ_Total']
            out['Family_Social'] = df['family_history'] * out['Social_Score']

        # Adjacent question pairwise interactions
        pairs = [('A1','A2'),('A1','A3'),('A2','A4'),('A3','A5'),
                 ('A4','A6'),('A5','A7'),('A6','A8'),('A7','A9'),('A8','A10')]
        for a, b in pairs:
            if a in df.columns and b in df.columns:
                out[f'{a}x{b}'] = df[a] * df[b]
        return out


# ============================================================
# PREPROCESSING + SMOTE
# SMOTE generates synthetic minority class samples to address
# class imbalance (ASD: 2834 vs Non-ASD: 5387)
# ============================================================

print('\nRunning feature engineering...')
X_raw = df[FEATURES].values
y     = df[TARGET].values

preprocessor = AURAPreprocessor()
X_trans      = preprocessor.fit_transform(X_raw)
feat_names   = preprocessor.feature_names_out_
print(f'Features after engineering: {len(feat_names)}')

print('Applying SMOTE for class balancing...')
smote        = SMOTE(random_state=42, k_neighbors=5)
X_bal, y_bal = smote.fit_resample(X_trans, y)
print(f'After SMOTE: {X_bal.shape}')

X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal)
print(f'Train: {X_train.shape} | Test: {X_test.shape}')

# ============================================================
# HYPERPARAMETER TUNING WITH OPTUNA
# Optimizes XGBoost hyperparameters using 5-fold CV
# Best params are then used in the final stacking ensemble
# ============================================================

print('\nTuning XGBoost hyperparameters (100 trials)...')

def objective_xgb(trial):
    p = {
        'n_estimators'     : trial.suggest_int('n_estimators', 200, 500),
        'max_depth'        : trial.suggest_int('max_depth', 4, 8),
        'learning_rate'    : trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample'        : trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'eval_metric'      : 'logloss',
        'random_state'     : 42,
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    return cross_val_score(
        xgb.XGBClassifier(**p), X_bal, y_bal,
        cv=cv, scoring='accuracy', n_jobs=-1
    ).mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective_xgb, n_trials=100, show_progress_bar=True)
BEST_XGB = {**study.best_params, 'eval_metric': 'logloss', 'random_state': 42}
print(f'Best XGBoost CV Accuracy: {study.best_value * 100:.2f}%')
print(f'Best params: {study.best_params}')

# ============================================================
# STACKING ENSEMBLE TRAINING
# Level 0 estimators: XGBoost + LightGBM + GradientBoosting
# Level 1 meta-learner: Logistic Regression
# 5-fold cross-validation for out-of-fold predictions
# ============================================================

print('\nTraining stacking ensemble...')

xgb_clf = xgb.XGBClassifier(**BEST_XGB)

lgb_clf = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=63,
    random_state=42,
    verbose=-1)

gb_clf = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    random_state=42)

ensemble = StackingClassifier(
    estimators=[('xgb', xgb_clf), ('lgb', lgb_clf), ('gb', gb_clf)],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    n_jobs=-1
)
ensemble.fit(X_train, y_train)
print('Ensemble training complete.')

# ============================================================
# EVALUATION
# ============================================================

y_pred  = ensemble.predict(X_test)
y_proba = ensemble.predict_proba(X_test)[:, 1]
acc     = accuracy_score(y_test, y_pred)
auc     = roc_auc_score(y_test, y_proba)

print(f'\n{"=" * 55}')
print(f'  Test Accuracy : {acc * 100:.2f}%')
print(f'  AUC Score     : {auc:.4f}')
print(f'{"=" * 55}')
print(classification_report(y_test, y_pred, target_names=['No ASD', 'ASD']))

# ============================================================
# SAVE ARTIFACTS
# xgb_model.pkl    - the trained stacking ensemble
# preprocessor.pkl - the feature engineering + scaler pipeline
#                    (required at inference time)
# feature_names.pkl - list of engineered feature names
# model_info.json   - accuracy metrics and best hyperparameters
# ============================================================


# ============================================================
# SHAP ANALYSIS
# Explains which features drive the model's predictions.
# Saved as feature_importance_shap.png in Results folder.
# ============================================================

print('\nRunning SHAP analysis...')
try:
    import shap
    import matplotlib.pyplot as plt

    # Use XGBoost base model for SHAP (TreeExplainer only works with tree models)
    xgb_base = xgb.XGBClassifier(**BEST_XGB)
    xgb_base.fit(X_train, y_train)

    explainer  = shap.TreeExplainer(xgb_base)
    shap_vals  = explainer.shap_values(X_test[:200])

    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_vals, X_test[:200],
        feature_names=feat_names,
        plot_type='bar',
        show=False,
        max_display=20
    )
    plt.title('AURA Model 1 - Top 20 Features (SHAP)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    shap_path = os.path.join(SAVE_DIR, 'feature_importance_shap.png')
    plt.savefig(shap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  SHAP chart saved -> {shap_path}')

    # Also save SHAP values as numpy for use in web app
    import pickle
    with open(os.path.join(SAVE_DIR, 'shap_explainer.pkl'), 'wb') as f:
        pickle.dump(explainer, f)
    print(f'  SHAP explainer saved -> {SAVE_DIR}')

except ImportError:
    print('  SHAP not installed. Run: pip install shap')
except Exception as e:
    print(f'  SHAP error: {e}')

print('\nSaving model artifacts...')

with open(os.path.join(SAVE_DIR, 'xgb_model.pkl'), 'wb') as f:
    pickle.dump(ensemble, f)

with open(os.path.join(SAVE_DIR, 'preprocessor.pkl'), 'wb') as f:
    pickle.dump(preprocessor, f)

with open(os.path.join(SAVE_DIR, 'feature_names.pkl'), 'wb') as f:
    pickle.dump(feat_names, f)

info = {
    'accuracy'    : round(acc * 100, 2),
    'auc'         : round(auc, 4),
    'n_features'  : len(feat_names),
    'best_params' : study.best_params
}
with open(os.path.join(SAVE_DIR, 'model_info.json'), 'w') as f:
    json.dump(info, f, indent=2)

print(f'  xgb_model.pkl     -> {SAVE_DIR}')
print(f'  preprocessor.pkl  -> {SAVE_DIR}')
print(f'  feature_names.pkl -> {SAVE_DIR}')
print(f'  model_info.json   -> {SAVE_DIR}')
print(f'\nTraining complete. Accuracy = {acc * 100:.2f}%')
