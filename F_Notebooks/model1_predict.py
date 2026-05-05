
import os, pickle, json, warnings
import numpy as np
warnings.filterwarnings('ignore')

MODEL_DIR = r'C:\Users\USER\Desktop\AURA_PROJECT\Results\behavioral_model'

_ensemble     = None
_preprocessor = None

def _load():
    global _ensemble, _preprocessor
    if _ensemble is None:
        gb_path  = os.path.join(MODEL_DIR, 'xgb_model.pkl')
        pre_path = os.path.join(MODEL_DIR, 'preprocessor.pkl')
        if os.path.exists(gb_path):
            with open(gb_path, 'rb') as f:
                _ensemble = pickle.load(f)
            with open(pre_path, 'rb') as f:
                _preprocessor = pickle.load(f)
            print('✅ XGBoost Ensemble loaded')
        else:
            _ensemble = 'fallback'
            print('⚠️ Model not found — using rule-based fallback')

AQ_COLS   = [f'A{i}' for i in range(1,11)]
META_COLS = ['age','gender','jaundice','family_history']
FEATURES  = AQ_COLS + META_COLS

def _rule_based(answers, age, jaundice, family_history):
    score = sum(answers)
    if   score <= 2: prob = score/10 * 0.4
    elif score <= 4: prob = 0.08 + (score-2)*0.10
    elif score <= 6: prob = 0.28 + (score-4)*0.12
    elif score <= 8: prob = 0.52 + (score-6)*0.14
    else:            prob = 0.80 + (score-8)*0.08
    age_f = 1.05 if age<3 else 1.0 if age<6 else 0.95 if age<12 else 0.9
    prob *= age_f
    if str(jaundice).lower()=='yes':       prob += 0.06
    if str(family_history).lower()=='yes': prob += 0.10
    return float(np.clip(prob, 0.03, 0.97))

def predict_questionnaire(answers, age, gender, jaundice, family_history):
    _load()
    answers = [int(a) for a in answers]
    q_score = sum(answers)
    gd = 1 if str(gender).lower() in ('m','male') else 0
    jd = 1 if str(jaundice).lower()=='yes' else 0
    fh = 1 if str(family_history).lower()=='yes' else 0

    if _ensemble != 'fallback':
        try:
            import pandas as pd
            X_raw = np.array(answers + [float(age), gd, jd, fh]).reshape(1,-1)
            X_df  = pd.DataFrame(X_raw, columns=FEATURES)
            X_pre = _preprocessor.transform(X_df.values)
            gb_prob = float(_ensemble.predict_proba(X_pre)[0][1]) * 100
        except Exception as e:
            print(f'⚠️ Predict error: {e}')
            gb_prob = _rule_based(answers, age, jaundice, family_history) * 100
    else:
        gb_prob = _rule_based(answers, age, jaundice, family_history) * 100

    final_prob = round(gb_prob, 2)
    if   final_prob >= 75: risk = 'Very High'
    elif final_prob >= 55: risk = 'High'
    elif final_prob >= 30: risk = 'Moderate'
    else:                  risk = 'Low'

    return {
        'final_prob': final_prob,
        'gb_prob'   : round(gb_prob, 2),
        'tabt_prob' : None,
        'risk_level': risk,
        'q_score'   : q_score,
    }

if __name__ == '__main__':
    print('='*50)
    tests = [
        ([0,0,0,0,0,0,1,0,0,0], 5, 'm', 'no',  'no'),
        ([1,0,1,0,1,0,1,0,1,0], 4, 'f', 'no',  'yes'),
        ([1,1,1,1,1,1,1,1,1,1], 3, 'm', 'yes', 'yes'),
    ]
    for a,age,g,j,fh in tests:
        r = predict_questionnaire(a,age,g,j,fh)
        print(f"Score:{r['q_score']}/10 | Prob:{r['final_prob']}% | Risk:{r['risk_level']}")
