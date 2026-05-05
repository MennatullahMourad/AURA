@echo off
echo ============================================
echo   AURA Project Setup
echo ============================================

cd /d C:\Users\USER\Desktop\AURA

echo.
echo [1/4] Creating virtual environment...
python -m venv aura

echo.
echo [2/4] Activating environment...
call aura\Scripts\activate

echo.
echo [3/4] Installing requirements...
pip install flask werkzeug
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pillow numpy pandas scikit-learn matplotlib seaborn
pip install xgboost lightgbm optuna imbalanced-learn
pip install pytorch-tabular
pip install mediapipe opencv-python
pip install joblib

echo.
echo [4/4] Creating folder structure...
mkdir Results\behavioral_model 2>nul
mkdir Results\image_model 2>nul
mkdir Results\eyetracking_model 2>nul
mkdir Results\eyetracking_model\data 2>nul
mkdir Results\fusion 2>nul
mkdir web\tmp 2>nul
mkdir web\templates 2>nul

echo.
echo ============================================
echo   Setup Complete!
echo   Now run: python web\app_final.py
echo ============================================
pause
