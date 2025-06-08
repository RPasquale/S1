@echo off
echo 🌙 OVERNIGHT TRAINING STATUS
echo ========================
echo.
echo Current time: %date% %time%
echo.

REM Check GPU status
echo 🚀 GPU Status:
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader

echo.
echo 📊 Recent checkpoints:
dir /od *.pkl | findstr checkpoint

echo.
echo 💾 Training logs:
dir /od overnight_logs\*.log

echo.
echo ⚡ Quick Python status check:
python check_training_status.py

echo.
echo 📈 To check detailed progress, run: python check_training_status.py
pause
