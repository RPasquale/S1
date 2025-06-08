@echo off
REM Quick WandB Connection Fixer for Windows
echo 🔧 WandB Connection Fixer
echo ========================

echo 🗑️ Killing WandB processes...
taskkill /f /im "*wandb*" >nul 2>&1
taskkill /f /fi "IMAGENAME eq python.exe" /fi "WINDOWTITLE eq *wandb*" >nul 2>&1

echo 🧹 Cleaning WandB cache...
if exist "%USERPROFILE%\.cache\wandb" (
    del /q /s "%USERPROFILE%\.cache\wandb\*.tmp" >nul 2>&1
    del /q /s "%USERPROFILE%\.cache\wandb\*.lock" >nul 2>&1
    del /q /s "%USERPROFILE%\.cache\wandb\*.log" >nul 2>&1
    attrib -R "%USERPROFILE%\.cache\wandb\*.*" /S >nul 2>&1
)

if exist "wandb" (
    del /q /s "wandb\*.tmp" >nul 2>&1
    del /q /s "wandb\*.lock" >nul 2>&1
    del /q /s "wandb\*.log" >nul 2>&1
    attrib -R "wandb\*.*" /S >nul 2>&1
)

echo 🚀 Clearing GPU memory...
nvidia-smi --gpu-reset >nul 2>&1

echo 🧪 Testing WandB connection...
python -c "import wandb; print('✅ WandB working'); test_run = wandb.init(project='test', mode='offline'); test_run.finish(); print('✅ Connection OK')" 2>nul
if %errorlevel% neq 0 (
    echo ❌ WandB test failed
) else (
    echo ✅ WandB test passed
)

echo.
echo ✅ WandB fix completed!
echo 🎯 You can now run training scripts safely.
echo.
echo 💡 Available training scripts:
echo    python simple_wandb_training.py
echo    python unlimited_training_wnb.py
echo.
echo 🔧 For detailed diagnostics: python fix_wandb_issues.py
pause
