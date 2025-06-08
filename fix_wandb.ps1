# Fix WandB Connection Issues
# PowerShell script to quickly resolve WandB [WinError 10054] and cleanup issues

Write-Host "🔧 WandB Connection Fixer for Windows" -ForegroundColor Cyan
Write-Host "=" * 50

# Kill hanging WandB processes
Write-Host "`n🗑️ Killing WandB processes..." -ForegroundColor Yellow
Get-Process -Name "*wandb*" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Get-Process | Where-Object {$_.ProcessName -like "*python*" -and $_.CommandLine -like "*wandb*"} | Stop-Process -Force -ErrorAction SilentlyContinue

# Clear WandB cache directories
Write-Host "🧹 Cleaning WandB cache..." -ForegroundColor Yellow

$wandbDirs = @(
    "$env:USERPROFILE\.cache\wandb",
    "$env:USERPROFILE\.wandb", 
    "$env:USERPROFILE\wandb",
    ".\wandb"
)

foreach ($dir in $wandbDirs) {
    if (Test-Path $dir) {
        Write-Host "  Cleaning: $dir"
        
        # Remove temporary and lock files
        Get-ChildItem -Path $dir -Recurse -Include "*.tmp", "*.lock", "*.log", "*debug*" | Remove-Item -Force -ErrorAction SilentlyContinue
        
        # Remove read-only attributes
        attrib -R "$dir\*.*" /S 2>$null
    }
}

# Clear GPU memory if available
Write-Host "🚀 Clearing GPU memory..." -ForegroundColor Yellow
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    # Reset GPU if available
    nvidia-smi --gpu-reset 2>$null
}

# Test Python WandB connection
Write-Host "🧪 Testing WandB connection..." -ForegroundColor Yellow
try {
    python -c "
import wandb
print('✅ WandB import successful')
test_run = wandb.init(project='connection-test', mode='offline')
test_run.finish()
print('✅ WandB offline test passed')
"
    Write-Host "✅ WandB connection test passed" -ForegroundColor Green
} catch {
    Write-Host "❌ WandB connection test failed" -ForegroundColor Red
}

# Network connectivity test
Write-Host "🌐 Testing network connectivity..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "https://api.wandb.ai/" -UseBasicParsing -TimeoutSec 10
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ WandB API is reachable" -ForegroundColor Green
    }
} catch {
    Write-Host "❌ Cannot reach WandB API" -ForegroundColor Red
}

Write-Host "`n✅ WandB fix completed!" -ForegroundColor Green
Write-Host "🎯 You can now run your training scripts safely." -ForegroundColor Cyan
Write-Host "`n💡 To use robust WandB training:" -ForegroundColor Yellow
Write-Host "   python simple_wandb_training.py" -ForegroundColor White
Write-Host "   python unlimited_training_wnb.py" -ForegroundColor White

Write-Host "`n🔧 For detailed diagnostics, run:" -ForegroundColor Yellow
Write-Host "   python fix_wandb_issues.py" -ForegroundColor White
