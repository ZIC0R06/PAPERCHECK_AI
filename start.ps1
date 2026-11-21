# =====================================================================
# AI Paper Checking — University AI Examination System Launcher
# =====================================================================
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "  🎓 AI Paper Checking : University AI Examination Platform" -ForegroundColor Yellow
Write-Host "=====================================================================" -ForegroundColor Cyan

Set-Location $PSScriptRoot

$pythonExe = $null

$candidates = @(
    "$env:LOCALAPPDATA\Programs\Python\Python314\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$PSScriptRoot\..\.venv\Scripts\python.exe",
    (Get-Command python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source),
    (Get-Command py -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)
)

foreach ($cand in $candidates) {
    if ($cand -and (Test-Path $cand)) {
        $pythonExe = $cand
        break
    }
}

if (-not $pythonExe) {
    Write-Host "[ERROR] Python was not found in standard installation paths." -ForegroundColor Red
    Write-Host "Please ensure Python 3.12+ is installed." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "[OK] Using Python: $pythonExe" -ForegroundColor Green

# 1. Initialize and migrate database with demo records
Write-Host "[*] Initializing University Database..." -ForegroundColor Cyan
& $pythonExe run.py init

Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "  Web Dashboard:  http://localhost:8501" -ForegroundColor Yellow
Write-Host "  FastAPI Docs:   http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "  Default Accounts:" -ForegroundColor White
Write-Host "    Admin:   admin   / admin123" -ForegroundColor Gray
Write-Host "    Faculty: faculty / faculty123" -ForegroundColor Gray
Write-Host "    Student: student / student123" -ForegroundColor Gray
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

# 2. Launch application
& $pythonExe run.py
