Write-Host "=== Starting CaseLawGPT ===" -ForegroundColor Cyan

# Check for venv
if (-not (Test-Path "venv")) {
    Write-Host "Error: Virtual environment not found. Please run .\setup.ps1 first." -ForegroundColor Red
    exit
}

# 1. Start Backend
Write-Host "`n[1/2] Launching Backend (FastAPI)..." -ForegroundColor Yellow
$backendJob = Start-Process -FilePath "venv\Scripts\python.exe" -ArgumentList "-m", "uvicorn", "src.api:app", "--reload", "--port", "8000" -PassThru -WindowStyle Minimized
Write-Host "Backend running on http://localhost:8000 (PID: $($backendJob.Id))"

# 2. Start Frontend
Write-Host "`n[2/2] Launching Frontend (Vite)..." -ForegroundColor Yellow
if (Test-Path "frontend") {
    Push-Location frontend
    # We use cmd /c to run npm so it stays in this console or opens properly
    npm run dev
    Pop-Location
} else {
    Write-Host "Error: frontend directory not found!" -ForegroundColor Red
}

# When frontend stops, we should probably kill backend, but for now let's just leave it.
Write-Host "`nBackend is still running in background."
Write-Host "To stop it, close the minimized window or run: Stop-Process -Id $($backendJob.Id)"
