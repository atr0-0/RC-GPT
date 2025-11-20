Write-Host "=== CaseLawGPT Setup ===" -ForegroundColor Cyan

# 1. Python Setup
Write-Host "`n[1/2] Setting up Python Environment..." -ForegroundColor Yellow
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
} else {
    Write-Host "Virtual environment already exists."
}

Write-Host "Installing Python dependencies..."
./venv/Scripts/python -m pip install -r requirements.txt

# 2. Frontend Setup
Write-Host "`n[2/2] Setting up Frontend..." -ForegroundColor Yellow
if (Test-Path "frontend") {
    Push-Location frontend
    if (-not (Test-Path "node_modules")) {
        Write-Host "Installing Node modules..."
        npm install
    } else {
        Write-Host "Node modules already installed."
    }
    Pop-Location
} else {
    Write-Host "Error: frontend directory not found!" -ForegroundColor Red
}

Write-Host "`n=== Setup Complete! ===" -ForegroundColor Green
Write-Host "You can now run the application using: .\run.ps1"
