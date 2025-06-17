# Step 1: Check for Python 3.10.x
Write-Host "`n[1] Checking for Python >= 3.10 and < 3.11..."

$python = Get-Command "python" -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Error "❌ Python not found in PATH. Please install Python 3.10 and make sure it's accessible."
    exit 1
}

$versionOutput = & python --version
if ($versionOutput -match "Python (\d+)\.(\d+)\.(\d+)") {
    $major = [int]$matches[1]
    $minor = [int]$matches[2]
    if ($major -ne 3 -or $minor -ne 10) {
        Write-Error "❌ Detected Python $major.$minor — please install Python 3.10.x to proceed."
        exit 1
    }
} else {
    Write-Error "❌ Unable to detect Python version. Please verify your Python installation."
    exit 1
}

# Step 2: Create virtual environment
Write-Host "`n[2] Creating virtual environment 'emoenv'..."
python -m venv emoenv

# Step 3: Activate virtual environment
Write-Host "`n[3] Activating virtual environment..."
& .\emoenv\Scripts\Activate.ps1

# Step 4: Upgrade pip
Write-Host "`n[4] Upgrading pip..."
pip install --upgrade pip

# Step 5: Install PyTorch with CUDA 12.1
Write-Host "`n[5] Installing PyTorch (CUDA 12.1 for RTX 4060)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Step 6: Clean requirements.txt
Write-Host "`n[6] Cleaning requirements.txt..."
$cleanedLines = Get-Content requirements.txt | Where-Object {$_ -notmatch "@ file://"}
$cleanedLines | Set-Content cleaned_requirements.txt

# Step 7: Install remaining dependencies
Write-Host "`n[7] Installing other dependencies from cleaned_requirements.txt..."
pip install -r cleaned_requirements.txt

# Step 8: Test CUDA availability
Write-Host "`n[8] Verifying CUDA is working..."
python -c "
import torch
print('✅ torch.cuda.is_available():', torch.cuda.is_available())
if torch.cuda.is_available():
    print('🟢 GPU:', torch.cuda.get_device_name(0))
else:
    print('⚠️  CUDA not available — check driver and PyTorch installation.')
"

Write-Host "`n✅ Setup complete. Virtual environment: 'emoenv'"
