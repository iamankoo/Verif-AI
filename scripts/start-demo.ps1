# Verif-AI local examiner demonstration: one command starts the backend (E4 loads as the default model) and serves the dashboard.
#   powershell -ExecutionPolicy Bypass -File scripts\start-demo.ps1            (or double-click scripts\start-demo.bat)
# What it does: (1) builds the frontend once if frontend\dist is missing, (2) picks the first free port from 8000, (3) starts the API
# with backend\.env (VERIFAI_MODEL_PATH=models/verifai_e4_candidate.pth), (4) waits for /api/health and prints the URL, (5) opens the browser.
# Stop with Ctrl+C. Nothing is uploaded anywhere; analyses are stored under backend\storage and deleted after 72 hours or on request.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$backend = Join-Path $root "backend"
$frontend = Join-Path $root "frontend"
$py = Join-Path $backend ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

if (-not (Test-Path (Join-Path $frontend "dist\index.html"))) {
    Write-Host "Building the frontend (first run only)..."
    Push-Location $frontend
    if (-not (Test-Path "node_modules")) { npm ci --no-audit --no-fund }
    npm run build
    Pop-Location
}

$e4 = Join-Path $backend "models\verifai_e4_candidate.pth"
if (-not (Test-Path $e4)) {
    Write-Warning "E4 checkpoint not found at $e4 - the server will fall back to Model 4 / v2. See docs/E4_INTEGRATION.md."
}

$port = 8000
while (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) { $port++ }
$env:VERIFAI_FRONTEND_DIR = $frontend + "\dist"
Write-Host "Starting Verif-AI on http://127.0.0.1:$port ..."
Push-Location $backend
$proc = Start-Process -FilePath $py -ArgumentList "-m", "uvicorn", "verifai.api.app:app", "--host", "127.0.0.1", "--port", $port -PassThru -NoNewWindow
Pop-Location
try {
    $ok = $false
    for ($i = 0; $i -lt 90 -and -not $ok; $i++) {
        Start-Sleep -Seconds 2
        try { $h = Invoke-RestMethod "http://127.0.0.1:$port/api/health" -TimeoutSec 3; $ok = ($h.status -eq "ok") } catch { }
    }
    if (-not $ok) { throw "The server did not become healthy in time." }
    $m = Invoke-RestMethod "http://127.0.0.1:$port/api/model"
    Write-Host ""
    Write-Host "READY: http://127.0.0.1:$port"
    Write-Host ("Default model: {0} ({1}), checkpoint {2}, sha256 {3}..." -f $m.model.registry_label, $m.model.role_label, $m.model.checkpoint_file, $m.model.checkpoint_sha256.Substring(0, 12))
    if (-not $env:VERIFAI_NO_BROWSER) { Start-Process "http://127.0.0.1:$port" }
    Write-Host "Press Ctrl+C to stop."
    Wait-Process -Id $proc.Id
} finally {
    if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
}
