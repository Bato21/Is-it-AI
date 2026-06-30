# setup_envs.ps1 - crea los tres entornos virtuales aislados (Windows / PowerShell).
# Ejecutar desde la raíz del repo:  ./setup_envs.ps1
#
# Usa python.org Python 3.11. Para otro intérprete:  ./setup_envs.ps1 -Python "C:\ruta\python.exe"

param([string]$Python = "")

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

function Resolve-Python {
    param([string]$override)
    if ($override -and (Test-Path $override)) { return $override }
    $cands = @("$env:LOCALAPPDATA\Programs\Python\Python311\python.exe", "C:\Python311\python.exe")
    foreach ($c in $cands) { if (Test-Path $c) { return $c } }
    throw "No se encontró Python 3.11. Instálalo (winget install Python.Python.3.11) o pasa -Python <ruta>."
}

function New-Venv {
    param([string]$dir, [string]$reqs, [string]$py)
    $venv = Join-Path $dir ".venv"
    Write-Host "`n=== $dir ===" -ForegroundColor Cyan
    if (-not (Test-Path $venv)) { & $py -m venv $venv }
    $vpy = Join-Path $venv "Scripts\python.exe"
    & $vpy -m pip install --upgrade pip
    & $vpy -m pip install -r $reqs
    Write-Host "listo: $dir" -ForegroundColor Green
}

$py = Resolve-Python $Python
Write-Host "Usando Python: $py"

# Un venv por carpeta de framework + uno para tools. Cada uno lee su requirements.
New-Venv -dir (Join-Path $root "tools")      -reqs (Join-Path $root "requirements-tools.txt")      -py $py
New-Venv -dir (Join-Path $root "PyTorch")    -reqs (Join-Path $root "requirements-pytorch.txt")    -py $py
New-Venv -dir (Join-Path $root "TensorFlow") -reqs (Join-Path $root "requirements-tensorflow.txt") -py $py

Write-Host "`nListo. Activa con, por ejemplo:  PyTorch\.venv\Scripts\Activate.ps1" -ForegroundColor Green
