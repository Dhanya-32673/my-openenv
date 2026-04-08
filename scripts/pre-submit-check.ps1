$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$pythonCandidates = @(
    ".\\.venv\\Scripts\\python",
    "..\\.venv\\Scripts\\python",
    ".\\.venv\\Scripts\\python.exe",
    "..\\.venv\\Scripts\\python.exe"
)

$python = $null
foreach ($candidate in $pythonCandidates) {
    if (Test-Path $candidate) {
        $python = $candidate
        break
    }
}

if (-not $python) {
    Write-Error "Missing virtual environment python at project or parent .venv"
}

$checks = [ordered]@{}

$checks.workspace = (Test-Path ".")
$checks.openenv_yaml_exists = (Test-Path ".\\openenv.yaml")
$checks.inference_exists = (Test-Path ".\\inference.py")
$checks.dockerfile_exists = (Test-Path ".\\Dockerfile")
$checks.readme_exists = (Test-Path ".\\README.md")
$checks.tasks_count = ((Get-ChildItem .\\tasks\\*_task.py -ErrorAction SilentlyContinue).Count)
$checks.graders_count = ((Get-ChildItem .\\graders\\grader_*.py -ErrorAction SilentlyContinue).Count)

& $python -m compileall env tasks graders inference.py | Out-Null
$checks.compile_ok = ($LASTEXITCODE -eq 0)

$inferenceOut = & $python .\\inference.py 2>&1
$checks.inference_exit_ok = ($LASTEXITCODE -eq 0)
$joinedInference = ($inferenceOut | Out-String)
$checks.has_start_logs = [bool](Select-String -InputObject $joinedInference -Pattern "[START]" -SimpleMatch)
$checks.has_step_logs = [bool](Select-String -InputObject $joinedInference -Pattern "[STEP]" -SimpleMatch)
$checks.has_end_logs = [bool](Select-String -InputObject $joinedInference -Pattern "[END]" -SimpleMatch)
$scoreLine = ($inferenceOut | Select-String "Final aggregate score:" | Select-Object -Last 1).Line
$checks.final_score_line = $scoreLine

$base = "http://127.0.0.1:7860"
try {
    $health = (Invoke-RestMethod -Method Get -Uri ($base + "/health") -TimeoutSec 3).status
    $checks.server_health = $health
} catch {
    $checks.server_health = "not-reachable"
}

try {
    Invoke-RestMethod -Method Post -Uri ($base + "/step") -ContentType "application/json" -Body '{"action_type":"classify_email"}' -TimeoutSec 3 -ErrorAction Stop | Out-Null
    $checks.invalid_payload_rejected = $false
} catch {
    $checks.invalid_payload_rejected = $true
}

try {
    docker --version | Out-Null
    $checks.docker_available = $true
} catch {
    $checks.docker_available = $false
}

$mandatoryPass = (
    $checks.workspace -and
    $checks.openenv_yaml_exists -and
    $checks.inference_exists -and
    $checks.dockerfile_exists -and
    $checks.readme_exists -and
    ($checks.tasks_count -ge 3) -and
    ($checks.graders_count -ge 3) -and
    $checks.compile_ok -and
    $checks.inference_exit_ok -and
    $checks.has_start_logs -and
    $checks.has_step_logs -and
    $checks.has_end_logs -and
    ($checks.server_health -eq "ok") -and
    $checks.invalid_payload_rejected
)

$checks.mandatory_pass = $mandatoryPass

$checks | ConvertTo-Json -Depth 5
