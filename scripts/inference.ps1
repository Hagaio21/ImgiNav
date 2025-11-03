# PowerShell script to run diffusion model inference
# Usage: .\scripts\inference.ps1

param(
    [string]$Checkpoint = "checkpoints/diffusion_ablation_capacity_unet64_d4_checkpoint_best.pt",
    [string]$Config = "experiments/diffusion/ablation/capacity_unet64_d4.yaml",
    [string]$Output = "outputs/inference_samples",
    [int]$NumSamples = 16,
    [int]$BatchSize = 4,
    [ValidateSet("ddpm", "ddim")]
    [string]$Method = "ddpm",
    [int]$NumSteps = 0,
    [float]$Eta = 0.0,
    [string]$Device = $null,
    [switch]$SaveLatents
)

$pythonArgs = @(
    "scripts/inference.py"
    "--checkpoint", $Checkpoint
    "--config", $Config
    "--output", $Output
    "--num_samples", $NumSamples
    "--batch_size", $BatchSize
    "--method", $Method
)

if ($NumSteps -gt 0) {
    $pythonArgs += "--num_steps", $NumSteps
}

if ($Eta -ne 0.0) {
    $pythonArgs += "--eta", $Eta
}

if ($Device) {
    $pythonArgs += "--device", $Device
}

if ($SaveLatents) {
    $pythonArgs += "--save_latents"
}

Write-Host "Running inference..." -ForegroundColor Green
Write-Host "Command: python $($pythonArgs -join ' ')" -ForegroundColor Cyan

python $pythonArgs[0] $pythonArgs[1..($pythonArgs.Length-1)]

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nInference completed successfully!" -ForegroundColor Green
    Write-Host "Results saved to: $Output" -ForegroundColor Cyan
} else {
    Write-Host "`nInference failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

