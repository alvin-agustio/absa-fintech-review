$ErrorActionPreference = 'Stop'

Write-Host "[RUNNER] Starting baseline experiments"
& .\scripts\run_baseline_epochs.ps1

Write-Host "[RUNNER] Starting LoRA experiments"
& .\scripts\run_lora_epochs.ps1

Write-Host "[RUNNER] All baseline and LoRA experiments completed successfully."
