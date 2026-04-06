$ErrorActionPreference = 'Stop'

param(
    [string]$ModelDir,
    [string]$ModelFamily,
    [string]$RunName = "epoch_15",
    [string]$InputCsv = "data/processed/dataset_absa_50k_v2_intersection.csv",
    [int]$NumMc = 30,
    [int]$BatchSize = 16,
    [string]$UncertaintyCol = "uncertainty_entropy",
    [double]$HighUncertaintyQuantile = 0.8,
    [string]$ThresholdScope = "global"
)

$python = ".\.venv\Scripts\python.exe"

if (-not $ModelDir) {
    throw "ModelDir is required."
}

if (-not $ModelFamily) {
    throw "ModelFamily is required."
}

$uncertaintyDir = "data/processed/uncertainty/$ModelFamily/$RunName"
$noiseDir = "data/processed/noise/$ModelFamily/$RunName"

Write-Host "[UNCERTAINTY] Running MC Dropout -> $uncertaintyDir"
& $python src/evaluation/predict_mc_dropout.py `
    --input_csv $InputCsv `
    --model_dir $ModelDir `
    --output_dir "data/processed/uncertainty" `
    --model_family $ModelFamily `
    --run_name $RunName `
    --num_mc $NumMc `
    --batch_size $BatchSize

if ($LASTEXITCODE -ne 0) {
    throw "MC Dropout failed for $ModelFamily / $RunName"
}

Write-Host "[UNCERTAINTY] Detecting noisy candidates -> $noiseDir"
& $python src/evaluation/detect_label_noise.py `
    --input_dir $uncertaintyDir `
    --output_dir "data/processed/noise" `
    --model_family $ModelFamily `
    --run_name $RunName `
    --uncertainty_col $UncertaintyCol `
    --high_uncertainty_quantile $HighUncertaintyQuantile `
    --threshold_scope $ThresholdScope

if ($LASTEXITCODE -ne 0) {
    throw "Noise detection failed for $ModelFamily / $RunName"
}

Write-Host "[UNCERTAINTY] Pipeline completed for $ModelFamily / $RunName"
