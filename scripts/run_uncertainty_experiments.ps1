$ErrorActionPreference = 'Stop'

param(
    [switch]$IncludeQloRA
)

$python = ".\.venv\Scripts\python.exe"
$inputCsv = "data/processed/dataset_absa_50k_v2_intersection.csv"
$maxLength = 128
$numMc = 30
$batchSize = 16
$uncertaintyCol = "uncertainty_entropy"
$highUncertaintyQuantile = 0.8

$families = @(
    @{ model_family = "baseline"; run_name = "epoch_15"; model_dir = "models/baseline/epoch_15/model" },
    @{ model_family = "lora"; run_name = "epoch_15"; model_dir = "models/lora/epoch_15/model" },
    @{ model_family = "dora"; run_name = "epoch_15"; model_dir = "models/dora/epoch_15/model" },
    @{ model_family = "adalora"; run_name = "epoch_15"; model_dir = "models/adalora/epoch_15/model" }
)

if ($IncludeQloRA) {
    $families += @{ model_family = "qlora"; run_name = "epoch_15"; model_dir = "models/qlora/epoch_15/model" }
}

foreach ($family in $families) {
    if (-not (Test-Path $family.model_dir)) {
        Write-Host "[UNCERTAINTY] Skip $($family.model_family) because model_dir not found: $($family.model_dir)"
        continue
    }

    Write-Host "[UNCERTAINTY] MC Dropout -> $($family.model_family) / $($family.run_name)"
    & $python src/evaluation/predict_mc_dropout.py `
        --input_csv $inputCsv `
        --model_dir $family.model_dir `
        --model_family $family.model_family `
        --run_name $family.run_name `
        --output_dir data/processed/uncertainty `
        --num_mc $numMc `
        --max_length $maxLength `
        --batch_size $batchSize
    if ($LASTEXITCODE -ne 0) {
        throw "MC Dropout failed for $($family.model_family)"
    }

    Write-Host "[UNCERTAINTY] Noise detection -> $($family.model_family) / $($family.run_name)"
    & $python src/evaluation/detect_label_noise.py `
        --input_dir ("data/processed/uncertainty/{0}/{1}" -f $family.model_family, $family.run_name) `
        --model_family $family.model_family `
        --run_name $family.run_name `
        --output_dir data/processed/noise `
        --uncertainty_col $uncertaintyCol `
        --high_uncertainty_quantile $highUncertaintyQuantile `
        --threshold_scope global
    if ($LASTEXITCODE -ne 0) {
        throw "Noise detection failed for $($family.model_family)"
    }
}

Write-Host "[UNCERTAINTY] Family-aware uncertainty runs completed."
