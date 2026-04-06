$ErrorActionPreference = 'Stop'

param(
    [ValidateSet('baseline','retrained')]
    [string]$Family = 'baseline',
    [string]$InputCsv = 'data/processed/dataset_absa_50k_v2_intersection.csv',
    [string]$CleanCsv = 'data/processed/noise/baseline/epoch_15/clean_data.csv',
    [string]$ModelName = 'indobenchmark/indobert-base-p1',
    [int]$MaxLength = 128,
    [double]$TestSize = 0.2,
    [double]$ValSize = 0.1,
    [int]$Seed = 42
)

$python = ".\.venv\Scripts\python.exe"
$cfg = & $python scripts/export_hparam_sweep_config.py | ConvertFrom-Json
$epochs = [int]$cfg.train_max_epochs
$batchSize = [int]$cfg.train_batch_size
$lrCandidates = @($cfg.full_finetune_lr_candidates)
$rootOutput = "experiments/hparam_sweeps/$Family"
$results = @()

foreach ($lr in $lrCandidates) {
    $lrToken = ("{0}" -f $lr).Replace('.', 'p')
    $outputDir = "$rootOutput/lr_$lrToken"
    $args = @()

    if ($Family -eq 'baseline') {
        $args = @(
            "src/training/train_baseline.py",
            "--input_csv", $InputCsv,
            "--model_name", $ModelName,
            "--output_dir", $outputDir,
            "--max_length", $MaxLength,
            "--test_size", $TestSize,
            "--val_size", $ValSize,
            "--epochs", $epochs,
            "--batch_size", $batchSize,
            "--lr", $lr,
            "--seed", $Seed,
            "--experiment_family", "baseline_tuning"
        )
    } else {
        if (-not (Test-Path $CleanCsv)) {
            throw "Clean CSV not found for retrained sweep: $CleanCsv"
        }
        $args = @(
            "src/training/retrain_filtered.py",
            "--clean_csv", $CleanCsv,
            "--model_name", $ModelName,
            "--output_dir", $outputDir,
            "--max_length", $MaxLength,
            "--test_size", $TestSize,
            "--val_size", $ValSize,
            "--epochs", $epochs,
            "--batch_size", $batchSize,
            "--lr", $lr,
            "--seed", $Seed,
            "--experiment_family", "retrained_tuning"
        )
    }

    Write-Host "[$Family] Sweep run lr=$lr -> $outputDir"
    & $python @args
    if ($LASTEXITCODE -ne 0) {
        throw "$Family sweep failed for lr=$lr"
    }

    $metricsPath = Join-Path $outputDir "metrics.json"
    $metrics = Get-Content $metricsPath -Raw | ConvertFrom-Json
    $results += [PSCustomObject]@{
        family = $Family
        learning_rate = [double]$lr
        best_epoch = $metrics.best_epoch
        best_validation_f1_macro = $metrics.best_validation_f1_macro
        best_validation_accuracy = $metrics.best_validation_accuracy
        best_validation_precision_macro = $metrics.best_validation_precision_macro
        best_validation_recall_macro = $metrics.best_validation_recall_macro
        training_time_seconds = $metrics.training_time_seconds
        output_dir = $outputDir
    }
}

$summaryDir = $rootOutput
New-Item -ItemType Directory -Force -Path $summaryDir | Out-Null
$csvPath = Join-Path $summaryDir "sweep_summary.csv"
$jsonPath = Join-Path $summaryDir "sweep_summary.json"
$results | Export-Csv -Path $csvPath -NoTypeInformation -Encoding UTF8
$results | ConvertTo-Json -Depth 5 | Set-Content -Path $jsonPath -Encoding UTF8

Write-Host "[SWEEP] Summary written to $csvPath"
