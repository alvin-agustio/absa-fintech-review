$ErrorActionPreference = 'Stop'

param(
    [switch]$IncludeQloRA
)

$python = ".\.venv\Scripts\python.exe"
$inputCsv = "data/processed/dataset_absa_50k_v2_intersection.csv"
$modelName = "indobenchmark/indobert-base-p1"
$maxLength = 128
$testSize = 0.2
$valSize = 0.1
$seed = 42
$maxEpoch = 15
$sharedBatchSize = 8

$baselineLearningRate = 2e-5
$peftLearningRate = 2e-4
$qloraLearningRate = 2e-5

$trainJobs = @(
    @{
        label = "BASELINE"
        model = "baseline"
        script = "src/training/train_baseline.py"
        output_dir = "models/baseline/epoch_15"
        lr = $baselineLearningRate
    },
    @{
        label = "LORA"
        model = "lora"
        script = "src/training/train_lora.py"
        output_dir = "models/lora/epoch_15"
        lr = $peftLearningRate
    },
    @{
        label = "DORA"
        model = "dora"
        script = "src/training/train_dora.py"
        output_dir = "models/dora/epoch_15"
        lr = $peftLearningRate
    },
    @{
        label = "ADALORA"
        model = "adalora"
        script = "src/training/train_adalora.py"
        output_dir = "models/adalora/epoch_15"
        lr = $peftLearningRate
    }
)

if ($IncludeQloRA) {
    $trainJobs += @{
        label = "QLORA"
        model = "qlora"
        script = "src/training/train_qlora.py"
        output_dir = "models/qlora/epoch_15"
        lr = $qloraLearningRate
    }
}

$results = @()

function Invoke-TrainingRun {
    param(
        [string]$Label,
        [string]$OutputDir,
        [string[]]$Arguments
    )

    Write-Host "[$Label] Running -> $OutputDir"
    & $python @Arguments

    if ($LASTEXITCODE -ne 0) {
        throw "$Label run failed for output dir: $OutputDir"
    }
}

function Read-Metrics {
    param(
        [string]$ModelType,
        [string]$OutputDir
    )

    $metricsPath = Join-Path $OutputDir "metrics.json"
    if (-not (Test-Path $metricsPath)) {
        throw "Metrics file not found: $metricsPath"
    }

    $metrics = Get-Content $metricsPath -Raw | ConvertFrom-Json
    return [PSCustomObject]@{
        model = $ModelType
        comparison_group = $metrics.comparison_group
        max_epochs = $maxEpoch
        best_epoch = if ($null -ne $metrics.best_epoch) { [int]$metrics.best_epoch } else { $null }
        best_validation_f1_macro = if ($null -ne $metrics.best_validation_f1_macro) { [double]$metrics.best_validation_f1_macro } else { $null }
        best_validation_accuracy = if ($null -ne $metrics.best_validation_accuracy) { [double]$metrics.best_validation_accuracy } else { $null }
        best_validation_precision_macro = if ($null -ne $metrics.best_validation_precision_macro) { [double]$metrics.best_validation_precision_macro } else { $null }
        best_validation_recall_macro = if ($null -ne $metrics.best_validation_recall_macro) { [double]$metrics.best_validation_recall_macro } else { $null }
        training_time_seconds = if ($null -ne $metrics.training_time_seconds) { [double]$metrics.training_time_seconds } else { $null }
        trainable_params = if ($null -ne $metrics.trainable_params) { [double]$metrics.trainable_params } else { $null }
        output_dir = $OutputDir
    }
}

foreach ($job in $trainJobs) {
    Invoke-TrainingRun `
        -Label $job.label `
        -OutputDir $job.output_dir `
        -Arguments @(
            $job.script,
            "--input_csv", $inputCsv,
            "--model_name", $modelName,
            "--output_dir", $job.output_dir,
            "--max_length", $maxLength,
            "--test_size", $testSize,
            "--val_size", $valSize,
            "--epochs", $maxEpoch,
            "--batch_size", $sharedBatchSize,
            "--lr", $job.lr,
            "--seed", $seed,
            "--experiment_family", $job.model
        )
    $results += Read-Metrics -ModelType $job.model -OutputDir $job.output_dir
}

$retrainJobs = @(
    @{
        label = "RETRAIN"
        model = "retrained"
        script = "src/training/retrain_filtered.py"
        clean_csv = "data/processed/noise/baseline/epoch_15/clean_data.csv"
        output_dir = "models/retrained/epoch_15"
        lr = $baselineLearningRate
        uncertainty_source_model_id = "baseline:epoch_15"
        noise_summary_json = "data/processed/noise/baseline/epoch_15/noise_summary.json"
        mc_summary_json = "data/processed/uncertainty/baseline/epoch_15/mc_summary.json"
    },
    @{
        label = "RETRAIN-LORA"
        model = "retrained_lora"
        script = "src/training/train_lora_filtered.py"
        clean_csv = "data/processed/noise/lora/epoch_15/clean_data.csv"
        output_dir = "models/retrained_lora/epoch_15"
        lr = $peftLearningRate
        uncertainty_source_model_id = "lora:epoch_15"
        noise_summary_json = "data/processed/noise/lora/epoch_15/noise_summary.json"
        mc_summary_json = "data/processed/uncertainty/lora/epoch_15/mc_summary.json"
    },
    @{
        label = "RETRAIN-DORA"
        model = "retrained_dora"
        script = "src/training/train_dora_filtered.py"
        clean_csv = "data/processed/noise/dora/epoch_15/clean_data.csv"
        output_dir = "models/retrained_dora/epoch_15"
        lr = $peftLearningRate
        uncertainty_source_model_id = "dora:epoch_15"
        noise_summary_json = "data/processed/noise/dora/epoch_15/noise_summary.json"
        mc_summary_json = "data/processed/uncertainty/dora/epoch_15/mc_summary.json"
    },
    @{
        label = "RETRAIN-ADALORA"
        model = "retrained_adalora"
        script = "src/training/train_adalora_filtered.py"
        clean_csv = "data/processed/noise/adalora/epoch_15/clean_data.csv"
        output_dir = "models/retrained_adalora/epoch_15"
        lr = $peftLearningRate
        uncertainty_source_model_id = "adalora:epoch_15"
        noise_summary_json = "data/processed/noise/adalora/epoch_15/noise_summary.json"
        mc_summary_json = "data/processed/uncertainty/adalora/epoch_15/mc_summary.json"
    }
)

if ($IncludeQloRA) {
    $retrainJobs += @{
        label = "RETRAIN-QLORA"
        model = "retrained_qlora"
        script = "src/training/train_qlora_filtered.py"
        clean_csv = "data/processed/noise/qlora/epoch_15/clean_data.csv"
        output_dir = "models/retrained_qlora/epoch_15"
        lr = $qloraLearningRate
        uncertainty_source_model_id = "qlora:epoch_15"
        noise_summary_json = "data/processed/noise/qlora/epoch_15/noise_summary.json"
        mc_summary_json = "data/processed/uncertainty/qlora/epoch_15/mc_summary.json"
    }
}

foreach ($job in $retrainJobs) {
    if (-not (Test-Path $job.clean_csv)) {
        Write-Host "[RUNNER] Skip $($job.label) because clean data was not found at $($job.clean_csv)"
        continue
    }

    Invoke-TrainingRun `
        -Label $job.label `
        -OutputDir $job.output_dir `
        -Arguments @(
            $job.script,
            "--clean_csv", $job.clean_csv,
            "--output_dir", $job.output_dir,
            "--max_length", $maxLength,
            "--test_size", $testSize,
            "--val_size", $valSize,
            "--epochs", $maxEpoch,
            "--batch_size", $sharedBatchSize,
            "--lr", $job.lr,
            "--seed", $seed,
            "--experiment_family", $job.model,
            "--uncertainty_source_model_id", $job.uncertainty_source_model_id,
            "--noise_summary_json", $job.noise_summary_json,
            "--mc_summary_json", $job.mc_summary_json
        )
    $results += Read-Metrics -ModelType $job.model -OutputDir $job.output_dir
}

Write-Host ""
Write-Host "[RUNNER] Validation-driven training summary"
$results |
    Sort-Object model |
    Select-Object `
        model,
        max_epochs,
        best_epoch,
        @{Name = "val_f1_macro"; Expression = { if ($null -ne $_.best_validation_f1_macro) { "{0:N4}" -f $_.best_validation_f1_macro } else { "N/A" } } },
        @{Name = "val_acc"; Expression = { if ($null -ne $_.best_validation_accuracy) { "{0:N4}" -f $_.best_validation_accuracy } else { "N/A" } } },
        @{Name = "val_precision"; Expression = { if ($null -ne $_.best_validation_precision_macro) { "{0:N4}" -f $_.best_validation_precision_macro } else { "N/A" } } },
        @{Name = "val_recall"; Expression = { if ($null -ne $_.best_validation_recall_macro) { "{0:N4}" -f $_.best_validation_recall_macro } else { "N/A" } } },
        @{Name = "time_s"; Expression = { if ($null -ne $_.training_time_seconds) { "{0:N2}" -f $_.training_time_seconds } else { "N/A" } } },
        @{Name = "trainable_params"; Expression = { if ($null -ne $_.trainable_params) { "{0:N0}" -f $_.trainable_params } else { "N/A" } } },
        output_dir |
    Format-Table -AutoSize

Write-Host ""
Write-Host "[RUNNER] All requested training runs up to epoch $maxEpoch completed successfully."
