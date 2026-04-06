$ErrorActionPreference = 'Stop'

param(
    [switch]$IncludeQloRA
)

$python = ".\.venv\Scripts\python.exe"
$maxLength = 128
$testSize = 0.2
$valSize = 0.1
$seed = 42
$maxEpoch = 15
$sharedBatchSize = 8
$baselineLearningRate = 2e-5
$peftLearningRate = 2e-4
$qloraLearningRate = 2e-5

$jobs = @(
    @{
        label = "RETRAIN"
        clean_csv = "data/processed/noise/baseline/epoch_15/clean_data.csv"
        output_dir = "models/retrained/epoch_15"
        script = "src/training/retrain_filtered.py"
        experiment_family = "retrained"
        uncertainty_source_model_id = "baseline:epoch_15"
        source_family = "baseline"
        lr = $baselineLearningRate
    },
    @{
        label = "RETRAIN-LORA"
        clean_csv = "data/processed/noise/lora/epoch_15/clean_data.csv"
        output_dir = "models/retrained_lora/epoch_15"
        script = "src/training/train_lora_filtered.py"
        experiment_family = "retrained_lora"
        uncertainty_source_model_id = "lora:epoch_15"
        source_family = "lora"
        lr = $peftLearningRate
    },
    @{
        label = "RETRAIN-DORA"
        clean_csv = "data/processed/noise/dora/epoch_15/clean_data.csv"
        output_dir = "models/retrained_dora/epoch_15"
        script = "src/training/train_dora_filtered.py"
        experiment_family = "retrained_dora"
        uncertainty_source_model_id = "dora:epoch_15"
        source_family = "dora"
        lr = $peftLearningRate
    },
    @{
        label = "RETRAIN-ADALORA"
        clean_csv = "data/processed/noise/adalora/epoch_15/clean_data.csv"
        output_dir = "models/retrained_adalora/epoch_15"
        script = "src/training/train_adalora_filtered.py"
        experiment_family = "retrained_adalora"
        uncertainty_source_model_id = "adalora:epoch_15"
        source_family = "adalora"
        lr = $peftLearningRate
    }
)

if ($IncludeQloRA) {
    $jobs += @{
        label = "RETRAIN-QLORA"
        clean_csv = "data/processed/noise/qlora/epoch_15/clean_data.csv"
        output_dir = "models/retrained_qlora/epoch_15"
        script = "src/training/train_qlora_filtered.py"
        experiment_family = "retrained_qlora"
        uncertainty_source_model_id = "qlora:epoch_15"
        source_family = "qlora"
        lr = $qloraLearningRate
    }
}

foreach ($job in $jobs) {
    if (-not (Test-Path $job.clean_csv)) {
        Write-Host "[$($job.label)] Skip because clean_csv not found: $($job.clean_csv)"
        continue
    }

    $arguments = @(
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
        "--experiment_family", $job.experiment_family,
        "--uncertainty_source_model_id", $job.uncertainty_source_model_id,
        "--noise_summary_json", ("data/processed/noise/{0}/epoch_15/noise_summary.json" -f $job.source_family),
        "--mc_summary_json", ("data/processed/uncertainty/{0}/epoch_15/mc_summary.json" -f $job.source_family)
    )

    Write-Host "[$($job.label)] Running -> $($job.output_dir)"
    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$($job.label) run failed"
    }
}

Write-Host "[UNCERTAINTY-RETRAIN] Family-aware retraining completed."
