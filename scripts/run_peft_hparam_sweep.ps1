$ErrorActionPreference = 'Stop'

param(
    [ValidateSet('lora','dora','adalora','qlora')]
    [string]$Family = 'lora',
    [switch]$Filtered,
    [string]$InputCsv = 'data/processed/dataset_absa_50k_v2_intersection.csv',
    [string]$CleanCsv = '',
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
$lrCandidates = @($cfg.peft_lr_candidates)
$dropoutCandidates = @($cfg.peft_dropout_candidates)
$rankCandidates = if ($Family -eq 'adalora') { @($cfg.adalora_init_r_candidates) } else { @($cfg.peft_r_candidates) }
$scriptName = if ($Filtered) { "src/training/train_${Family}_filtered.py" } else { "src/training/train_${Family}.py" }
$scope = if ($Filtered) { "filtered" } else { "standard" }
$rootOutput = "experiments/hparam_sweeps/$Family/$scope"
$results = @()

if ($Filtered -and [string]::IsNullOrWhiteSpace($CleanCsv)) {
    throw "Set -CleanCsv when using -Filtered."
}

foreach ($lr in $lrCandidates) {
    foreach ($rank in $rankCandidates) {
        foreach ($dropout in $dropoutCandidates) {
            $lrToken = ("{0}" -f $lr).Replace('.', 'p')
            $dropoutToken = ("{0}" -f $dropout).Replace('.', 'p')
            $outputDir = "$rootOutput/lr_$lrToken/r_$rank/dropout_$dropoutToken"
            $args = @(
                $scriptName,
                "--model_name", $ModelName,
                "--output_dir", $outputDir,
                "--max_length", $MaxLength,
                "--test_size", $TestSize,
                "--val_size", $ValSize,
                "--epochs", $epochs,
                "--batch_size", $batchSize,
                "--lr", $lr,
                "--seed", $Seed,
                "--peft_alpha", 32,
                "--peft_dropout", $dropout,
                "--experiment_family", "${Family}_tuning"
            )

            if ($Family -eq 'adalora') {
                $targetRank = [int]$cfg.adalora_target_r_default
                if ($rank -lt $targetRank) {
                    $targetRank = [int][Math]::Max(4, [Math]::Floor($rank / 2))
                }
                $args += @(
                    "--adalora_init_r", $rank,
                    "--adalora_target_r", $targetRank,
                    "--peft_r", $rank
                )
            } else {
                $args += @("--peft_r", $rank)
            }

            if ($Filtered) {
                $args += @(
                    "--clean_csv", $CleanCsv
                )
            } else {
                $args += @(
                    "--input_csv", $InputCsv
                )
            }

            Write-Host "[$Family] Sweep run lr=$lr rank=$rank dropout=$dropout -> $outputDir"
            & $python @args
            if ($LASTEXITCODE -ne 0) {
                throw "$Family sweep failed for lr=$lr rank=$rank dropout=$dropout"
            }

            $metricsPath = Join-Path $outputDir "metrics.json"
            $metrics = Get-Content $metricsPath -Raw | ConvertFrom-Json
            $results += [PSCustomObject]@{
                family = $Family
                scope = $scope
                learning_rate = [double]$lr
                rank = [int]$rank
                dropout = [double]$dropout
                best_epoch = $metrics.best_epoch
                best_validation_f1_macro = $metrics.best_validation_f1_macro
                best_validation_accuracy = $metrics.best_validation_accuracy
                best_validation_precision_macro = $metrics.best_validation_precision_macro
                best_validation_recall_macro = $metrics.best_validation_recall_macro
                training_time_seconds = $metrics.training_time_seconds
                output_dir = $outputDir
            }
        }
    }
}

New-Item -ItemType Directory -Force -Path $rootOutput | Out-Null
$csvPath = Join-Path $rootOutput "sweep_summary.csv"
$jsonPath = Join-Path $rootOutput "sweep_summary.json"
$results | Export-Csv -Path $csvPath -NoTypeInformation -Encoding UTF8
$results | ConvertTo-Json -Depth 5 | Set-Content -Path $jsonPath -Encoding UTF8

Write-Host "[SWEEP] Summary written to $csvPath"
