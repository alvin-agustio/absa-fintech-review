$ErrorActionPreference = 'Stop'

$python = ".\.venv\Scripts\python.exe"
$inputCsv = "data/processed/dataset_absa_50k_v2_intersection.csv"
$modelName = "indobenchmark/indobert-base-p1"
$maxLength = 128
$testSize = 0.2
$valSize = 0.1
$batchSize = 8
$learningRate = 2e-5
$seed = 42
$maxEpoch = 15
$outputDir = "models/baseline/epoch_$maxEpoch"

Write-Host "[BASELINE] Running max_epoch=$maxEpoch -> $outputDir"
& $python src/training/train_baseline.py `
    --input_csv $inputCsv `
    --model_name $modelName `
    --output_dir $outputDir `
    --max_length $maxLength `
    --test_size $testSize `
    --val_size $valSize `
    --epochs $maxEpoch `
    --batch_size $batchSize `
    --lr $learningRate `
    --seed $seed

if ($LASTEXITCODE -ne 0) {
    throw "Baseline run failed for max_epoch=$maxEpoch"
}

Write-Host "[BASELINE] Run completed successfully."
