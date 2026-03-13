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
$epochsList = @(3, 5)

foreach ($epochs in $epochsList) {
    $outputDir = "models/baseline/epoch_$epochs"
    Write-Host "[BASELINE] Running epoch=$epochs -> $outputDir"
    & $python train_baseline.py `
        --input_csv $inputCsv `
        --model_name $modelName `
        --output_dir $outputDir `
        --max_length $maxLength `
        --test_size $testSize `
        --val_size $valSize `
        --epochs $epochs `
        --batch_size $batchSize `
        --lr $learningRate `
        --seed $seed

    if ($LASTEXITCODE -ne 0) {
        throw "Baseline run failed for epoch=$epochs"
    }
}

Write-Host "[BASELINE] All runs completed successfully."
