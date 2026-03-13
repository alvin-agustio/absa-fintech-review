$ErrorActionPreference = 'Stop'

$python = ".\.venv\Scripts\python.exe"
$inputCsv = "data/processed/dataset_absa_50k_v2_intersection.csv"
$modelName = "indobenchmark/indobert-base-p1"
$maxLength = 128
$testSize = 0.2
$valSize = 0.1
$batchSize = 16
$learningRate = 2e-4
$seed = 42
$loraR = 16
$loraAlpha = 32
$loraDropout = 0.1
$epochsList = @(3, 5)

foreach ($epochs in $epochsList) {
    $outputDir = "models/lora/epoch_$epochs"
    Write-Host "[LORA] Running epoch=$epochs -> $outputDir"
    & $python train_lora.py `
        --input_csv $inputCsv `
        --model_name $modelName `
        --output_dir $outputDir `
        --max_length $maxLength `
        --test_size $testSize `
        --val_size $valSize `
        --epochs $epochs `
        --batch_size $batchSize `
        --lr $learningRate `
        --seed $seed `
        --lora_r $loraR `
        --lora_alpha $loraAlpha `
        --lora_dropout $loraDropout

    if ($LASTEXITCODE -ne 0) {
        throw "LoRA run failed for epoch=$epochs"
    }
}

Write-Host "[LORA] All runs completed successfully."
