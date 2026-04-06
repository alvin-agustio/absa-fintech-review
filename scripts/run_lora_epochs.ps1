$ErrorActionPreference = 'Stop'

$python = ".\.venv\Scripts\python.exe"
$inputCsv = "data/processed/dataset_absa_50k_v2_intersection.csv"
$modelName = "indobenchmark/indobert-base-p1"
$maxLength = 128
$testSize = 0.2
$valSize = 0.1
$batchSize = 8
$learningRate = 2e-4
$seed = 42
$loraR = 16
$loraAlpha = 32
$loraDropout = 0.1
$maxEpoch = 15
$outputDir = "models/lora/epoch_$maxEpoch"

Write-Host "[LORA] Running max_epoch=$maxEpoch -> $outputDir"
& $python src/training/train_lora.py `
    --input_csv $inputCsv `
    --model_name $modelName `
    --output_dir $outputDir `
    --max_length $maxLength `
    --test_size $testSize `
    --val_size $valSize `
    --epochs $maxEpoch `
    --batch_size $batchSize `
    --lr $learningRate `
    --seed $seed `
    --lora_r $loraR `
    --lora_alpha $loraAlpha `
    --lora_dropout $loraDropout

if ($LASTEXITCODE -ne 0) {
    throw "LoRA run failed for max_epoch=$maxEpoch"
}

Write-Host "[LORA] Run completed successfully."
