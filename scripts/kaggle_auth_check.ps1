param(
    [string]$EnvFile = ".env"
)

. "$PSScriptRoot/load_env.ps1" -EnvFile $EnvFile

function Resolve-KaggleCli {
    $candidates = @(
        "$env:USERPROFILE\miniconda3\envs\dl-env\Scripts\kaggle.exe",
        "$env:USERPROFILE\miniconda3\Scripts\kaggle.exe",
        "$env:USERPROFILE\anaconda3\Scripts\kaggle.exe",
        "$env:USERPROFILE\AppData\Local\miniconda3\Scripts\kaggle.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }
    return "kaggle"
}

if (-not $env:KAGGLE_API_TOKEN -and (-not $env:KAGGLE_USERNAME -or -not $env:KAGGLE_KEY)) {
    throw "Kaggle credentials are missing from $EnvFile"
}

$displayToken = if ($env:KAGGLE_API_TOKEN) { $env:KAGGLE_API_TOKEN } else { $env:KAGGLE_KEY }
$maskedKey = if ($displayToken.Length -gt 8) {
    $displayToken.Substring(0, 4) + "..." + $displayToken.Substring($displayToken.Length - 4)
} else {
    "***"
}

if ($env:KAGGLE_USERNAME) {
    Write-Host "Kaggle username: $($env:KAGGLE_USERNAME)"
}
Write-Host "Kaggle token: $maskedKey"
Write-Host "Checking Kaggle CLI authentication..."
$kaggle = Resolve-KaggleCli
$classicConfigDir = if ($env:KAGGLE_CONFIG_DIR) { $env:KAGGLE_CONFIG_DIR } else { Join-Path $env:USERPROFILE ".kaggle" }
$classicJson = Join-Path $classicConfigDir "kaggle.json"
if (Test-Path -LiteralPath $classicJson) {
    $env:KAGGLE_CONFIG_DIR = $classicConfigDir
    $env:KAGGLE_API_TOKEN = $null
    $env:KAGGLE_USERNAME = $null
    $env:KAGGLE_KEY = $null
    Write-Host "Using classic Kaggle CLI credentials from $classicJson"
}
& $kaggle config view
