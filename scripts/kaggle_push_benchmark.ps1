param(
    [string]$EnvFile = ".env",
    [switch]$VersionDataset,
    [switch]$PushKernel,
    [string]$KernelPath = "kaggle/aqi_gpu_benchmark"
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

$kaggle = Resolve-KaggleCli
$classicConfigDir = if ($env:KAGGLE_CONFIG_DIR) { $env:KAGGLE_CONFIG_DIR } else { Join-Path $env:USERPROFILE ".kaggle" }
$classicJson = Join-Path $classicConfigDir "kaggle.json"

function Invoke-KaggleClassic {
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command
    )

    $accessToken = Join-Path $classicConfigDir "access_token"
    $hiddenToken = Join-Path $classicConfigDir "access_token.codex_hold"
    $movedToken = $false

    if (-not (Test-Path -LiteralPath $classicJson)) {
        & $Command
        return
    }

    if (Test-Path -LiteralPath $hiddenToken) {
        Remove-Item -LiteralPath $hiddenToken -Force
    }

    if (Test-Path -LiteralPath $accessToken) {
        Move-Item -LiteralPath $accessToken -Destination $hiddenToken
        $movedToken = $true
    }

    try {
        $env:KAGGLE_CONFIG_DIR = $classicConfigDir
        $env:KAGGLE_API_TOKEN = $null
        $env:KAGGLE_USERNAME = $null
        $env:KAGGLE_KEY = $null
        Write-Host "Using classic Kaggle CLI credentials from $classicJson"
        & $Command
    } finally {
        if ($movedToken -and (Test-Path -LiteralPath $hiddenToken)) {
            Move-Item -LiteralPath $hiddenToken -Destination $accessToken
        }
    }
}

if ($VersionDataset) {
    Write-Host "Versioning Kaggle dataset bundle..."
    & $kaggle datasets version -p data/kaggle_dataset -m "Update 3-city AQI benchmark bundle" -r zip
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Dataset version failed; trying initial dataset create..."
        & $kaggle datasets create -p data/kaggle_dataset -r zip
    }
}

if ($PushKernel) {
    Write-Host "Pushing Kaggle GPU kernel..."
    Invoke-KaggleClassic { & $kaggle kernels push -p $KernelPath -t 43200 }
}

if (-not $VersionDataset -and -not $PushKernel) {
    Write-Host "Nothing selected. Use -VersionDataset and/or -PushKernel."
}
