param(
    [string]$EnvFile = ".env",
    [switch]$PersistUserEnv,
    [switch]$WriteAccessToken
)

. "$PSScriptRoot/load_env.ps1" -EnvFile $EnvFile

function Resolve-KaggleApiToken {
    if ($env:KAGGLE_KEY -and $env:KAGGLE_KEY.StartsWith("KGAT_")) {
        return $env:KAGGLE_KEY
    }

    if ($env:KAGGLE_API_TOKEN) {
        return $env:KAGGLE_API_TOKEN
    }

    return $null
}

$apiToken = Resolve-KaggleApiToken
if (-not $apiToken) {
    throw "No Kaggle MCP token found. Set KAGGLE_API_TOKEN or use a KGAT_* value in KAGGLE_KEY."
}

if ($PersistUserEnv) {
    [Environment]::SetEnvironmentVariable("KAGGLE_API_TOKEN", $apiToken, "User")

    if ($env:KAGGLE_USERNAME) {
        [Environment]::SetEnvironmentVariable("KAGGLE_USERNAME", $env:KAGGLE_USERNAME, "User")
    }

    if ($env:KAGGLE_KEY) {
        [Environment]::SetEnvironmentVariable("KAGGLE_KEY", $env:KAGGLE_KEY, "User")
    }

    if ($env:KAGGLE_CONFIG_DIR) {
        [Environment]::SetEnvironmentVariable("KAGGLE_CONFIG_DIR", $env:KAGGLE_CONFIG_DIR, "User")
    }

    Write-Host "Persisted Kaggle environment variables to the current user profile."
}

if ($WriteAccessToken) {
    $kaggleDir = Join-Path $env:USERPROFILE ".kaggle"
    $accessTokenPath = Join-Path $kaggleDir "access_token"

    if (-not (Test-Path -LiteralPath $kaggleDir)) {
        New-Item -ItemType Directory -Path $kaggleDir | Out-Null
    }

    Set-Content -LiteralPath $accessTokenPath -Value $apiToken -NoNewline
    Write-Host "Wrote Kaggle access token to $accessTokenPath"
}

if (-not $PersistUserEnv -and -not $WriteAccessToken) {
    Write-Host "Nothing changed. Use -PersistUserEnv and/or -WriteAccessToken."
}
