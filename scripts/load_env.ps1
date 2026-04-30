param(
    [string]$EnvFile = ".env"
)

if (-not (Test-Path -LiteralPath $EnvFile)) {
    throw "Env file not found: $EnvFile"
}

$lines = Get-Content -LiteralPath $EnvFile
$loadedNames = @{}
foreach ($line in $lines) {
    $trimmed = $line.Trim()
    if (-not $trimmed -or $trimmed.StartsWith("#")) {
        continue
    }

    $parts = $trimmed -split "=", 2
    if ($parts.Count -ne 2) {
        continue
    }

    $name = $parts[0].Trim()
    $value = $parts[1].Trim()

    if (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'"))) {
        $value = $value.Substring(1, $value.Length - 2)
    }

    Set-Item -Path "Env:$name" -Value $value
    $loadedNames[$name] = $true
}

# Kaggle's MCP endpoint expects KAGGLE_API_TOKEN. If this env file stores a
# KGAT-scoped token in KAGGLE_KEY, let it override any stale process-level
# KAGGLE_API_TOKEN inherited from an earlier shell.
if (-not $loadedNames.ContainsKey("KAGGLE_API_TOKEN") -and $loadedNames.ContainsKey("KAGGLE_KEY") -and $env:KAGGLE_KEY.StartsWith("KGAT_")) {
    Set-Item -Path "Env:KAGGLE_API_TOKEN" -Value $env:KAGGLE_KEY
} elseif (-not $env:KAGGLE_API_TOKEN -and $env:KAGGLE_KEY -and $env:KAGGLE_KEY.StartsWith("KGAT_")) {
    Set-Item -Path "Env:KAGGLE_API_TOKEN" -Value $env:KAGGLE_KEY
}

Write-Host "Loaded environment from $EnvFile"
