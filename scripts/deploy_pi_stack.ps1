param(
    [string]$RemoteHost = "bura@100.111.13.58",
    [string]$RemoteRoot = "/home/bura/projects/final_year"
)

$ErrorActionPreference = "Stop"

function Invoke-CheckedNative {
    param(
        [string]$Label,
        [scriptblock]$Command
    )

    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

$files = @(
    "src/data/cities.py",
    "src/ingestion/india_aq.py",
    "deployment/pi_airflow/dags/aq_city_factory.py",
    "deployment/pi_airflow/docker-compose.yml",
    "deployment/pi_airflow/docker-compose-fixed.yml",
    "scripts/migrate_aggregates.py",
    "scripts/load_forecasts_to_db.py",
    "web/dashboard/app/api/observations/route.js",
    "web/dashboard/app/api/observations/[citySlug]/route.js",
    "web/dashboard/app/api/predictions/[citySlug]/route.js",
    "web/dashboard/app/lib/cities.js",
    "web/dashboard/lib/cities.js",
    "web/dashboard/components/CitiesGrid.jsx",
    "web/dashboard/components/CityDashboard.jsx",
    "web/dashboard/components/PredictionDashboard.jsx"
)

$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("final_year_pi_deploy_" + [System.Guid]::NewGuid().ToString("N"))
$archivePath = Join-Path $tempRoot "final_year_patch.tar"
$fileListPath = Join-Path $tempRoot "files.txt"

New-Item -ItemType Directory -Path $tempRoot | Out-Null

try {
    $files | Set-Content -LiteralPath $fileListPath

    Write-Host "Creating deploy archive"
    Invoke-CheckedNative "tar archive creation" { tar -cf $archivePath -T $fileListPath }

    Write-Host "Copying deploy archive"
    Invoke-CheckedNative "scp deploy archive" { scp $archivePath "$RemoteHost`:/tmp/final_year_patch.tar" }

    Write-Host "Preparing remote project paths"
    Invoke-CheckedNative "remote path preparation" { ssh $RemoteHost "sudo -n mkdir -p $RemoteRoot/deployment/pi_airflow/dags && sudo -n chown -R bura:bura $RemoteRoot/deployment/pi_airflow" }

    Write-Host "Extracting updated files"
    Invoke-CheckedNative "remote archive extraction" { ssh $RemoteHost "tar -xf /tmp/final_year_patch.tar -C $RemoteRoot && rm -f /tmp/final_year_patch.tar" }
}
finally {
    if (Test-Path -LiteralPath $tempRoot) {
        Remove-Item -LiteralPath $tempRoot -Recurse -Force
    }
}

Write-Host "Recreating Pi stack"
Invoke-CheckedNative "remote compose restart" { ssh $RemoteHost "docker rm -f final-year-dashboard >/dev/null 2>&1 || true; cd $RemoteRoot/deployment/pi_airflow && FINAL_YEAR_ROOT=$RemoteRoot docker compose up -d --build --force-recreate airflow-init airflow-webserver airflow-scheduler airflow-triggerer dashboard" }

Write-Host "Running warehouse migrations"
Invoke-CheckedNative "remote warehouse migrations" { ssh $RemoteHost "cd $RemoteRoot/deployment/pi_airflow && FINAL_YEAR_ROOT=$RemoteRoot docker compose exec -T airflow-webserver python /opt/final_year/scripts/migrate_aggregates.py" }
