[CmdletBinding()]
param(
    [string]$CondaEnvironment = "D:\DevelopmentTools\Miniconda3\envs\rag-kb",
    [string]$DataDirectory = "",
    [int]$Port = 5432
)

$ErrorActionPreference = "Stop"
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$localRoot = [IO.Path]::GetFullPath((Join-Path $projectRoot "data\local"))
if (-not $DataDirectory) {
    $DataDirectory = Join-Path $localRoot "postgresql"
}
$DataDirectory = [IO.Path]::GetFullPath($DataDirectory)
$expectedPrefix = $localRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
if (-not $DataDirectory.StartsWith($expectedPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "postgresql_data_directory_must_be_under_project_local_data"
}
if ($Port -lt 1024 -or $Port -gt 65535) {
    throw "invalid_postgresql_port"
}

$postgresBin = Join-Path $CondaEnvironment "Library\bin"
$initdb = Join-Path $postgresBin "initdb.exe"
$pgCtl = Join-Path $postgresBin "pg_ctl.exe"
$psql = Join-Path $postgresBin "psql.exe"
$createdb = Join-Path $postgresBin "createdb.exe"
foreach ($executable in @($initdb, $pgCtl, $psql, $createdb)) {
    if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
        throw "postgresql_executable_missing"
    }
}

New-Item -ItemType Directory -Path $localRoot -Force | Out-Null
$logPath = Join-Path $localRoot "postgresql.log"
$credentialsPath = Join-Path $localRoot "postgresql-credentials.env"
$clusterVersion = Join-Path $DataDirectory "PG_VERSION"
$isNewCluster = -not (Test-Path -LiteralPath $clusterVersion -PathType Leaf)

if ($isNewCluster) {
    if (Test-Path -LiteralPath $DataDirectory) {
        $existingEntry = Get-ChildItem -LiteralPath $DataDirectory -Force | Select-Object -First 1
        if ($null -ne $existingEntry) {
            throw "postgresql_data_directory_not_empty"
        }
    } else {
        New-Item -ItemType Directory -Path $DataDirectory | Out-Null
    }

    $randomBytes = [byte[]]::new(32)
    [Security.Cryptography.RandomNumberGenerator]::Fill($randomBytes)
    $password = [Convert]::ToBase64String($randomBytes).TrimEnd("=").Replace("+", "-").Replace("/", "_")
    $passwordFile = Join-Path $localRoot (".postgresql-bootstrap-{0}.pw" -f [Guid]::NewGuid().ToString("N"))
    try {
        [IO.File]::WriteAllText($passwordFile, $password + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
        & $initdb -D $DataDirectory -U "rag_kb" --encoding="UTF8" --locale="C" --auth-local="scram-sha-256" --auth-host="scram-sha-256" --pwfile=$passwordFile
        if ($LASTEXITCODE -ne 0) {
            throw "postgresql_initdb_failed"
        }
    } finally {
        Remove-Item -LiteralPath $passwordFile -Force -ErrorAction SilentlyContinue
    }

    $databaseUrl = "postgresql+psycopg://rag_kb:{0}@127.0.0.1:{1}/rag_kb" -f $password, $Port
    [IO.File]::WriteAllText(
        $credentialsPath,
        "RAG_KB_DATABASE_URL=" + $databaseUrl + [Environment]::NewLine,
        [Text.UTF8Encoding]::new($false)
    )
} elseif (-not (Test-Path -LiteralPath $credentialsPath -PathType Leaf)) {
    throw "postgresql_credentials_missing"
}

& $pgCtl status -D $DataDirectory *> $null
$isRunning = $LASTEXITCODE -eq 0
if (-not $isRunning) {
    $serverOptions = "-h 127.0.0.1 -p $Port"
    & $pgCtl start -D $DataDirectory -l $logPath -o $serverOptions -w
    if ($LASTEXITCODE -ne 0) {
        throw "postgresql_start_failed"
    }
}

$credentialLine = [IO.File]::ReadAllText($credentialsPath).Trim()
$databaseUrl = $credentialLine.Substring("RAG_KB_DATABASE_URL=".Length)
$uri = [Uri]($databaseUrl.Replace("postgresql+psycopg://", "postgresql://"))
$password = $uri.UserInfo.Split(":", 2)[1]
$previousPassword = [Environment]::GetEnvironmentVariable("PGPASSWORD", "Process")
try {
    [Environment]::SetEnvironmentVariable("PGPASSWORD", $password, "Process")
    $databaseExists = & $psql -h "127.0.0.1" -p $Port -U "rag_kb" -d "postgres" -Atqc "SELECT 1 FROM pg_database WHERE datname = 'rag_kb'"
    if ($LASTEXITCODE -ne 0) {
        throw "postgresql_probe_failed"
    }
    if ($databaseExists -ne "1") {
        & $createdb -h "127.0.0.1" -p $Port -U "rag_kb" "rag_kb"
        if ($LASTEXITCODE -ne 0) {
            throw "postgresql_database_creation_failed"
        }
    }
    $serverVersion = & $psql -h "127.0.0.1" -p $Port -U "rag_kb" -d "rag_kb" -Atqc "SHOW server_version"
    if ($LASTEXITCODE -ne 0) {
        throw "postgresql_version_probe_failed"
    }
} finally {
    [Environment]::SetEnvironmentVariable("PGPASSWORD", $previousPassword, "Process")
}

Write-Output ("postgresql_ready version={0} host=127.0.0.1 port={1}" -f $serverVersion, $Port)
Write-Output ("data_directory={0}" -f $DataDirectory)
Write-Output ("credentials_file={0}" -f $credentialsPath)
