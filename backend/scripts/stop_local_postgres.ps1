[CmdletBinding()]
param(
    [string]$CondaEnvironment = "D:\DevelopmentTools\Miniconda3\envs\rag-kb",
    [string]$DataDirectory = ""
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

$pgCtl = Join-Path $CondaEnvironment "Library\bin\pg_ctl.exe"
if (-not (Test-Path -LiteralPath (Join-Path $DataDirectory "PG_VERSION") -PathType Leaf)) {
    throw "postgresql_cluster_not_initialized"
}
& $pgCtl stop -D $DataDirectory -m fast -w
if ($LASTEXITCODE -ne 0) {
    throw "postgresql_stop_failed"
}
Write-Output "postgresql_stopped"
