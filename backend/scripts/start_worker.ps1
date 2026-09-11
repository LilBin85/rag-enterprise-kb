param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("task", "ingestion", "projection")]
    [string]$Queue
)

# 启动一个显式队列的持续轮询 Worker。
# 脚本通过 $PSScriptRoot 定位仓库根并切换工作目录,保持相对配置语义不变;
# try/finally 恢复调用方目录,exit 透传子进程退出码。

$repoRoot = (Get-Item (Join-Path $PSScriptRoot ".." "..")).FullName
Push-Location $repoRoot
try {
    $env:PYTHONUTF8 = "1"
    conda run -n rag-kb --no-capture-output python -m rag_kb.worker --queue $Queue --watch --poll-seconds 2
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
