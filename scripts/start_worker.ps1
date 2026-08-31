param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("task", "ingestion", "projection")]
    [string]$Queue
)

# 在项目根目录启动一个显式队列的持续轮询 Worker。
$env:PYTHONUTF8 = "1"
conda run -n rag-kb --no-capture-output python -m rag_kb.worker --queue $Queue --watch --poll-seconds 2
