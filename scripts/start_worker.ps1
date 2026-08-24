# 在项目根目录启动持续轮询的异步摄取工作器。
$env:PYTHONUTF8 = "1"
conda run -n rag-kb --no-capture-output python -m rag_kb.worker --watch --poll-seconds 2
